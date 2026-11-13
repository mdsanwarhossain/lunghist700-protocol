"""Run a long CPU job under a conservative thermal governor.

The study's encoding passes saturate every core for hours. On a laptop that
drives the package into thermal throttling, where the job runs *slower* than it
would at a lower clock, and an overheating machine that cuts power mid-pass
loses the pass. This supervisor polls the package temperature and suspends the
child with SIGSTOP once it crosses ``--hot``, resuming it at ``--cool``. The
child is a checkpointed encoder, so a suspension costs wall-clock time and
nothing else.

An earlier run of this study lost the machine to a hard power cut after 5.5
hours of sustained load, so the governor defends against heat *soak* and not
just against instantaneous temperature:

* it refuses to start until the package is already cool and the machine idle,
  so a pass never begins on top of somebody else's load;
* every pause lasts at least ``--min-rest`` seconds. The package sensor falls
  within seconds but the heatsink does not, and resuming on the sensor alone is
  what let the chassis soak;
* it duty-cycles, resting ``--rest`` seconds after every ``--work`` seconds of
  running even when the package never crosses ``--hot``, which bounds heat soak
  over a pass lasting hours;
* crossing ``--panic`` forces a long cooldown, covering the case where a spike
  outruns the poll interval;
* it suspends the job whenever the lid is shut, because a closed lid blocks the
  exhaust and was a direct cause of the earlier shutdown.

Usage: thermal_run.py [--hot 85] [--cool 72] [--threads 12] -- <cmd> ...
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ZONES = sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
LID = sorted(Path("/proc/acpi/button/lid").glob("*/state"))

# Only true CPU package/core sensors. `acpitz` was previously included and is
# the reason an idle machine looked hot: measured over 15 idle seconds it had a
# median of 69C and a maximum of 80C while the cores sat at 54C. It is a slow
# chassis zone, not the thing a CPU thermal limit should track, and taking a
# max across both let it dominate the controller.
_HOT_ZONES = ("x86_pkg_temp", "coretemp", "TCPU")

# These sensors are extremely spiky: on the same idle machine x86_pkg_temp had
# a median of 54C and a maximum of 97C. Feeding instantaneous readings to a
# hysteresis controller makes it suspend on a transient and then fail to see a
# reading below the resume threshold, which stalled passes for tens of minutes.
# A short rolling median keeps the controller responsive to real heating while
# ignoring single-sample excursions.
_SMOOTH_N = 5
_history: list[float] = []


def _instant_temp() -> float:
    """Hottest instantaneous reading across the CPU package / core zones."""
    best = 0.0
    for z in ZONES:
        try:
            if not (z / "type").read_text().strip().startswith(_HOT_ZONES):
                continue
            best = max(best, int((z / "temp").read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return best


def package_temp() -> float:
    """Rolling median of the CPU package temperature, in degrees C."""
    t = _instant_temp()
    if t <= 0:
        return 0.0
    _history.append(t)
    del _history[:-_SMOOTH_N]
    return sorted(_history)[len(_history) // 2]


def proc_state(pid: int) -> str:
    """Scheduler state letter from /proc, or "" if the process is gone.

    The supervisor's own record of whether it has suspended the child can
    drift from the truth -- an observed pass sat stopped for 27 minutes on a
    54C machine with every resume condition satisfied. Rather than trust the
    bookkeeping, each tick compares it against the kernel's view and corrects
    whichever side is wrong.
    """
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        return data[data.rindex(")") + 2]
    except (OSError, ValueError, IndexError):
        return ""


def lid_closed() -> bool:
    """True only when the firmware positively reports a shut lid."""
    for s in LID:
        try:
            if "closed" in s.read_text().lower():
                return True
        except OSError:
            continue
    return False


def load1() -> float:
    return os.getloadavg()[0]


def wait_until_ready(cool: float, max_load: float, poll: float,
                     timeout: float) -> None:
    """Block until the machine is cool, idle and open-lidded."""
    t0 = time.time()
    while True:
        t, ld, shut = package_temp(), load1(), lid_closed()
        if t <= cool and ld <= max_load and not shut:
            print(f"[thermal_run] ready: {t:.0f}C, load {ld:.1f}", flush=True)
            return
        if time.time() - t0 > timeout:
            print(f"[thermal_run] WARNING: still {t:.0f}C / load {ld:.1f} after "
                  f"{timeout:.0f}s wait; starting anyway", flush=True)
            return
        why = []
        if t > cool:
            why.append(f"{t:.0f}C > {cool:.0f}C")
        if ld > max_load:
            why.append(f"load {ld:.1f} > {max_load:.1f}")
        if shut:
            why.append("lid shut")
        print(f"[thermal_run] waiting to start ({', '.join(why)}), "
              f"{time.time()-t0:.0f}s elapsed", flush=True)
        time.sleep(max(poll, 15.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=float, default=85.0, help="suspend above this (C)")
    ap.add_argument("--cool", type=float, default=72.0, help="resume below this (C)")
    ap.add_argument("--panic", type=float, default=95.0,
                    help="above this, rest for --panic-rest regardless of cooling")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--min-rest", type=float, default=60.0,
                    help="every pause lasts at least this long (s)")
    ap.add_argument("--panic-rest", type=float, default=300.0,
                    help="minimum pause after a --panic crossing (s)")
    # The duty cycle guards against heat *soak*, which builds over hours and is
    # what cost us the machine. Measured at 12 threads the package sits in the
    # 50-65C range, roughly 20C below --hot, so the cycle was relaxed from
    # 240s/90s (27% overhead) to 480s/60s (12.5%). The temperature, panic and
    # lid guards are untouched and still interrupt a work period at any point.
    ap.add_argument("--work", type=float, default=480.0,
                    help="duty cycle: rest after this much running (s); 0 disables")
    ap.add_argument("--rest", type=float, default=60.0,
                    help="duty-cycle rest length (s)")
    ap.add_argument("--max-load", type=float, default=6.0,
                    help="refuse to start while 1-min load exceeds this")
    ap.add_argument("--wait-timeout", type=float, default=1800.0,
                    help="give up waiting for a cool machine after this long (s)")
    ap.add_argument("--no-wait", action="store_true", help="skip the pre-flight gate")
    ap.add_argument("--gate-only", action="store_true",
                    help="just wait for a cool, idle machine and exit")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if a.gate_only:
        wait_until_ready(a.cool, a.max_load, a.poll, a.wait_timeout)
        return 0
    if not cmd:
        ap.error("no command given")

    env = dict(os.environ,
               OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               OPENBLAS_NUM_THREADS=str(a.threads), NUMEXPR_NUM_THREADS=str(a.threads),
               TORCH_NUM_THREADS=str(a.threads), TOKENIZERS_PARALLELISM="false")

    if not a.no_wait:
        wait_until_ready(a.cool, a.max_load, a.poll, a.wait_timeout)

    t0 = time.time()
    print(f"[thermal_run] {a.threads} threads, suspend>{a.hot}C resume<{a.cool}C "
          f"(min rest {a.min_rest:.0f}s, duty {a.work:.0f}s/{a.rest:.0f}s), "
          f"start {package_temp():.0f}C :: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, env=env, start_new_session=True,
                            preexec_fn=lambda: os.nice(a.nice))

    paused = False
    pause_total, n_pauses = 0.0, 0
    p0 = 0.0            # when the current pause began
    rest_needed = 0.0   # how long the current pause must last
    reason = ""
    ran_since_rest = 0.0
    last = time.time()
    peak = package_temp()

    def pause(why: str, hold: float) -> None:
        nonlocal paused, p0, n_pauses, rest_needed, reason, ran_since_rest
        os.killpg(proc.pid, signal.SIGSTOP)
        paused, p0, rest_needed, reason = True, time.time(), hold, why
        n_pauses += 1
        ran_since_rest = 0.0
        print(f"[thermal_run] {why}: suspended (pause #{n_pauses}, "
              f"hold >= {hold:.0f}s)", flush=True)

    try:
        while proc.poll() is None:
            time.sleep(a.poll)
            now = time.time()
            dt, last = now - last, now
            t = package_temp()
            peak = max(peak, t)

            # Reconcile belief with reality before acting on either.
            st = proc_state(proc.pid)
            if not paused and st == "T":
                os.killpg(proc.pid, signal.SIGCONT)
                print("[thermal_run] WARNING: child was stopped while the "
                      "supervisor believed it running; resumed", flush=True)
            elif paused and st and st != "T":
                print("[thermal_run] WARNING: child is running while the "
                      "supervisor believed it paused; belief corrected",
                      flush=True)
                paused = False
                pause_total += now - p0

            if not paused:
                ran_since_rest += dt
                if lid_closed():
                    pause("lid shut", a.min_rest)
                elif t >= a.panic:
                    pause(f"{t:.0f}C >= panic {a.panic:.0f}C", a.panic_rest)
                elif t >= a.hot:
                    pause(f"{t:.0f}C >= {a.hot:.0f}C", a.min_rest)
                elif a.work > 0 and ran_since_rest >= a.work:
                    pause(f"duty cycle after {ran_since_rest:.0f}s", a.rest)
                continue

            held = now - p0
            if held < rest_needed or t > a.cool or lid_closed():
                continue
            os.killpg(proc.pid, signal.SIGCONT)
            paused = False
            pause_total += held
            print(f"[thermal_run] {t:.0f}C <= {a.cool:.0f}C: resumed after "
                  f"{held:.0f}s ({reason}; total cooling {pause_total:.0f}s)",
                  flush=True)
    except KeyboardInterrupt:
        if paused:
            os.killpg(proc.pid, signal.SIGCONT)
        os.killpg(proc.pid, signal.SIGTERM)
        return 130

    rc = proc.returncode
    print(f"[thermal_run] exit {rc} after {time.time()-t0:.0f}s "
          f"({n_pauses} pauses, {pause_total:.0f}s cooling, peak {peak:.0f}C), "
          f"now {package_temp():.0f}C", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
