#!/usr/bin/env python3
"""Push-button start for the robot: press the button on the Pi and main.py runs.

    nothing pressed yet         yellow LED blips once every 2 s = "booted, ready, waiting"
    press                       main.py starts -- the mission begins
    press again while it runs   main.py is stopped exactly as Ctrl-C would stop it
                                (motors off, LEDs off, map saved to runs/), then it
                                goes back to waiting for the next press
    main.py crashes             red LED flashes 3 times, then back to waiting

    python3 button_start.py                     # wait for the button, then run main.py
    python3 button_start.py -- --test-mode      # anything after -- is passed to main.py
    ./install_button_service.sh                 # run this automatically every boot

WIRING: one leg of the push button to GPIO 20 (physical pin 38), the other leg to GND
(physical pin 39, right next to it, or pin 34). No resistor needed: the Pi's internal
pull-up holds the pin HIGH, and pressing the button shorts it to GND (LOW).
GPIO 20 is clear of everything main.py uses -- I2C 2/3, LEDs 13/19/26, sonars 22/27,
23/24, 5/6.

Why a separate launcher instead of a loop inside main.py: main.py is a long-lived
mission with a lot of state (the map, odometry, the camera). Starting it fresh for every
run guarantees a clean start, and stopping it with SIGINT goes through main.py's own
`finally:` block, so the motors are always stopped the same, tested way.
"""

import os
import signal
import subprocess
import sys
import threading
import time

# ===========================================================================
# CONFIG
# ===========================================================================
BUTTON_BCM = 20             # button between this pin and GND (physical pin 38)
READY_LED_BCM = 19          # yellow (same as main.py): blips while waiting. None = off
FAULT_LED_BCM = 26          # red (same as main.py): flashes if main.py crashes. None = off
BOUNCE_S = 0.1              # contact bounce shorter than this is ignored
STOP_ARM_S = 1.0            # a press this soon after starting is ignored (double-tap)
STOP_GRACE_S = 10.0         # after "Ctrl-C", main.py gets this long to tidy up...
TERM_GRACE_S = 3.0          # ...then SIGTERM, this long, then SIGKILL

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.py")


# ===========================================================================
# helpers
# ===========================================================================
def log(msg):
    print(f"[button {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main_args(extra):
    """The arguments main.py gets.

    Started at boot there's no screen: an OpenCV window with no display can kill the
    process outright (Qt aborts, it doesn't raise), so --no-display is added whenever
    there's no DISPLAY. And with no terminal, the self-rewriting status line would fill
    the log file 20 times a second, so --no-status is added too. The event lines --
    state changes, victims, warnings -- still print and land in the log.
    """
    args = list(extra)
    if not os.environ.get("DISPLAY") and "--no-display" not in args:
        args.append("--no-display")
    if not sys.stdout.isatty() and "--no-status" not in args:
        args.append("--no-status")
    return args


def make_led(pin):
    """An LED on `pin`, or None if it's turned off in CONFIG or can't be claimed."""
    if pin is None:
        return None
    try:
        from gpiozero import LED
        return LED(pin)
    except Exception as exc:                          # noqa: BLE001
        log(f"LED on GPIO {pin} unavailable -- {type(exc).__name__}: {exc}")
        return None


def release(led):
    """Give the pin back, so main.py can claim it for its own LEDs."""
    if led is not None:
        try:
            led.close()
        except Exception:                             # noqa: BLE001
            pass


def start_main(main_path, args):
    """Launch main.py with the same Python (so a venv carries over), from its folder.

    start_new_session: a Ctrl-C typed at this launcher's terminal reaches only the
    launcher, which then stops main.py deliberately -- not both at once.
    """
    cmd = [sys.executable, "-u", main_path] + args
    log("starting: main.py " + " ".join(args))
    return subprocess.Popen(cmd, cwd=os.path.dirname(main_path), start_new_session=True)


def stop_main(proc):
    """Stop main.py, politest first. Returns its exit code.

    SIGINT is exactly what Ctrl-C sends: main.py gets a KeyboardInterrupt and runs its
    finally: block (motors stop, LEDs off, camera released, map saved). Only if it
    hangs does it get SIGTERM, then SIGKILL. Even then the motors don't run on: the
    motor board's own 0.5 s watchdog cuts them when the commands stop arriving.
    """
    for sig, grace in ((signal.SIGINT, STOP_GRACE_S), (signal.SIGTERM, TERM_GRACE_S),
                       (signal.SIGKILL, None)):
        if proc.poll() is not None:
            break
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            break
        if grace is None:
            proc.wait()
            break
        try:
            proc.wait(timeout=grace)
            break
        except subprocess.TimeoutExpired:
            log(f"main.py still running {grace:g}s after {signal.Signals(sig).name} -- escalating")
    return proc.returncode


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


# ===========================================================================
# the loop
# ===========================================================================
def run(extra=(), main_path=MAIN, button_pin=BUTTON_BCM, max_runs=None):
    """Wait for a press, run main.py, stop it on the next press; forever.

    max_runs is only for testing -- it returns after that many runs.
    """
    from gpiozero import Button

    # Our own handlers (rather than "ignore") for both signals: a program started with
    # exec() gets caught signals reset to default, so main.py always gets a working
    # Ctrl-C. SIGTERM is what systemd sends on `systemctl stop`.
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)

    button = Button(button_pin, pull_up=True, bounce_time=BOUNCE_S)
    # React to the PRESS EDGE (HIGH -> LOW), not to "is it down right now": holding the
    # button after starting can't then count as a second press.
    pressed = threading.Event()
    button.when_pressed = pressed.set

    args = main_args(extra)
    proc = None
    runs = 0
    log(f"ready -- press the button on GPIO {button_pin} to start main.py")
    try:
        while max_runs is None or runs < max_runs:
            # --- waiting ---
            ready = make_led(READY_LED_BCM)
            if ready is not None:
                ready.blink(on_time=0.1, off_time=1.9)
            pressed.clear()
            while not pressed.wait(0.2):
                pass
            release(ready)                       # main.py needs that pin for itself

            # --- running ---
            log("button pressed -- go")
            proc = start_main(main_path, args)
            started = time.monotonic()
            pressed.clear()
            stopped_by_us = False
            while proc.poll() is None:
                if pressed.wait(0.1):
                    pressed.clear()
                    if time.monotonic() - started < STOP_ARM_S:
                        continue                 # a double-tap, not a stop
                    log("button pressed -- stopping main.py")
                    stop_main(proc)
                    stopped_by_us = True
            code = proc.returncode
            proc = None
            runs += 1

            if stopped_by_us or code == 0:
                log(f"main.py finished (exit {code})")
            else:
                log(f"main.py EXITED WITH AN ERROR (exit {code}) -- see the lines above")
                fault = make_led(FAULT_LED_BCM)
                if fault is not None:
                    fault.blink(on_time=0.15, off_time=0.15, n=3, background=False)
                release(fault)
            log("ready -- press to run again")
    except KeyboardInterrupt:
        log("launcher stopping")
    finally:
        if proc is not None and proc.poll() is None:
            log("stopping main.py first")
            stop_main(proc)
        button.close()


def main():
    extra = sys.argv[1:]
    if extra and extra[0] == "--":
        extra = extra[1:]
    if not os.path.exists(MAIN):
        raise SystemExit(f"main.py not found next to this file ({MAIN})")
    run(extra)


if __name__ == "__main__":
    main()
