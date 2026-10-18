#!/usr/bin/env python3
"""Status LED bring-up: flash the three LEDs in a line.

Self-contained -- no other files needed, no camera, no motors, no I2C.

    python3 led_test.py                 # chase green -> yellow -> red, forever
    python3 led_test.py --interval 0.1  # faster
    python3 led_test.py --reverse       # red -> yellow -> green
    python3 led_test.py --bounce        # sweep down the line and back
    python3 led_test.py --identify      # one at a time, naming each -- checks wiring
    python3 led_test.py --all           # all three on, to check brightness/current
    python3 led_test.py --off           # everything off, then exit

Use --identify first. It lights one LED at a time and prints which colour SHOULD be
lit, which is the only way to catch a green LED wired to the yellow pin -- every
other mode looks perfectly fine with two LEDs swapped.

Ctrl-C at any point leaves all three off.

WIRING: LED anode -> GPIO through a resistor (330R is fine for 3.3 V), cathode -> GND.
Pi GPIO sources ~16 mA per pin comfortably; don't drive these without a resistor.
"""

import argparse
import time

# BCM numbering, matching Robot_Code/main.py
LED_PINS = [
    ("green",  16),   # victim detected
    ("yellow", 20),   # searching / exploring
    ("red",    21),   # returning to base
]


def build_leds():
    from gpiozero import LED
    leds = []
    for name, pin in LED_PINS:
        try:
            leds.append((name, pin, LED(pin)))
        except Exception as exc:                   # noqa: BLE001
            raise SystemExit(
                f"could not claim GPIO {pin} for the {name} LED -- "
                f"{type(exc).__name__}: {exc}\n"
                "Something else may already hold the pin (another script still "
                "running?), or the pin number is wrong."
            )
    print("LEDs: " + ", ".join(f"{n}=GPIO{p}" for n, p, _ in leds))
    return leds


def all_off(leds):
    for _, _, led in leds:
        try:
            led.off()
        except Exception:                          # noqa: BLE001
            pass


def chase(leds, interval, reverse=False, bounce=False):
    """Light each LED in turn, one at a time, down the line."""
    order = list(range(len(leds)))
    if reverse:
        order.reverse()
    if bounce:
        order = order + order[-2:0:-1]             # e.g. 0,1,2,1 -- ends don't repeat
    print("chasing " + " -> ".join(leds[i][0] for i in order) + "   (Ctrl-C to stop)")
    while True:
        for i in order:
            all_off(leds)
            leds[i][2].on()
            time.sleep(interval)


def identify(leds, seconds):
    """One at a time, announced -- the only mode that catches swapped wiring."""
    print("Watch the robot. Each LED lights for "
          f"{seconds:.0f}s; check the colour matches what's printed.\n")
    for name, pin, led in leds:
        all_off(leds)
        led.on()
        print(f"  GPIO {pin:>2} should now be lit  -->  {name.upper()}")
        time.sleep(seconds)
    all_off(leds)
    print("\nIf any colour didn't match, swap those two entries in LED_PINS here "
          "AND in main.py.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--interval", type=float, default=0.25, help="seconds per LED")
    ap.add_argument("--reverse", action="store_true", help="chase the other way")
    ap.add_argument("--bounce", action="store_true", help="sweep down the line and back")
    ap.add_argument("--identify", action="store_true", help="one at a time, named")
    ap.add_argument("--all", action="store_true", help="all three on, then exit")
    ap.add_argument("--off", action="store_true", help="all off, then exit")
    args = ap.parse_args()

    leds = build_leds()
    try:
        if args.off:
            all_off(leds)
            print("all off")
        elif args.all:
            for _, _, led in leds:
                led.on()
            print("all three on -- Ctrl-C to stop")
            while True:
                time.sleep(0.5)
        elif args.identify:
            identify(leds, max(1.0, args.interval * 8))
        else:
            chase(leds, args.interval, reverse=args.reverse, bounce=args.bounce)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        all_off(leds)


if __name__ == "__main__":
    main()
