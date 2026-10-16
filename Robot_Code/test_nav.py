"""Nav logic checks that need no camera, no motors, no sonar and no Pi.

    python3 test_nav.py

Fakes all four subsystems and drives Nav.step() through scripted detections, so the
parts that fail quietly get checked on a laptop: detection debounce, the bearing sign
flip between vision and drive, the 10 cm stop, when the sonar may and may not be
believed, and the refusals -- no ranging, no echo, wall-marker geometry.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as M
import maze as MZ
import motors as MOT

# Most of this file exercises the legacy path: differential steering while driving and no
# pivot trim at the start of a leg. The shipped defaults are the other way round (the real
# chassis can't steer by a speed difference -- see STEER_WHILE_DRIVING), and section 27
# tests those. The defaults are saved here and restored there.
_DEFAULT_STEER, _DEFAULT_TRIM = M.STEER_WHILE_DRIVING, M.TRIM_ENABLED
_DEFAULT_MINRATE = M.STEER_MIN_RATE
M.STEER_WHILE_DRIVING, M.TRIM_ENABLED = True, False
M.STEER_MIN_RATE = 0.0            # plain proportional steering, as these tests were written


# --------------------------------------------------------------------- fakes
class FakeDrive:
    """Records commands. `ticks` is writable so a test can simulate track slip."""
    def __init__(self, ticks=None):
        self.last = (0.0, 0.0); self.board = None; self.ticks = ticks
    def set_velocity(self, v, w): self.last = (v, w)
    def read_encoders(self): return self.ticks
    def stop(self): self.last = (0.0, 0.0)


class FakeLeds:
    def __init__(self): self.g = None; self.y = None; self.r = None; self.ylog = []
    def green(self, on): self.g = on
    def yellow(self, on): self.y = on; self.ylog.append(on)
    def red(self, on): self.r = on
    def all_off(self): self.g = self.y = self.r = False


class FakeVision:
    """Replays a scripted list of Victim|None, one per look()."""
    def __init__(self, script, geometry_ok=True):
        self.script = list(script); self.geometry_ok = geometry_ok
    def look(self):
        return self.script.pop(0) if self.script else None


class FakeSonar:
    """Front sonar only; .front is writable mid-test to simulate closing in."""
    def __init__(self, front=None, available=True):
        self.front = front; self.available = available
        self.sensors = {"front": 1} if available else {}
        self._n = 0
    def update(self): self._n += 1
    def get(self, name): return self.front if name == "front" else None
    def since(self, name, t):
        return (True, self.front) if name in self.sensors else (False, None)
    def stamp(self, name): return self._n if name in self.sensors else None


NO_SONAR = FakeSonar(available=False)


def V(dist=None, bearing=0.0):
    """A victim detection. dist is the CAMERA estimate; None = camera can't range it."""
    return M.Victim(bearing, dist, (100, 100, 50, 50), 2500)


fails = []
def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        fails.append(name)

def run(nav, ticks):
    for _ in range(ticks):
        nav.step()


def A(drive, vision, leds, sonar):
    """A Nav already in APPROACH -- for testing the steer-by-eye controller itself.
    From SEARCH, a victim it can range goes to SEEK instead (see 2)."""
    n = M.Nav(drive, vision, leds, sonar)
    n._enter(M.APPROACH)
    return n


# --------------------------------------------------------------------- tests
print("1) EXPLORING: reads the walls, then drives one cell, yellow on")
d, l = FakeDrive(ticks=(0, 0)), FakeLeds()
n = M.Nav(d, FakeVision([None] * 5), l, FakeSonar(front=1.0))
run(n, 3)
check("starts in the base cell, facing north", n.cell == M.BASE_CELL and n.facing == 0)
check("base cell mapped before moving", M.BASE_CELL in n.map.visited)
check("open ahead is recorded as open", n.map.wall(M.BASE_CELL, 0) is False)
check("driving forward at SEARCH_SPEED", d.last[0] == M.SEARCH_SPEED)
check("dead straight with no encoder error", abs(d.last[1]) < 1e-9)
check("yellow on, green and red off", l.y is True and l.g is False and l.r is False)

print("1a) the encoders hold it straight -- neither track falls behind")
d.ticks = (500, 500)                       # both tracks equal
n.step()
check("tracks equal -> no correction", abs(d.last[1]) < 1e-9)
d.ticks = (700, 500)                       # left ran ahead: veered RIGHT
n.step()
check("veering right -> turns LEFT to correct", d.last[1] > 0)
d.ticks = (700, 900)                       # right caught up and passed: veered LEFT
n.step()
check("veering left -> turns RIGHT to correct", d.last[1] < 0)
d.ticks = (99999, 900)                     # absurd error
n.step()
check("correction is capped", abs(d.last[1]) <= M.MAX_STEER_RATE)
check("still driving forward while correcting", d.last[0] > 0)

print("1b) no encoders -> still drives, open loop, rather than crashing")
d = FakeDrive(ticks=None)
run(M.Nav(d, FakeVision([None] * 3), FakeLeds(), FakeSonar(front=1.0)), 3)
check("still drives forward", d.last[0] == M.SEARCH_SPEED and abs(d.last[1]) < 1e-9)

print("1c) a wall ahead: turns on the spot to an open side")
d = FakeDrive(ticks=(0, 0))
n = M.Nav(d, FakeVision([None] * 5), FakeLeds(), FakeSonar(front=0.04))
run(n, 3)
check("wall ahead is mapped", n.map.wall(M.BASE_CELL, 0) is True)
check("stops driving and spins", d.last[0] == 0.0 and d.last[1] != 0.0)
check("turns RIGHT, the only way out of the SW corner", d.last[1] < 0)
check("hard enough to pivot a tracked chassis", abs(d.last[1]) >= M.MIN_TURN_RATE_PIVOT)
_l, _r = MOT.wheel_speeds(*d.last)
check("a tank turn: tracks counter-rotate", _l * _r < 0 and abs(_l + _r) < 1e-9)

print("1d) centring: off to one side, it aims back to the middle")
class _Sides(FakeSonar):
    def __init__(self, left, right):
        super().__init__(front=1.0); self.l, self.r = left, right
        self.sensors = {"front": 1, "left": 1, "right": 1}
    def get(self, name):
        return {"front": self.front, "left": self.l, "right": self.r}[name]
for left, right, want, why in [(0.08, 0.02, 1, "close to the right wall -> steers LEFT"),
                               (0.02, 0.08, -1, "close to the left wall -> steers RIGHT"),
                               (0.05, 0.05, 0, "centred -> straight")]:
    d = FakeDrive(ticks=(0, 0))
    n = M.Nav(d, FakeVision([None] * 5), FakeLeds(), _Sides(left, right))
    run(n, 3)
    got = 0 if abs(d.last[1]) < 1e-6 else (1 if d.last[1] > 0 else -1)
    check(why, got == want and d.last[0] > 0)

print("2) one frame doesn't commit; DETECTION_DEBOUNCE frames do")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V(), None, V(), V(), V()]), l, FakeSonar(front=0.5))
n.step()
check("still SEARCH after a single hit", n.state == M.SEARCH)
run(n, 4)
check("once debounced, it places the victim and goes to it (SEEK)", n.state == M.SEEK)
check("...in the cell straight ahead, from the sonar range",
      n.victim_cell == n.map.cell_at(*n.map.centre(M.BASE_CELL)) or n.victim_cell == (0, 4))
check("green ON", l.g is True)

print("2a) approach holds heading to within HEADING_TOLERANCE_DEG")
check("dead centre -> no correction", M.heading_correction(0.0) == 0.0)
check("inside tolerance -> no correction, no hunting",
      M.heading_correction(M.HEADING_TOLERANCE_DEG) == 0.0)
check("just outside tolerance -> it does correct",
      M.heading_correction(M.HEADING_TOLERANCE_DEG + 0.2) != 0.0)
check("a small error while rolling still commands a real turn",
      abs(M.heading_correction(2.0)) >= M.MIN_TURN_RATE_MOVING)
check("a small error while STATIONARY gets enough to break track friction",
      abs(M.heading_correction(2.0, pivoting=True)) >= M.MIN_TURN_RATE_PIVOT)
check("the pivot floor is the bigger of the two",
      M.MIN_TURN_RATE_PIVOT > M.MIN_TURN_RATE_MOVING)
check("victim to the right -> turn right", M.heading_correction(10.0) < 0)
check("victim to the left -> turn left", M.heading_correction(-10.0) > 0)
check("capped at MAX_TURN_RATE", abs(M.heading_correction(180.0)) <= M.MAX_TURN_RATE)
check("pivot floor is under the cap, or every turn would saturate",
      M.MIN_TURN_RATE_PIVOT < M.MAX_TURN_RATE)
check("symmetric", M.heading_correction(7.0) == -M.heading_correction(-7.0))

print("2b) badly off heading, it turns on the spot rather than driving off course")
d = FakeDrive()
run(A(d, FakeVision([V(bearing=35.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("no forward motion while way off heading", d.last[0] == 0.0)
check("turning towards it", d.last[1] < 0)
check("hard enough to actually pivot", abs(d.last[1]) >= M.MIN_TURN_RATE_PIVOT)
d = FakeDrive()
run(A(d, FakeVision([V(bearing=15.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("15 deg: outside the sonar's trust cone, so it pivots to centre first",
      d.last[0] == 0.0 and abs(d.last[1]) >= M.MIN_TURN_RATE_PIVOT)
d = FakeDrive()
run(A(d, FakeVision([V(bearing=8.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("8 deg: sonar believed, so it corrects while rolling",
      d.last[0] > 0 and d.last[1] < 0)
check("...and gently, not at the pivot floor", abs(d.last[1]) < M.MIN_TURN_RATE_PIVOT)
d = FakeDrive()
run(A(d, FakeVision([V(bearing=0.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("lined up -> drives, no correction", d.last[0] > 0 and d.last[1] == 0.0)

print("2c) heading locks inside HEADING_LOCK_DEG and then drives straight")
d, l = FakeDrive(), FakeLeds()
n = A(d, FakeVision([V(bearing=3.0)] * 8), l, FakeSonar(front=0.5))
run(n, 6)
check("locked once inside the lock angle", n.heading_locked is True)
check("stops steering entirely", d.last[1] == 0.0)
check("still driving forward", d.last[0] > 0)

print("2c-ii) noise inside the unlock band does NOT break the lock")
for noisy in (4.0, -4.0, 9.0, -12.0, 14.0):
    n.vision.script = [V(bearing=noisy)] * 3
    run(n, 3)
    check(f"{noisy:+5.1f} deg wobble keeps the lock", n.heading_locked is True)
    check(f"...and keeps driving straight", d.last[1] == 0.0)

print("2c-iii) a real drift past HEADING_UNLOCK_DEG does break it")
n.vision.script = [V(bearing=25.0)] * 4
run(n, 4)
check("unlocked", n.heading_locked is False)
check("steering again", d.last[1] != 0.0)

print("2c-iv) the lock is dropped when the approach restarts")
d2, l2 = FakeDrive(), FakeLeds()
n2 = A(d2, FakeVision([V()] * 4 + [None] * 20 + [V()] * 6), l2, FakeSonar(front=0.9))
run(n2, 5)
check("locked during the first approach", n2.heading_locked is True)
run(n2, 20)
check("victim lost -> SEARCH", n2.state == M.SEARCH)
check("lock cleared", n2.heading_locked is False)

print("2d) the approach keeps the green LED on")
d3, l3 = FakeDrive(), FakeLeds()
n3 = A(d3, FakeVision([V()] * 6), l3, FakeSonar(front=0.5))
run(n3, 5)
check("green stays on through the approach", l3.g is True)

print("3) approach steers the right way (+bearing = right = negative yaw)")
# Bearing inside HEADING_COARSE_DEG so it drives while correcting (2b covers the
# turn-on-the-spot case) and inside US_TRUST_BEARING_DEG so the sonar is believed.
d = FakeDrive()
run(A(d, FakeVision([V(bearing=8.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("driving forward", d.last[0] > 0)
check("yaw negative for a victim to the right", d.last[1] < 0)
d = FakeDrive()
run(A(d, FakeVision([V(bearing=-8.0)] * 5), FakeLeds(), FakeSonar(front=0.5)), 5)
check("yaw positive for a victim to the left", d.last[1] > 0)

print("4) creeps when close")
d1 = FakeDrive()
run(A(d1, FakeVision([V()] * 4), FakeLeds(), FakeSonar(front=0.50)), 4)
d2 = FakeDrive()
run(A(d2, FakeVision([V()] * 4), FakeLeds(), FakeSonar(front=0.20)), 4)
check("slower inside CREEP_RANGE_M", d2.last[0] < d1.last[0])

print("5) stops 10 cm short, green on while collecting, then home with red")
d, l, son = FakeDrive(), FakeLeds(), FakeSonar(front=0.50)
n = A(d, FakeVision([V()] * 50), l, son)
run(n, 4)
check("approaching", n.state == M.APPROACH)
son.front = 0.11
n.step()
check("AT_VICTIM at 11 cm", n.state == M.AT_VICTIM)
check("motors stopped", d.last == (0.0, 0.0))
check("green on, yellow off while collecting", l.g is True and l.y is False)
n.rescue.started = time.monotonic() - (M.RESCUE_TIME_S + 0.1)
n.step()
check("collected -> RETURN", n.state == M.RETURN and n.carrying)
check("red on, green off on the way home", l.r is True and l.g is False and l.y is False)

print("6) losing the victim returns to SEARCH")
d, l = FakeDrive(), FakeLeds()
n = A(d, FakeVision([V()] * 3 + [None] * 12), l, FakeSonar(front=0.5))
run(n, 15)
check("back to SEARCH", n.state == M.SEARCH)
check("green off again", l.g is False)

print("7) no ranging at all: green on, but it will NOT drive at the victim")
d, l = FakeDrive(), FakeLeds()
n = M.Nav(d, FakeVision([V()] * 5, geometry_ok=False), l, NO_SONAR)
run(n, 5)
check("stays in SEARCH", n.state == M.SEARCH)
check("green still ON", l.g is True)
check("not driving", d.last == (0.0, 0.0))

print("8) the front sonar alone is enough -- no camera geometry needed")
d = FakeDrive()
n = M.Nav(d, FakeVision([V()] * 6, geometry_ok=False), FakeLeds(), FakeSonar(front=0.50))
run(n, 6)
check("places the victim and goes for it on sonar range alone", n.state == M.SEEK)
check("range came from the sonar", n.victim_range()[1] == "sonar")

print("9) sonar is not believed when the victim is off to one side")
d = FakeDrive()
n = A(d, FakeVision([V(bearing=30.0)] * 6, geometry_ok=False), FakeLeds(),
          FakeSonar(front=0.50))
run(n, 6)
check("claims no distance off-axis", n.victim_range()[0] is None)
check("turns to centre it", abs(d.last[1]) > 0)
check("does NOT close on an unknown range", d.last[0] == 0.0)

print("10) horizon filter separates the floor victim from the wall marker")
# 480-row frame, camera level -> horizon at row 240. The victim object rests on the
# floor so its base is below that; the marker is mounted up the wall, above it.
check("horizon is the centre row when level", abs(M.horizon_row(480) - 240.0) < 1e-6)
check("victim on the floor is kept",     M.is_on_floor((300, 200, 60, 90), 480))   # base 290
check("wall marker is rejected",     not M.is_on_floor((300, 120, 60, 70), 480))   # base 190
check("a box straddling the horizon counts as floor",
      M.is_on_floor((300, 180, 60, 90), 480))                                      # base 270
check("tilting down raises the horizon",
      M.horizon_row(480, tilt_deg=10.0, vfov_deg=41.0) < 240.0)

print("11) floor object: camera range used when the victim is off-axis")
M.VICTIM_ON_FLOOR = True
n = M.Nav(FakeDrive(), FakeVision([V(0.40, 30.0)] * 6, geometry_ok=True), FakeLeds(),
          FakeSonar(front=0.50))
run(n, 6)
dist, src = n.victim_range()
check("falls back to the camera", src == "camera")
check("uses the camera's number, not the sonar's", abs(dist - 0.40) < 1e-9)

print("12) echo lost mid-approach: halt rather than coast on a stale range")
d, son = FakeDrive(), FakeSonar(front=0.50)
n = A(d, FakeVision([V()] * 20, geometry_ok=False), FakeLeds(), son)
run(n, 4)
son.front = None
n.step()
check("stops closing", d.last[0] == 0.0)
check("stays in APPROACH", n.state == M.APPROACH)

print("12a) losing sight of the victim close up commits to a sonar run-in")
# At 10 cm camera height the victim leaves the frame around 27 cm out. That must not
# read as "victim lost" -- it's the moment the approach is most nearly finished.
d, l, son = FakeDrive(), FakeLeds(), FakeSonar(front=0.25)
n = A(d, FakeVision([V()] * 3 + [None] * 20), l, son)
run(n, 4)
check("approaching first", n.state == M.APPROACH)
run(n, 12)                                     # victim disappears under the camera
check("hands over to CLOSING, does not go back to SEARCH", n.state == M.CLOSING)
check("still driving forward", d.last[0] > 0)
check("straight -- no steering with nothing to steer on", d.last[1] == 0.0)
check("green stays on", l.g is True)
son.front = 0.10
n.step()
check("sonar alone gets it to AT_VICTIM", n.state == M.AT_VICTIM)

print("12a-ii) but it only commits when lined up and close")
for front, bearing, why in [(0.80, 0.0, "too far to be the under-camera case"),
                            (0.25, 40.0, "off to one side, not lined up"),
                            (None, 0.0, "no sonar range at all")]:
    d2, l2 = FakeDrive(), FakeLeds()
    n2 = A(d2, FakeVision([V(bearing=bearing)] * 3 + [None] * 20), l2,
               FakeSonar(front=front))
    run(n2, 16)
    check(f"{why} -> back to SEARCH", n2.state == M.SEARCH)

print("12a-iii) the blind run-in gives up rather than driving forever")
d3 = FakeDrive()
n3 = A(d3, FakeVision([V()] * 3 + [None] * 40), FakeLeds(), FakeSonar(front=0.25))
run(n3, 16)
check("in CLOSING", n3.state == M.CLOSING)
n3.closing_since = time.monotonic() - (M.CLOSING_TIMEOUT_S + 0.1)
n3.step()
check("timed out back to SEARCH", n3.state == M.SEARCH)
check("stopped", d3.last == (0.0, 0.0))

print("12a-iv) no range mid-run-in: hold, don't drive on faith")
d4, son4 = FakeDrive(), FakeSonar(front=0.25)
n4 = A(d4, FakeVision([V()] * 3 + [None] * 40), FakeLeds(), son4)
run(n4, 16)
son4.front = None
n4.step()
check("stops when the echo drops out", d4.last == (0.0, 0.0))
check("stays in CLOSING", n4.state == M.CLOSING)

print("12a-v) the geometry the handover is based on")
_b = M.camera_blind_range_m()
check("blind range is derived, not hardcoded", _b is not None and 0.2 < _b < 0.35)
check("handover starts before the camera goes blind", M.CLOSING_TRIGGER_M > _b)

print("12b) collecting waits for the mechanism; releasing at base counts the rescue")
d, l, son = FakeDrive(), FakeLeds(), FakeSonar(front=0.11)
n = A(d, FakeVision([V()] * 400), l, son)
run(n, 2)
check("reached AT_VICTIM", n.state == M.AT_VICTIM)
n.rescue.started = time.monotonic() - (M.RESCUE_TIME_S - 0.2)
n.step()
check("still collecting until the mechanism says done", n.state == M.AT_VICTIM)
n.rescue.started = time.monotonic() - (M.RESCUE_TIME_S + 0.1)
n.step()
check("then RETURN, carrying", n.state == M.RETURN and n.carrying)
n._enter(M.AT_BASE)
check("red stays on while releasing", l.r is True)
n.release.started = time.monotonic() - (M.RELEASE_TIME_S + 0.1)
n.step()
check("released -> counted, back out exploring", n.rescued == 1 and n.state == M.SEARCH)
check("yellow again, red off", l.y is True and l.r is False)
n.rescued = M.VICTIMS_TOTAL - 1
n._enter(M.AT_BASE); n.carrying = True
n.release.started = time.monotonic() - (M.RELEASE_TIME_S + 0.1)
n.step()
check("the last one home -> DONE", n.state == M.DONE)
check("motors stopped, LEDs off", d.last == (0.0, 0.0) and not (l.g or l.y or l.r))
run(n, 20)
check("stays DONE -- terminal, even with a victim in view", n.state == M.DONE)

print("12b-ii) the 7-minute limit stops it wherever it is")
d, l = FakeDrive(ticks=(0, 0)), FakeLeds()
n = M.Nav(d, FakeVision([None] * 10), l, FakeSonar(front=1.0))
run(n, 2)
n.mission_start = time.monotonic() - M.MISSION_TIME_S - 1
n.step()
check("DONE at the time limit", n.state == M.DONE and d.last == (0.0, 0.0))

print("12c) motors.py maths: differential drive and raw-speed mapping")
import motors as MOT
_l, _r = MOT.wheel_speeds(0.10, 0.0)
check("straight: both wheels equal", _l == _r == 0.10)
_l, _r = MOT.wheel_speeds(0.0, 1.0)
check("spin: equal and opposite", abs(_l + _r) < 1e-9 and _r > 0)
_l, _r = MOT.wheel_speeds(0.10, 0.5)
check("+w turns left, so the right wheel runs faster", _r > _l)
check("stop maps to raw 0", MOT.to_raw(0.0) == 0)
check("a creep still clears stiction", MOT.to_raw(0.004) >= MOT.MIN_SPEED_RAW)
check("full speed is capped at MAX_SPEED_RAW",
      MOT.to_raw(99.0) == MOT.MAX_SPEED_RAW)
check("never exceeds the board's -127..127, which controller.py rejects",
      all(-127 <= MOT.to_raw(v) <= 127 for v in (-99, -1, -0.001, 0, 0.001, 1, 99)))
check("no dependency on the EGB320_Examples files",
      not any(w in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f)).read()
              for f in ("motors.py", "main.py", "motor_spin_test.py")
              for w in ("from controller import", "import controller\n")))
check("encoder wraparound: 65535 -> 0 counts as +1", MOT._to_i16(0 - 65535) == 1)
check("...and reversing counts down", MOT._to_i16(65535 - 0) == -1)
check("standby is the reserved -128", MOT.MotorController.STANDBY == -128)

# Speeds must actually reach the board hard enough to move a tracked chassis. These
# caught the robot barely moving: three separate limits were each throttling it.
import main as _N
_fwd = MOT.to_raw(MOT.wheel_speeds(_N.SEARCH_SPEED, 0.0)[0])
_spin = abs(MOT.to_raw(MOT.wheel_speeds(0.0, _N.SEARCH_TURN_RATE)[0]))
_nudge = abs(MOT.to_raw(MOT.wheel_speeds(0.0, _N.MIN_TURN_RATE_PIVOT)[0]))
check(f"explore drives hard ({_fwd}/127, want >=100)", _fwd >= 100)
check(f"spin is decisive ({_spin}/127, want >=80)", _spin >= 80)
check(f"the smallest pivot still breaks track friction ({_nudge}/127, want >=100)",
      _nudge >= 100)
check("full board range is available", MOT.MAX_SPEED_RAW == 127)


# Motor direction: the sign flip must reach the board AND the odometry, or a
# reversed motor drives right and counts backwards.
class _FakeBoard:
    STANDBY = -128
    def __init__(s): s.sent = None; s.raw = (0, 0)
    def set_raw_motor_speed(s, l, r): s.sent = (l, r)
    def set_motor_shutdown_timeout(s, t): pass
    def get_raw_encoder_ticks(s): return s.raw
    def get_firmware_version(s): return (1, 3, 0)
    def standby(s): pass

_fb = _FakeBoard()
_old = (MOT.LEFT_SIGN, MOT.RIGHT_SIGN, MOT.LEFT_TRIM, MOT.RIGHT_TRIM,
        MOT.ENCODER_LEFT_SIGN, MOT.ENCODER_RIGHT_SIGN)
MOT.LEFT_SIGN = MOT.RIGHT_SIGN = -1
MOT.LEFT_TRIM = MOT.RIGHT_TRIM = 1.0
MOT.ENCODER_LEFT_SIGN = MOT.ENCODER_RIGHT_SIGN = 1
_d = MOT.MotorDriver(controller=_fb)
_d.set_raw(50, 50)
check("LEFT_SIGN/RIGHT_SIGN reach the board", _fb.sent == (-50, -50))
_fb.raw = (100, 100)                       # counters advanced
_t = _d.read_encoders()
check("the MOTOR signs don't flip the encoders: a reversed motor's encoder still "
      "counts forward as up", _t == (100, 100))
# ...because the encoders have their own signs. On this robot the right motor needs
# RIGHT_SIGN = -1 but its encoder counts up going forward; flipping it by the motor sign
# made the odometry see one track going backwards (so: spinning, never arriving).
_fb.raw = (200, 200)
MOT.ENCODER_RIGHT_SIGN = -1
_t = _d.read_encoders()
check("ENCODER_RIGHT_SIGN flips only the right encoder", _t == (200, 0))
MOT.LEFT_SIGN, MOT.RIGHT_SIGN, MOT.LEFT_TRIM, MOT.RIGHT_TRIM, \
    MOT.ENCODER_LEFT_SIGN, MOT.ENCODER_RIGHT_SIGN = _old
check("shipped encoder signs both count forward as up -- opposite signs would cancel "
      "to zero forward distance",
      MOT.ENCODER_LEFT_SIGN == 1 and MOT.ENCODER_RIGHT_SIGN == 1)

# SWAP_MOTORS has to reach the encoders as well as the motors. If it only swapped
# the commands, the straight-line correction would read the wrong track and steer
# away from straight -- a bug that only shows up while driving, not on the bench.
_was = MOT.SWAP_MOTORS
for _swap in (False, True):
    MOT.SWAP_MOTORS = _swap
    _fb2 = _FakeBoard()
    _d2 = MOT.MotorDriver(controller=_fb2)
    _d2.set_velocity(0.0, 2.0)                 # spin left
    _fb2.raw = (300, 100)
    _t2 = _d2.read_encoders()
    if not _swap:
        _unswapped_cmd, _unswapped_ticks = _fb2.sent, _t2
    else:
        check("SWAP_MOTORS mirrors the motor command",
              _fb2.sent == (_unswapped_cmd[1], _unswapped_cmd[0]))
        check("SWAP_MOTORS mirrors the encoder channels too",
              _t2 == (_unswapped_ticks[1], _unswapped_ticks[0]))
MOT.SWAP_MOTORS = _was

# Track trim, for a robot that pulls to one side.
_trims = (MOT.LEFT_TRIM, MOT.RIGHT_TRIM)
def _tracks(fb):
    return (fb.sent[1], fb.sent[0]) if MOT.SWAP_MOTORS else fb.sent
MOT.LEFT_TRIM, MOT.RIGHT_TRIM = 1.0, 1.0
_fb3 = _FakeBoard(); MOT.MotorDriver(controller=_fb3).set_raw(80, 80)
_base = _tracks(_fb3)
check("untrimmed, both tracks get the same", _base[0] == _base[1])
MOT.LEFT_TRIM = 1.10
_fb4 = _FakeBoard(); MOT.MotorDriver(controller=_fb4).set_raw(80, 80)
_trimmed = _tracks(_fb4)
check("LEFT_TRIM drives the LEFT track harder",
      abs(_trimmed[0]) > abs(_base[0]) and abs(_trimmed[1]) == abs(_base[1]))
check("trim lands on the track, not the board channel -- survives SWAP_MOTORS",
      abs(_trimmed[0]) == round(abs(_base[0]) * 1.10))
MOT.LEFT_TRIM, MOT.RIGHT_TRIM = _trims
check("reverse is negative", MOT.to_raw(-0.10) < 0)
check("magnitude matches forward", abs(MOT.to_raw(-0.10)) == MOT.to_raw(0.10))
check("board address is the unit's, not the DFRobot one",
      (MOT.I2C_ADDR, MOT.I2C_BUS) == (0x57, 8))

print("12d) only fitted sonars are pinged")
# A sensor that isn't wired still costs a full echo timeout every time its turn
# comes round -- ~22 ms against a 50 ms tick. Skipping them keeps the loop rate.
check("all three sonars are fitted for the maze", set(M.SONARS_FITTED) == {"front", "left", "right"})
_u = M.Ultrasonics(enabled=False)
_u.sensors = {"front": object()}
_seq = []
for _ in range(6):
    for _ in range(len(_u.ORDER)):
        _nm = _u.ORDER[_u._i % len(_u.ORDER)]; _u._i += 1
        if _nm in _u.sensors: break
    _seq.append(_nm)
check("with only the front plugged in, every tick reads it, none wasted",
      _seq == ["front"] * 6)

print("13) a blurred frame is not the victim disappearing")
# classify_frame() returns None on a motion-blurred frame, which is NOT an empty
# result. Nav must hold what it knows rather than counting it as a miss -- otherwise
# a fast pan looks identical to the victim vanishing.
d, l, son = FakeDrive(), FakeLeds(), FakeSonar(front=0.50)
n = A(d, FakeVision([V()] * 3 + [M.UNUSABLE] * 60), l, son)
run(n, 4)
check("approaching before the blur", n.state == M.APPROACH)
run(n, 30)                                  # far more than LOST_GRACE_FRAMES
check("still APPROACH through a long blur", n.state == M.APPROACH)
check("green still on", l.g is True)
check("misses not counted", n.misses == 0)

print("14) a genuinely empty frame still loses the victim")
d, l = FakeDrive(), FakeLeds()
n = A(d, FakeVision([V()] * 3 + [None] * 12), l, FakeSonar(front=0.50))
run(n, 15)
check("empty frames DO drop to SEARCH", n.state == M.SEARCH)

print("15) every name main() reaches for actually exists")
# The rest of this file injects fakes, so it never touches the real Leds, Drive or
# Ultrasonics classes -- a refactor once deleted all three and every test still
# passed. This walks main() for the globals it calls and checks they're defined.
import ast
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")).read()
_tree = ast.parse(_src)
_main_fn = next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
_called = {n.func.id for n in ast.walk(_main_fn)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
_missing = sorted(c for c in _called if not hasattr(M, c) and not hasattr(__builtins__, c)
                  and c not in dir(__builtins__))
check(f"main() references nothing undefined{(' -- missing ' + ', '.join(_missing)) if _missing else ''}",
      not _missing)
for _name in ("Leds", "Drive", "Ultrasonics", "VictimVision", "Display", "Nav",
              "StatusLine", "render_hud", "status_text", "horizon_row", "is_on_floor"):
    check(f"{_name} is defined", hasattr(M, _name))

print("16) the real Leds and Ultrasonics construct off-Pi")
# Both must degrade instead of raising when there's no GPIO, or nothing can be
# tested on a laptop.
_leds = M.Leds()
_leds.green(True); _leds.yellow(True); _leds.red(True); _leds.all_off()
check("Leds falls back to console without gpiozero", _leds.real is False)
_son = M.Ultrasonics(enabled=False)
check("Ultrasonics(enabled=False) reports unavailable", _son.available is False)
check("...and returns no reading rather than raising", _son.front is None)
_son.update()
check("...and update() is a no-op", _son.get("front") is None)
_st = M.status_text(M.Nav(FakeDrive(), FakeVision([]), _leds, _son), _son, _leds, 20.0)
check("status_text renders", "EXPLORING" in _st and "sonar" in _st)

print("17) a frame that goes nowhere says so instead of looking empty")
# The blur guard in VisionSystem.classify_frame discards a whole frame, which used to
# produce no boxes and no message -- indistinguishable from "nothing is there". That
# cost real bench time, so the three outcomes must now be told apart in the summary.
try:
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "vision"))
    import camera_capture_v2_0 as _cc

    _frames = []
    class _FakeCam:
        def __init__(self, *a, **k): pass
        def read(self): return _frames.pop(0) if _frames else None
        def release(self): pass
    _cc.CameraCapture = _FakeCam

    _cwd = os.getcwd()
    _vision = M.VictimVision()
    os.chdir(_cwd)
    _rng = np.random.default_rng(2)

    _flat = np.full((480, 640, 3), 70, np.uint8)                  # no texture -> rejected
    _cyan = _rng.integers(45, 95, (480, 640, 3), dtype=np.uint8)
    _cyan[320:390, 250:390] = (255, 255, 0)                       # BGR cyan on the floor
    _empty = _rng.integers(45, 95, (480, 640, 3), dtype=np.uint8)

    # The vision team's blur guard is RELATIVE: a frame is "blurred" only against the camera's
    # own recent sharp frames, so it needs some history first (a lone frame is never flagged).
    for _ in range(6):
        _frames.append(_empty.copy())
        _vision.look()
    _frames.append(_flat)
    check("blurred frame returns UNUSABLE", _vision.look() is M.UNUSABLE)
    check("...and is reported as a rejection", "BLUR-REJECT" in M.vision_summary(_vision))

    _frames.append(_cyan)
    _r = _vision.look()
    check("cyan object on the floor is a victim", isinstance(_r, M.Victim))
    check("...and the summary counts it", "victim 1/1" in M.vision_summary(_vision))

    _frames.append(_empty)
    check("textured but empty frame returns None", _vision.look() is None)
    _sum = M.vision_summary(_vision)
    check("...reported as empty, NOT as a rejection",
          "none" in _sum and "BLUR-REJECT" not in _sum)
except ImportError as _e:
    print(f"  SKIP  needs numpy + opencv + the vision package ({_e})")

print("18) the blur guard is OFF by default, so a dim frame still detects")
# This was a flag you had to remember to pass, and forgetting it meant no victim
# detection at all from main.py. It's now the default; this locks that in.
check("BLUR_THRESHOLD defaults to off", M.BLUR_THRESHOLD == 0.0)
try:
    import numpy as np
    import vision_system_v2_0 as _vsmod
    _vsmod.BLUR_VARIANCE_THRESHOLD = M.BLUR_THRESHOLD
    _frames.append(np.full((480, 640, 3), 70, np.uint8))
    _frames[-1][300:390, 250:390] = (255, 255, 0)     # turquoise, low-texture scene
    _v2 = M.VictimVision(); os.chdir(_cwd)
    _r2 = _v2.look()
    check("dim low-texture frame still yields a victim", isinstance(_r2, M.Victim))
except (ImportError, NameError) as _e:
    print(f"  SKIP  needs numpy + opencv ({_e})")

print("19) the camera as a second wall sensor")
import types


def view(**kw):
    base = dict(t=time.monotonic(), front_close=False, front_m=None, left_m=None,
                right_m=None, left_slope=None, right_slope=None, columns=0,
                yaw_rad=None, yaw_src=None)
    base.update(kw)
    v = types.SimpleNamespace(**base)
    v.age = lambda now=None: time.monotonic() - v.t
    return v


class _Open(FakeSonar):
    """Sonars that see open space on all three sides (no echo)."""
    def __init__(self):
        super().__init__(front=None)
        self.sensors = {"front": 1, "left": 1, "right": 1}
    def get(self, name): return None


def cam_nav(v):
    n = M.Nav(FakeDrive(), FakeVision([None] * 5), FakeLeds(), _Open(),
              wall_camera=object())
    n.cam_walls = v
    return n


n = cam_nav(view(front_close=True))
n.sense_from = {k: 0.0 for k in ("front", "left", "right")}
n._record_walls(["front", "left", "right"])
check("sonar sees no front wall, camera sees one close -> marked", n.map.wall(n.cell, 0) is True)
check("...but nothing is invented on the open east side", n.map.wall(n.cell, 1) is False)

n = cam_nav(view(left_m=0.15, right_m=0.14))
n.sense_from = {k: 0.0 for k in ("front", "left", "right")}
n._record_walls(["front", "left", "right"])
check("camera side walls at ~half a cell -> both marked",
      n.map.wall(n.cell, 3) is True and n.map.wall(n.cell, 1) is True)

n = cam_nav(view(right_m=0.43))
n.sense_from = {k: 0.0 for k in ("front", "left", "right")}
n._record_walls(["front", "left", "right"])
check("a wall a whole cell further off is the next cell's, not this one's",
      n.map.wall(n.cell, 1) is False)

n = cam_nav(view(t=time.monotonic() - 5.0, front_close=True))
check("a stale view is ignored", n._camera_view() is None)

n = cam_nav(view(front_close=True))
n.victim_cell = n.map.neighbour(n.cell, 0)
n.sense_from = {k: 0.0 for k in ("front", "left", "right")}
n._record_walls(["front", "left", "right"])
check("never seals off the cell a victim is in", n.map.wall(n.cell, 0) is False)

# centring blend
class _Sides(FakeSonar):
    def __init__(self, l, r):
        super().__init__(front=1.0); self.l, self.r = l, r
        self.sensors = {"front": 1, "left": 1, "right": 1}
    def get(self, name): return {"front": 1.0, "left": self.l, "right": self.r}[name]
    def stamp(self, name): return time.monotonic()


def offset(sonar_l, sonar_r, cam):
    n = M.Nav(FakeDrive(), FakeVision([None]), FakeLeds(), _Sides(sonar_l, sonar_r),
              wall_camera=object())
    n.cam_walls = cam
    return n.mover._blended_offset()


half, side = M.CELL_M / 2, M.SIDE_WALL_AT_CENTRE_M
check("sonar and camera agree 2 cm right -> blended 2 cm",
      abs(offset(side + 0.02, side - 0.02, view(left_m=half + 0.02, right_m=half - 0.02))
          - 0.02) < 1e-9)
check("they disagree by more than CAMERA_AGREE_M -> the sonars win",
      abs(offset(side + 0.02, side - 0.02, view(left_m=half - 0.04, right_m=half + 0.04))
          - 0.02) < 1e-9)
check("sonars blind (no echo) -> the camera's offset is used",
      abs(offset(None, None, view(left_m=half + 0.03, right_m=half - 0.03)) - 0.03) < 1e-9)
check("camera sees only the right wall -> offset from that wall alone",
      abs(offset(None, None, view(right_m=half - 0.025)) - 0.025) < 1e-9)
check("neither has anything -> None (falls back to the filtered pose)",
      offset(None, None, view()) is None)
check("no camera at all -> sonar offset unchanged",
      abs(offset(side + 0.02, side - 0.02, None) - 0.02) < 1e-9)

print("20) squaring up to the walls after a turn")
import math


class LogDrive(FakeDrive):
    """Remembers every set_velocity, since stop() wipes .last."""
    def __init__(self):
        super().__init__(); self.log = []
    def set_velocity(self, v, w):
        super().set_velocity(v, w); self.log.append((v, w))


def mover_nav(yaw=None, t=None):
    d = LogDrive()
    n = M.Nav(d, FakeVision([None] * 5), FakeLeds(), _Open(), wall_camera=object())
    n.cam_walls = view(yaw_rad=yaw, yaw_src="walls" if yaw is not None else None,
                       **({} if t is None else {"t": t}))
    return n, d


EAST = M.HEADING_RAD[1]
n, d = mover_nav()
n.odo.theta = M.HEADING_RAD[0]
n.mover.move_to(0.42, 0.14, EAST)
check("starts with a turn on the spot", n.mover.phase == M.Mover.TURN)
n.odo.theta = EAST + 0.01                       # the turn lands close enough
n.mover.update()
check("...then squares up before driving off", n.mover.phase == M.Mover.ALIGN)

n, d = mover_nav(yaw=0.10)
n.odo.theta = EAST
n.mover._begin_align(time.monotonic())
n.mover.align_ready = 0.0
n.mover._align()
check("turned 5.7 deg LEFT of the walls -> pulses a turn to the RIGHT",
      d.log and d.log[-1][0] == 0.0 and d.log[-1][1] < 0)
check("...and stops again afterwards", d.last == (0.0, 0.0))
check("still squaring up (not off driving)", n.mover.phase == M.Mover.ALIGN)

n, d = mover_nav(yaw=-0.10)
n.mover._begin_align(time.monotonic()); n.mover.align_ready = 0.0
n.mover._align()
check("turned RIGHT of the walls -> pulses LEFT", d.log and d.log[-1][1] > 0)

n, d = mover_nav(yaw=0.10, t=time.monotonic())
n.mover.move_to(0.42, 0.14, EAST)
n.mover._begin_align(time.monotonic())          # frame taken BEFORE it stopped moving
n.mover._align()
check("a frame from before it settled is not acted on", d.log == [])

n, d = mover_nav(yaw=0.005)
n.odo.theta = EAST + 0.2                         # encoders say 11 deg out; walls say square
n.mover.move_to(0.42, 0.14, EAST)
n.mover._begin_align(time.monotonic()); n.mover.align_ready = 0.0
n.mover._align()
check("square -> drives off", n.mover.phase == M.Mover.DRIVE)
check("...and the heading estimate is snapped to the walls, not the encoders",
      abs(n.odo.theta - (EAST + 0.005)) < 1e-9)

n, d = mover_nav()                               # nothing in view to square up to
n.mover.move_to(0.42, 0.14, EAST)
n.mover._begin_align(time.monotonic() - 1.0)
n.mover._align()
check("no wall angle available -> doesn't hang, drives on", n.mover.phase == M.Mover.DRIVE)

n, d = mover_nav(yaw=0.0)
n.odo.theta = EAST + 0.01
n.mover.move_to(0.42, 0.14, EAST)
n.odo.theta = EAST + 0.10                        # drifted 5.7 deg by the encoders' count
n.mover.update()
want = EAST + 0.10 * (1 - M.CAMERA_YAW_NUDGE)
check("driving: heading estimate pulled towards what the walls say",
      abs(n.odo.theta - want) < 1e-6)
n.mover.update()
check("...once per fresh frame, not once per tick", abs(n.odo.theta - want) < 1e-3)

n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Open())    # no wall camera
n.odo.theta = M.HEADING_RAD[0]
n.mover.move_to(0.42, 0.14, EAST)
n.odo.theta = EAST + 0.01
n.mover.update()
check("no camera -> no squaring-up phase, straight to driving", n.mover.phase == M.Mover.DRIVE)

print("21) the wall ahead: the camera where it can range, the sonar where it can't")
_half_to_sonar = M.CELL_M / 2 - M.FRONT_WALL_AT_CENTRE_M     # sonar sits this far ahead of the centre


class _Front(_Open):
    """Sonars with a front reading only."""
    def __init__(self, front):
        super().__init__(); self.front = front
    def stamp(self, name): return time.monotonic()


def front_nav(sonar_front, **view_kw):
    n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Front(sonar_front), wall_camera=object())
    n.cam_walls = view(**view_kw)
    return n


d, src = front_nav(0.20, front_m=0.62).front_distance()
check("camera can range the wall -> camera's distance", src == "cam"
      and abs(d - (0.62 - M.CAMERA_FORWARD_M)) < 1e-9)
d, src = front_nav(0.05, front_close=True).front_distance()
check("camera says 'too close to see' -> the sonar gives the distance", src == "sonar"
      and abs(d - (0.05 + _half_to_sonar)) < 1e-9)
d, src = front_nav(0.30, front_m=None).front_distance()
check("camera sees no wall ahead -> the sonar", src == "sonar")
d, src = front_nav(None, front_close=True).front_distance()
check("close to the camera and no echo -> says it can't tell", d is None and src is None)
d, src = front_nav(None, front_m=0.50).front_distance()
check("sonar silent but the camera ranges it -> camera", src == "cam")

n = front_nav(0.07, front_close=True)
r, _t = n.mover._front_reading()
check("blocked-ahead check: sonar reading used as is when it has an echo", r == 0.07)
n = front_nav(None, front_m=0.62)
r, _t = n.mover._front_reading()
check("...camera, expressed as an equivalent sonar reading, when the sonar is silent",
      abs(r - (0.62 - M.CAMERA_FORWARD_M - _half_to_sonar)) < 1e-9)
n = front_nav(None, front_close=True)
check("...nothing when neither can say", n.mover._front_reading() == (None, None))

# the camera alone spots something in the way when the sonar gets no echo
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Front(None), wall_camera=object())
n.odo.theta = M.HEADING_RAD[0]
n.mover.move_to(*n.map.centre((0, 5)), M.HEADING_RAD[0])      # one cell north, open on the map
for i in range(M.BLOCK_CONFIRM + 2):
    # a wall only 30 cm from the lens, with ~28 cm to go: the cell ahead is walled off
    n.cam_walls = view(t=time.monotonic() + i * 1e-3, front_m=0.30)
    n.mover.update()
    if n.mover.phase == M.Mover.BACKOUT:
        break
check("silent sonar, camera sees a wall too near -> backs out",
      n.mover.phase == M.Mover.BACKOUT)

print("22) it says so when the encoders aren't keeping up with the drive command")
_events = []
_orig_event = M.STATUS.event
M.STATUS.event = lambda msg: _events.append(msg)


def stalled_drive(left_m=0.0, right_m=0.0, gone=0.0, secs=2.5):
    n, d = mover_nav()
    n.odo.theta = M.HEADING_RAD[0]
    n.mover.move_to(*n.map.centre((0, 5)), M.HEADING_RAD[0])
    n.mover.phase_started = time.monotonic() - secs
    n.odo.y += gone
    n.odo.left_m, n.odo.right_m = left_m, right_m
    n.mover._start_lr = (0.0, 0.0)
    n.mover.update()
    return n


_events.clear(); stalled_drive()
check("encoders counting nothing -> says so", any("NOTHING" in e for e in _events))
_events.clear(); stalled_drive(left_m=0.03, right_m=0.03, gone=0.03)
check("encoders counting far too little -> suggests TICKS_PER_M",
      any("TICKS_PER_M" in e and "calibrate" in e for e in _events))
_events.clear(); stalled_drive(left_m=0.2, right_m=0.2, gone=0.2)
check("keeping up fine -> silent", not any("[odo]" in e for e in _events))
_events.clear(); stalled_drive(secs=0.5)
check("too early to judge -> silent", not any("[odo]" in e for e in _events))
_events.clear(); n = stalled_drive(); n.mover.update()
check("warns once per drive, not every tick", sum("[odo]" in e for e in _events) == 1)
M.STATUS.event = _orig_event

print("23) the camera never overrules a sonar that sees clear space")
_ev = []
_orig_ev = M.STATUS.event
M.STATUS.event = lambda msg: _ev.append(msg)


def sense_front(sonar_front, **view_kw):
    n = front_nav(sonar_front, **view_kw)
    n.sense_from = {k: 0.0 for k in ("front", "left", "right")}
    n._record_walls(["front", "left", "right"])
    return n


n = sense_front(0.67, front_close=True)
check("sonar reads 67 cm clear, camera claims a wall right ahead -> NO wall marked",
      n.map.wall(n.cell, 0) is False)
check("...and the disagreement is logged with the camera's numbers",
      any("trusting the sonar" in e and "cam:" in e for e in _ev))
n = sense_front(0.17, front_close=True)
check("sonar reads 17 cm (just outside the wall window), camera sees a wall -> marked",
      n.map.wall(n.cell, 0) is True)
n = sense_front(None, front_close=True)
check("sonar no echo, camera sees a wall -> marked", n.map.wall(n.cell, 0) is True)
n = sense_front(0.05, front_close=False)
check("sonar sees the wall itself -> marked, camera not needed", n.map.wall(n.cell, 0) is True)

# a camera that keeps contradicting the sonar gets reported, once
_ev.clear()
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Front(0.80), wall_camera=object())
for _i in range(M.CAMERA_DISAGREE_TICKS + 5):
    n.cam_walls = view(front_close=True)
    n._check_camera_agrees()
check("camera 'wall ahead' vs sonar 80 cm for many frames -> one warning",
      sum("WARNING: the camera says a wall is right ahead" in e for e in _ev) == 1)
_ev.clear()
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Front(0.10), wall_camera=object())
for _i in range(M.CAMERA_DISAGREE_TICKS + 5):
    n.cam_walls = view(front_close=True)
    n._check_camera_agrees()
check("camera and sonar agree there's a wall -> silent", not _ev)
M.STATUS.event = _orig_ev

# the status line never wraps
import io, contextlib
_sl = M.StatusLine(); _buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _sl.update("x" * 500)
check("status line is cut to the terminal width", len(_buf.getvalue().strip("\r")) <
      shutil_cols if (shutil_cols := __import__("shutil").get_terminal_size().columns) else True)

print("24) a camera that contradicts the sonar is ignored for a while")
_ev = []
_orig_ev = M.STATUS.event
M.STATUS.event = lambda msg: _ev.append(msg)
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Front(0.80), wall_camera=object())
n.cam_walls = view(front_close=True)
check("before it has disagreed, the camera view is used", n._camera_view() is not None)
for _i in range(M.CAMERA_DISAGREE_TICKS):
    n.cam_walls = view(front_close=True)
    n._check_camera_agrees()
n.cam_walls = view(front_close=True, left_m=0.14, right_m=0.14, yaw_rad=0.05)
check("after the camera keeps saying 'wall ahead' against a far sonar, its view is not acted on",
      n._camera_view() is None)
check("...so it can't add walls, steer the centring, or square up",
      n._camera_sees_wall("front") is None and n.mover._camera_offset() is None
      and n.mover._wall_yaw() == (None, None))
n._cam_suspect_until = 0.0
check("...and it comes back once the window has passed", n._camera_view() is not None)
for _i in range(M.CAMERA_DISAGREE_TICKS * 3):
    n.cam_walls = view(front_close=True)
    n._check_camera_agrees()
check("the warning is rate-limited, not printed every few frames",
      sum("WARNING: the camera says a wall" in e for e in _ev) <= 2)
M.STATUS.event = _orig_ev

n, d = mover_nav(yaw=0.30)                       # 17 deg: past what the camera can measure
check("a wall angle bigger than CAMERA_YAW_MAX_RAD is not trusted", n.mover._wall_yaw() == (None, None))

print("25) straight_test.py: steering check and trim advice")
import straight_test as ST


class _Clock:
    def __init__(self): self.t = 0.0
    def monotonic(self): return self.t
    def sleep(self, s): self.t += s; self.on_sleep(s)
    def on_sleep(self, s): pass


class _Tracks:
    """Fake chassis: ticks accumulate from the commanded track speeds. `gain_l/gain_r`
    are how strong each track really is; `mirrored` swaps which encoder counts which."""
    def __init__(self, clock, gain_l=1.0, gain_r=1.0, mirrored=False):
        self.l = self.r = 0.0; self.v = self.w = 0.0
        self.gl, self.gr, self.mirrored = gain_l, gain_r, mirrored
        clock.on_sleep = self.advance
    def set_velocity(self, v, w): self.v, self.w = v, w
    def stop(self): self.v = self.w = 0.0
    def read_encoders(self):
        a, b = (self.r, self.l) if self.mirrored else (self.l, self.r)
        return (int(a), int(b))
    def advance(self, dt):
        half = 0.0615
        self.l += (self.v - self.w * half) * self.gl * dt * M.TICKS_PER_M
        self.r += (self.v + self.w * half) * self.gr * dt * M.TICKS_PER_M


def _run_steer(**kw):
    clock = _Clock()
    drive = _Tracks(clock, **kw)
    odo = M.Odometry(M.TICKS_PER_M, M.EFFECTIVE_TRACK_M)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        real = ST.drive_run
        ST.drive_run = lambda d, o, sec, v, st, label, clock=clock: real(d, o, sec, v, st, label, clock)
        try:
            ok = ST.step1_steering(drive, odo, 2.5, M.TICKS_PER_M)
        finally:
            ST.drive_run = real
    return ok, buf.getvalue()


ok, out = _run_steer()
check("steering and encoders agree -> OK", ok and "OK" in out)
ok, out = _run_steer(mirrored=True)
check("encoder channels mirrored vs the tracks -> says so and names SWAP_MOTORS",
      not ok and "MIRRORED" in out and "SWAP_MOTORS" in out)

_old_trims = (ST.MOT.LEFT_TRIM, ST.MOT.RIGHT_TRIM)
ST.MOT.LEFT_TRIM = ST.MOT.RIGHT_TRIM = 1.0
ratio, adv = ST.trim_advice(1000, 1000)
check("even tracks -> trims are fine", "fine" in adv)
ratio, adv = ST.trim_advice(1000, 1250)
check("right track 25% further -> pulls LEFT, lower RIGHT_TRIM to 0.8",
      "pulls LEFT" in adv and "RIGHT_TRIM = 0.800" in adv)
ratio, adv = ST.trim_advice(1250, 1000)
check("left track 25% further -> pulls RIGHT, lower LEFT_TRIM to 0.8",
      "pulls RIGHT" in adv and "LEFT_TRIM = 0.800" in adv)
ST.MOT.LEFT_TRIM, ST.MOT.RIGHT_TRIM = _old_trims

clock = _Clock(); drive = _Tracks(clock, gain_l=1.0, gain_r=1.15)
odo = M.Odometry(M.TICKS_PER_M, M.EFFECTIVE_TRACK_M)
r = ST.drive_run(drive, odo, 2.5, 0.13, lambda o: 0.0, "open", clock)
check("a stronger right track -> heading drifts LEFT (+) in the open-loop run", r["heading"] > 3)
odo = M.Odometry(M.TICKS_PER_M, M.EFFECTIVE_TRACK_M)
clock = _Clock(); drive = _Tracks(clock, gain_l=1.0, gain_r=1.15)
r2 = ST.drive_run(drive, odo, 2.5, 0.13,
                  lambda o: max(-1.0, min(1.0, 4.0 * M.wrap(0.0 - o.theta))), "hold", clock)
check("...and the heading hold pulls it back to within a few degrees",
      abs(r2["heading"]) < 3 and r2["peak"] < abs(r["heading"]))

class _Board(_Tracks):
    """_Tracks plus the .driver.set_raw the speed-curve step drives. Speed saturates, like
    the real motors seem to: it climbs to raw 60, then barely changes."""
    def __init__(self, clock):
        super().__init__(clock)
        self.driver = self
    def set_raw(self, left, right):
        def mps(raw):
            return math.copysign(min(abs(raw), 60) / 60 * 0.26 + (max(abs(raw), 60) - 60) * 0.0002, raw)
        self.l_v, self.r_v = mps(left), mps(right)
    def advance(self, dt):
        self.l += getattr(self, "l_v", 0.0) * dt * M.TICKS_PER_M
        self.r += getattr(self, "r_v", 0.0) * dt * M.TICKS_PER_M
    def stop(self): self.l_v = self.r_v = 0.0


import io, contextlib
_clock = _Clock(); _board = _Board(_clock); _clock.on_sleep = _board.advance
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _rows = ST.step4_speed_curve(_board, M.TICKS_PER_M, _clock)
check("speed curve: one row per raw command, measured speeds positive in both directions",
      len(_rows) == len(ST.SPEED_CURVE_RAWS) and all(r[1] > 0 and r[2] > 0 for r in _rows))
check("...and it reports where the speed stops responding (steering has to work below that)",
      "reaches 90%" in _buf.getvalue())
_clock = _Clock(); _tr = _Tracks(_clock, gain_l=1.0, gain_r=1.0)
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _real = ST.drive_run
    ST.drive_run = lambda d, o, sec, v, st, label, clock=_clock: _real(d, o, sec, v, st, label, clock)
    try:
        ST.step1_steering(_tr, M.Odometry(M.TICKS_PER_M, M.EFFECTIVE_TRACK_M), 2.5, M.TICKS_PER_M)
    finally:
        ST.drive_run = _real
check("step 1 reports steering authority as a percentage", "steering authority" in _buf.getvalue())

_clock = _Clock(); _board = _Board(_clock); _clock.on_sleep = _board.advance
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _rows5 = ST.step5_yaw_response(_board, M.Odometry(M.TICKS_PER_M, M.EFFECTIVE_TRACK_M),
                                   M.TICKS_PER_M, M.EFFECTIVE_TRACK_M, _clock)
check("yaw response: one row per difference, zero difference turns nothing",
      len(_rows5) == len(ST.YAW_RESPONSE_DIFFS) and abs(_rows5[0][3]) < 1.0)
check("...and a bigger difference turns left more (right track faster)",
      _rows5[-1][3] > _rows5[1][3] > -1.0)

print("25b) trims are for driving, not pivots")
_saved = (MOT.LEFT_TRIM, MOT.RIGHT_TRIM, MOT.SWAP_MOTORS, MOT.LEFT_SIGN, MOT.RIGHT_SIGN)
MOT.LEFT_TRIM, MOT.RIGHT_TRIM, MOT.SWAP_MOTORS, MOT.LEFT_SIGN, MOT.RIGHT_SIGN = 1.0, 0.8, False, 1, 1
_fbp = _FakeBoard(); _dp = MOT.MotorDriver(controller=_fbp)
_dp.set_raw(100, 100)
check("driving straight: the right track is trimmed down", _fbp.sent == (100, 80))
_dp.set_raw(-100, 100)
check("pivoting: both tracks get the same power (no trim)", _fbp.sent == (-100, 100))
_dp.set_raw(100, -100)
check("...either way round", _fbp.sent == (100, -100))
_dp.set_velocity(0.0, 2.2)
check("a spin through set_velocity is not trimmed either", abs(_fbp.sent[0]) == abs(_fbp.sent[1]))
_dp.set_raw(0, 100)
check("swinging on one track (the other stopped) still trims", _fbp.sent == (0, 80))
MOT.LEFT_TRIM, MOT.RIGHT_TRIM, MOT.SWAP_MOTORS, MOT.LEFT_SIGN, MOT.RIGHT_SIGN = _saved

print("25c) which way it physically turns (step 6)")


def _step6(answer, **kw):
    clock = _Clock(); drive = _Tracks(clock, **kw)
    odo = M.Odometry(M.TICKS_PER_M, M.EFFECTIVE_TRACK_M)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = ST.step6_physical_direction(drive, odo, clock, answer=answer)
    return ok, buf.getvalue()


ok, out = _step6("l")
check("turned left as commanded (odometry + and the user saw LEFT) -> OK", ok is True)
ok, out = _step6("r")
check("odometry says left but the robot visibly went RIGHT -> MIRRORED, names SWAP_MOTORS",
      ok is False and "MIRRORED" in out and "SWAP_MOTORS" in out)
ok, out = _step6("")
check("no answer -> doesn't pretend to judge", ok is None)

print("25d) the shipped motor wiring: named tracks land on the right physical motors")
# Measured: board channel 1 drives the physical RIGHT motor, channel 2 the physical LEFT.
# With the NEW motors (2026-10-15) both turn forward on a positive command (the new left
# motor needed LEFT_SIGN flipped to +1). Whatever the SWAP/SIGN/TRIM values are, this must
# hold for the config that ships, or a "turn left" turns the robot right.
_fbw = _FakeBoard()
_dw = MOT.MotorDriver(controller=_fbw)
_t = (MOT.LEFT_TRIM, MOT.RIGHT_TRIM)
MOT.LEFT_TRIM = MOT.RIGHT_TRIM = 1.0
_dw.set_raw(100, 100)
check("both tracks forward = (+ch1, +ch2) -- what drives the new motors straight",
      _fbw.sent[0] > 0 and _fbw.sent[1] > 0 and abs(_fbw.sent[0]) == abs(_fbw.sent[1]))
_dw.set_raw(-100, 100)                                 # LEFT pivot: left back, right forward
check("a LEFT pivot drives the physical RIGHT motor (ch1) forward and the LEFT (ch2) back",
      _fbw.sent[0] > 0 > _fbw.sent[1])
_dw.set_raw(100, -100)                                 # RIGHT pivot
check("a RIGHT pivot is the reverse", _fbw.sent[0] < 0 < _fbw.sent[1])
_dw.set_raw(100, 0)                                    # only the named LEFT track
check("the named LEFT track alone is the physical left motor (ch2)",
      _fbw.sent[0] == 0 and _fbw.sent[1] != 0)
_dw.set_raw(0, 100)
check("the named RIGHT track alone is the physical right motor (ch1)",
      _fbw.sent[1] == 0 and _fbw.sent[0] != 0)
MOT.LEFT_TRIM, MOT.RIGHT_TRIM = _t
_fbw2 = _FakeBoard(); _dw2 = MOT.MotorDriver(controller=_fbw2)
_fbw2.raw = (100, 100); _dw2.read_encoders(); _fbw2.raw = (300, 280)
_tk = _dw2.read_encoders()
check("the encoders follow the motors: ch1 (physical right) counts into the RIGHT track",
      _tk is not None and _tk[1] > _tk[0] > 0)

print("26) --no-centring drives on the encoder heading alone")
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Sides(0.14, 0.02))   # wall hard on the right
n.odo.theta = M.HEADING_RAD[0]
n.mover.move_to(*n.map.centre((0, 5)), M.HEADING_RAD[0])
n.mover.update()
check("with centring on, a wall close on the right steers it away (lateral set)",
      n.mover.lateral is not None and abs(n.mover.lateral) > 0.01)
M.CENTRING = False
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Sides(0.14, 0.02))
n.odo.theta = M.HEADING_RAD[0]
n.mover.move_to(*n.map.centre((0, 5)), M.HEADING_RAD[0])
n.mover.update()
check("with --no-centring it ignores the walls and holds heading", n.mover.lateral == 0.0)
M.CENTRING = True
check("camera walls are OFF by default", M.USE_CAMERA_WALLS is False)

print("27) the shipped motion: no steering while driving, pivot pulses set the heading")
M.STEER_WHILE_DRIVING, M.TRIM_ENABLED = _DEFAULT_STEER, _DEFAULT_TRIM
M.STEER_MIN_RATE = _DEFAULT_MINRATE
check("defaults: steering while driving is ON (the new motors steer)", _DEFAULT_STEER is True)
check("defaults: steering has a minimum correction to clear the motors' dead band",
      M.STEER_MIN_RATE > 0.0)
check("defaults: heading trim on", _DEFAULT_TRIM is True)


class _Robot:
    """A fake clock that turns the fake robot while a pivot is commanded.

    `eff` is how much of the commanded turn really happens; pulses shorter than
    `dead_s` do nothing (a motor dead band); `stuck` means it never turns at all."""
    def __init__(self, nav, drive, eff=1.0, dead_s=0.0, stuck=False):
        self.t, self.nav, self.drive = 1000.0, nav, drive
        self.eff, self.dead_s, self.stuck = eff, dead_s, stuck
    def monotonic(self): return self.t
    def sleep(self, dt):
        v, w = self.drive.last
        if v == 0.0 and w != 0.0 and not self.stuck and dt >= self.dead_s:
            self.nav.odo.theta = M.wrap(self.nav.odo.theta + w * dt * self.eff)
        self.t += dt


def trim_run(start_err_deg, **robot_kw):
    d = LogDrive()
    n = M.Nav(d, FakeVision([None]), FakeLeds(), _Open())
    robot = _Robot(n, d, **robot_kw)
    real_time = M.time
    M.time = types.SimpleNamespace(monotonic=robot.monotonic, sleep=robot.sleep)
    try:
        n.odo.theta = M.HEADING_RAD[1] + math.radians(start_err_deg)
        n.mover.grid = True
        n.mover.heading, n.mover.target, n.mover.through = M.HEADING_RAD[1], (0.42, 0.14), False
        n.mover._front_stamp = None
        n.mover._begin_trim(robot.t)
        ticks = 0
        while n.mover.phase == M.Mover.TRIM and ticks < 80:
            n.mover.update()
            robot.sleep(0.05)
            ticks += 1
        return n, d, robot, ticks
    finally:
        M.time = real_time


n, d, robot, ticks = trim_run(-8.0)           # pointing 8 deg RIGHT of where it should
check("8 deg right of the heading: pivots LEFT", d.log and d.log[0][0] == 0.0 and d.log[0][1] > 0)
check("...and ends within the tolerance, driving",
      n.mover.phase == M.Mover.DRIVE
      and abs(M.wrap(n.mover.aim - n.odo.theta)) <= M.TRIM_TOL_RAD)
check(f"...in a handful of pulses ({n.mover.trim_pulses})", n.mover.trim_pulses <= 4)
n, d, robot, ticks = trim_run(+8.0)
check("8 deg left: pivots RIGHT and settles", d.log and d.log[0][1] < 0
      and n.mover.phase == M.Mover.DRIVE)
n, d, robot, ticks = trim_run(5.0, eff=0.4)   # the chassis only turns 40% of the command
check("a robot that turns 40% of what's commanded still converges",
      n.mover.phase == M.Mover.DRIVE and abs(M.wrap(n.mover.aim - n.odo.theta)) <= M.TRIM_TOL_RAD)
check(f"...and learned roughly that (trim_eff {n.mover.trim_eff:.2f})", 0.2 < n.mover.trim_eff < 0.8)
n, d, robot, ticks = trim_run(5.0, dead_s=0.04)   # pulses under 40 ms turn nothing
check("pulses inside a dead band get longer until they bite",
      n.mover.phase == M.Mover.DRIVE and abs(M.wrap(n.mover.aim - n.odo.theta)) <= M.TRIM_TOL_RAD)
_ev = []
_o = M.STATUS.event; M.STATUS.event = lambda m: _ev.append(m)
n, d, robot, ticks = trim_run(8.0, stuck=True)    # nothing it does turns the robot
check("a robot that never turns: gives up (not forever), says so, drives on",
      n.mover.phase == M.Mover.DRIVE and any("trim gave up" in e for e in _ev)
      and n.mover.trim_pulses <= M.TRIM_MAX_PULSES)
M.STATUS.event = _o
n, d, robot, ticks = trim_run(0.5)
check("already within tolerance: no pulse at all", d.log == [] and n.mover.phase == M.Mover.DRIVE)

# leaning towards the centre of the corridor
_half = M.SIDE_WALL_AT_CENTRE_M
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Sides(_half + 0.04, _half - 0.04))
n.mover.grid, n.mover.heading = True, M.HEADING_RAD[1]
n.mover._begin_trim(time.monotonic())
check("4 cm right of centre -> aims left of the cardinal heading",
      n.mover.aim - M.HEADING_RAD[1] > 0.1)
check("...but never more than CENTRE_AIM_MAX_RAD", n.mover.aim - M.HEADING_RAD[1] <= M.CENTRE_AIM_MAX_RAD + 1e-9)
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Sides(_half - 0.04, _half + 0.04))
n.mover.grid, n.mover.heading = True, M.HEADING_RAD[1]
n.mover._begin_trim(time.monotonic())
check("4 cm left of centre -> aims right", n.mover.aim - M.HEADING_RAD[1] < -0.1)
n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Sides(_half + 0.005, _half - 0.005))
n.mover.grid, n.mover.heading = True, M.HEADING_RAD[1]
n.mover._begin_trim(time.monotonic())
check("under 2 cm off: noise, not worth a pulse -> aims straight", n.mover.aim == M.HEADING_RAD[1])

# and the leg itself sends no yaw
d = LogDrive()
n = M.Nav(d, FakeVision([None]), FakeLeds(), _Sides(_half, _half))      # centred, facing east
n.odo.theta = M.HEADING_RAD[1]
n.mover.move_to(0.70, n.odo.y, M.HEADING_RAD[1])
n.mover.update()                                  # TRIM: already square -> straight to driving
n.mover.update()                                  # DRIVE
check("driving a leg commands forward speed and NO yaw (a speed difference can't steer it)",
      n.mover.phase == M.Mover.DRIVE and d.log and d.log[-1][0] > 0 and d.log[-1][1] == 0.0)

print("28) the blur guard setting can be renamed by the vision team without breaking nav")
_old_vs = types.SimpleNamespace(BLUR_VARIANCE_THRESHOLD=100.0)
_new_vs = types.SimpleNamespace(BLUR_RELATIVE_THRESHOLD=0.3)
_none_vs = types.SimpleNamespace()
check("finds the original name", M.blur_setting_name(_old_vs) == "BLUR_VARIANCE_THRESHOLD")
check("finds the renamed one", M.blur_setting_name(_new_vs) == "BLUR_RELATIVE_THRESHOLD")
check("copes with neither (returns None instead of raising)", M.blur_setting_name(_none_vs) is None)
check("the old name wins if both exist (it's the one nav's default of 0 = off is defined for)",
      M.blur_setting_name(types.SimpleNamespace(BLUR_VARIANCE_THRESHOLD=1.0,
                                                BLUR_RELATIVE_THRESHOLD=0.3))
      == "BLUR_VARIANCE_THRESHOLD")

print("29) stopped short of a cell centre: creep up to the wall, learn the distance scale")
_ev = []
_o = M.STATUS.event; M.STATUS.event = lambda m: _ev.append(m)


def creep_nav(front, path=1.0):
    """A nav standing in an unvisited cell facing north with `front` on the front sonar."""
    n = M.Nav(LogDrive(), FakeVision([None]), FakeLeds(), _Front(front))
    n.cell, n.facing = (0, 4), M.NORTH if hasattr(M, "NORTH") else 0
    n.odo.reset(*n.map.centre((0, 4)), M.HEADING_RAD[0])
    n.sense_from = {k: 0.0 for k in ("front", "left", "right")}
    n.odo.path_m = path
    return n


_half = M.FRONT_WALL_AT_CENTRE_M
n = creep_nav(_half + 0.10)
_y0 = n.odo.y
check("a wall 10 cm further than it should be -> starts a creep", n._front_creep(["front"]) is True)
check("...aimed 10 cm ahead", abs(n.mover.target[1] - (_y0 + 0.10)) < 1e-9)
check("...and says so, with how short it was", any("10cm short" in e and "creeping" in e for e in _ev))
n._arrived()
check("on arrival the along-track position is pulled onto the cell centre",
      abs(n.odo.y - n.map.centre((0, 4))[1]) < 1e-9 and n.sensing is True)
check("...and the cell is sensed again from there", n.sense_from is None)
n = creep_nav(_half + 0.10)
n._front_creep(["front"])
check("only one creep per cell (it can't loop)", n._front_creep(["front"]) is False)
check("reading a normal wall distance -> no creep", creep_nav(_half + 0.01)._front_creep(["front"]) is False)
check("a wall a whole cell further off isn't this cell's -> no creep",
      creep_nav(_half + 0.28)._front_creep(["front"]) is False)
n = creep_nav(_half + 0.10); n._victim_seen_at = time.monotonic()
check("a victim seen a moment ago might be what the sonar reads -> no creep",
      n._front_creep(["front"]) is False)
n = creep_nav(_half + 0.10); n.victim_cell = n.map.neighbour(n.cell, 0)
check("the next cell is a known victim cell -> no creep", n._front_creep(["front"]) is False)
M.CREEP_ENABLED = False
check("CREEP_ENABLED = False turns it off", creep_nav(_half + 0.10)._front_creep(["front"]) is False)
M.CREEP_ENABLED = True

n = creep_nav(_half + 0.10, path=1.0)
n._learn_scale(0.10)
check("10 cm out over a metre -> scale drops (odometry over-counts)",
      abs(n.odo.dscale - (1 - M.SCALE_LEARN_RATE * 0.10)) < 1e-9)
n = creep_nav(_half, path=1.0)
n._learn_scale(0.01)
check("a centimetre out is just a stop's tolerance -> scale untouched", n.odo.dscale == 1.0)
n = creep_nav(_half, path=0.1)
n._learn_scale(0.10)
check("too short a run to say anything about the scale", n.odo.dscale == 1.0)
n = creep_nav(_half, path=1.0)
n._learn_scale(-0.04)
check("overshot (wall nearer than expected) -> scale goes UP", n.odo.dscale > 1.0)
n = creep_nav(_half, path=1.0)
for _i in range(30):
    n.odo.path_m += 1.0
    n._learn_scale(0.5)
check("the scale never leaves its limits", M.SCALE_LIMITS[0] <= n.odo.dscale <= M.SCALE_LIMITS[1])
n = creep_nav(_half, path=1.0)
n.odo.dscale = 0.9
n.odo.update((0, 0)); n.odo.update((int(0.5 * M.TICKS_PER_M), int(0.5 * M.TICKS_PER_M)))
check("the odometry applies the scale to the distance it integrates (0.5 m counted -> 0.45)",
      abs(math.hypot(n.odo.x - n.map.centre((0, 4))[0], n.odo.y - n.map.centre((0, 4))[1]) - 0.45) < 0.01)

_ev.clear()
n = creep_nav(_half + 0.22, path=1.6)           # 22 cm short after a long corridor
check("after a long run (1.6 m since a wall) a 22 cm gap is within reach -> creeps",
      n._front_creep(["front"]) is True)
_ev.clear()
n = creep_nav(_half + 0.22, path=0.4)           # same gap after only 40 cm: not plausible
check("...but the same gap after only 40 cm driven isn't -> no creep, and says why",
      n._front_creep(["front"]) is False and any("too far to be this cell's wall" in e for e in _ev))
n = creep_nav(_half + 0.29, path=3.0)
check("a gap of 29 cm is a whole cell off (an open cell, then a wall): never creeps",
      n._front_creep(["front"]) is False)
_ev.clear()
n = creep_nav(_half + 0.10)
n.sense_from = {k: 0.0 for k in ("front", "left", "right")}
n._trace_stop(["front", "left", "right"])
check("each stop prints one trace line with the sonar readings and the odometry offset",
      len(_ev) == 1 and "[stop]" in _ev[0] and "odo off centre" in _ev[0]
      and f"F {(M.FRONT_WALL_AT_CENTRE_M + 0.10) * 100:.0f}" in _ev[0])
n._trace_stop(["front", "left", "right"])
check("...and only once per cell", len(_ev) == 1)
M.STATUS.event = _o

print("30) weak motors: full-power pivots, and a turn that stops short keeps turning")
check("maze pivots run at full power (raw 127)",
      MOT.to_raw(MOT.wheel_speeds(0.0, M.GRID_PIVOT_RATE)[0]) == -MOT.MAX_SPEED_RAW)
check("...the victim approach keeps its gentler floor", M.MIN_TURN_RATE_PIVOT < M.GRID_PIVOT_RATE)
check("a turn is given long enough to finish a weak pivot", M.TURN_TIMEOUT_S >= 8.0)

_ev = []
_o = M.STATUS.event; M.STATUS.event = lambda m: _ev.append(m)
n, d, robot, ticks = trim_run(-58.0, eff=0.02)    # nothing it does turns the robot much
check("a trim that can't finish a big error does NOT drive off: it turns again",
      n.mover.phase == M.Mover.TURN and any("turning again (1/" in e for e in _ev))
check("...it said how far short it was",
      any("deg short" in e and "turning again" in e for e in _ev))
check("...and that doesn't loop forever: it's bounded by TURN_MAX_RETRIES",
      M.TURN_MAX_RETRIES >= 1)

# once the retries are used up it does drive on (a robot that can't turn at all mustn't hang)
_ev.clear()
d2 = LogDrive()
n2 = M.Nav(d2, FakeVision([None]), FakeLeds(), _Open())
robot2 = _Robot(n2, d2, stuck=True)
_real_time = M.time
M.time = types.SimpleNamespace(monotonic=robot2.monotonic, sleep=robot2.sleep)
try:
    n2.odo.theta = M.HEADING_RAD[1] - math.radians(80)
    n2.mover.grid, n2.mover.heading, n2.mover.target = True, M.HEADING_RAD[1], (0.42, 0.14)
    n2.mover.through, n2.mover.turn_retries = False, 0
    n2.mover._front_stamp = None
    n2.mover._begin_trim(robot2.t)
    for _i in range(4000):
        n2.mover.update()
        robot2.sleep(0.05)
        if n2.mover.phase == M.Mover.DRIVE:
            break
finally:
    M.time = _real_time
check("a robot that never turns, after all the retries, drives on rather than hanging",
      n2.mover.phase == M.Mover.DRIVE and n2.mover.turn_retries == M.TURN_MAX_RETRIES)
M.STATUS.event = _o

print("31) --test-mode: find the victim, flash all LEDs, return home (the mock collection)")


class _BlinkLeds(FakeLeds):
    def __init__(self): super().__init__(); self.snaps = []
    def _snap(self): self.snaps.append((self.g, self.y, self.r))
    def green(self, on): super().green(on); self._snap()
    def yellow(self, on): super().yellow(on); self._snap()
    def red(self, on): super().red(on); self._snap()


def run_collection(seconds=8.0):
    """Drive _at_victim for `seconds` of fake time; returns (nav, drive, leds, frames, t_done)."""
    d = FakeDrive(); l = _BlinkLeds()
    n = M.Nav(d, FakeVision([None]), l, _Open())
    n._enter(M.AT_VICTIM)
    n.arrived_at = _clk.t
    t0, frames, done = _clk.t, [], None
    while _clk.t - t0 < seconds:
        _clk.t += 0.05
        n._at_victim()
        if n.state != M.AT_VICTIM:
            done = _clk.t - t0
            break
        frames.append((l.g, l.y, l.r))
    return n, d, l, frames, done


check("defaults: the mock collection is on, --test-mode is off",
      M.MOCK_COLLECTION is True and M.TEST_MODE is False)
_old_t = M.time
_clk = types.SimpleNamespace(t=500.0)
M.time = types.SimpleNamespace(monotonic=lambda: _clk.t, sleep=lambda s: None)
_ev = []
_o = M.STATUS.event; M.STATUS.event = lambda m: _ev.append(m)
_mock, _tm = M.MOCK_COLLECTION, M.TEST_MODE
try:
    # --test-mode with the mock switched off (i.e. once a real mechanism is the default)
    M.MOCK_COLLECTION, M.TEST_MODE = False, True
    n, d, l, frames, done = run_collection()
    check("--test-mode: finds the victim, stands still and flashes ALL the LEDs together",
          done is not None and d.last == (0.0, 0.0)
          and all(f in ((True, True, True), (False, False, False)) for f in frames)
          and frames.count((True, True, True)) >= 20 and frames.count((False, False, False)) >= 20)
    check(f"...for the 5 s collection time ({done:.2f} s), then heads HOME carrying it",
          4.9 <= done <= 5.2 and n.carrying and n.state == M.RETURN)
    check("...red alone, solid, on the way home", l.r is True and not l.g and not l.y)
    check("...and says it's the test-mode mock, once",
          sum("TEST MODE" in e and "mock collection" in e for e in _ev) == 1)

    # no flag, and the mock off: the real mechanism's turn -- LEDs left as the state set them
    M.MOCK_COLLECTION, M.TEST_MODE = False, False
    _ev.clear()
    n, d, l, frames, done = run_collection()
    check("no mock and no flag: no LED flashing (that's the real mechanism's to do)",
          all(f == (True, False, False) for f in frames)
          and not any("mock collection" in e for e in _ev))
    check("...it still waits out the collection and goes home", n.carrying and n.state == M.RETURN)

    # the shipped default (mock on) needs no flag
    M.MOCK_COLLECTION, M.TEST_MODE = True, False
    n, d, l, frames, done = run_collection()
    check("default: flashes all LEDs and goes home with no flag at all",
          frames.count((True, True, True)) >= 20 and n.carrying and n.state == M.RETURN)
finally:
    M.time = _old_t
    M.MOCK_COLLECTION, M.TEST_MODE = _mock, _tm
    M.STATUS.event = _o

# test mode finishes after ONE victim; the normal run goes back out for the next
for _tm, _want_done in ((True, True), (False, False)):
    M.TEST_MODE = _tm
    _ev2 = []
    _o2 = M.STATUS.event; M.STATUS.event = lambda m: _ev2.append(m)
    _t1 = M.time
    _c2 = types.SimpleNamespace(t=900.0)
    M.time = types.SimpleNamespace(monotonic=lambda: _c2.t, sleep=lambda s: None)
    try:
        n = M.Nav(FakeDrive(), FakeVision([None]), FakeLeds(), _Open())
        n.state, n.carrying = M.AT_BASE, True
        n.release.start()
        _c2.t += M.RELEASE_TIME_S + 0.1
        n._at_base()
    finally:
        M.time = _t1; M.STATUS.event = _o2; M.TEST_MODE = False
    check(f"first victim home{' in --test-mode' if _tm else ' (normal run)'}: "
          f"{'finished, saying why' if _want_done else 'back out to search for the next'}",
          (n.state == M.DONE and any("TEST MODE" in e and "1 victim rescued" in e for e in _ev2))
          if _want_done else n.state == M.SEARCH)
check("--test-mode's target is one victim", M.TEST_MODE_VICTIMS == 1)

print("35) steering while driving on the new motors: a minimum correction clears the dead band")


def steer_w(heading_err_deg):
    """The yaw rate the leg commands with the robot `heading_err_deg` off (centred, east)."""
    d = LogDrive()
    n = M.Nav(d, FakeVision([None]), FakeLeds(), _Sides(_half, _half))
    n.odo.theta = M.HEADING_RAD[1] + math.radians(heading_err_deg)
    n.mover.grid, n.mover.heading = True, M.HEADING_RAD[1]
    n.mover.target, n.mover.through = (n.odo.x + 0.28, n.odo.y), False
    n.mover._front_stamp = None
    n.mover._begin_drive(time.monotonic())
    n.mover.update()
    return d.log[-1][1]


check("defaults: STEER_MIN_RATE is 0.5 rad/s", M.STEER_MIN_RATE == 0.5)
check("1 deg off: inside the tolerance, left alone", steer_w(1.0) == 0.0)
w = steer_w(-4.0)
check(f"4 deg right of heading: turns LEFT at least the minimum ({w:+.2f} rad/s)",
      w >= M.STEER_MIN_RATE - 1e-9)
w = steer_w(4.0)
check(f"4 deg left of heading: turns RIGHT at least the minimum ({w:+.2f} rad/s)",
      w <= -M.STEER_MIN_RATE + 1e-9)
check("a big error still just uses the proportional command, capped at MAX_STEER_RATE",
      abs(steer_w(40.0)) <= M.MAX_STEER_RATE + 1e-9 and abs(steer_w(40.0)) >= M.STEER_MIN_RATE)
M.STEER_MIN_RATE = 0.0
check("with the minimum off it's plain proportional again (a small error -> small command)",
      0.0 < abs(steer_w(4.0)) < M.STEER_MIN_RATE + 0.5 and abs(steer_w(4.0)) < 0.4)
M.STEER_MIN_RATE = _DEFAULT_MINRATE

print("36) the map's outer edge: say so when the sonar sees open space where the map ends")
_ev = []
_o = M.STATUS.event; M.STATUS.event = lambda m: _ev.append(m)
n = M.Nav(FakeDrive(), FakeVision([None]), FakeLeds(), _Open())    # base (0, 6): west is the edge
n.cell, n.facing = (0, 4), 0
n.sense_from = {k: 0.0 for k in ("front", "left", "right")}
n._check_edge_is_open(M.WEST if hasattr(M, "WEST") else 3, 0.55, 0.159)
check("open space (55 cm) where the map says the edge is -> warns, naming base cell and the flags",
      any("edge of the map" in e and "--base-cell" in e and "55cm" in e for e in _ev))
_ev.clear()
n._check_edge_is_open(3, 0.55, 0.159)
check("...only once per edge", not _ev)
_ev.clear()
n._check_edge_is_open(3, 0.07, 0.159)
check("a wall there (7 cm) -> nothing said", not _ev)
n2 = M.Nav(FakeDrive(), FakeVision([None]), FakeLeds(), _Open())
n2.cell, n2.facing = (3, 3), 0
n2._check_edge_is_open(3, 0.55, 0.159)
check("an interior edge isn't the map's edge -> nothing said", not _ev)
M.STATUS.event = _o

print("32) status LEDs, per the project description")
# Yellow = searching/exploring. Green = victim detected & collection in progress.
# Red = returning to base with a victim. Exactly one lit at a time; none when finished.
_RULES = [
    (M.SEARCH, False, "yellow"), (M.SEEK, False, "green"), (M.APPROACH, False, "green"),
    (M.CLOSING, False, "green"), (M.AT_VICTIM, False, "green"),
    (M.RETURN, True, "red"), (M.AT_BASE, True, "red"),
    (M.RETURN, False, "yellow"),      # heading home with nothing aboard: still just exploring
]
for _state, _carry, _want in _RULES:
    l = FakeLeds()
    n = M.Nav(FakeDrive(), FakeVision([None]), l, _Open())
    n.state, n.carrying = _state, _carry
    n._set_leds()
    lit = [c for c, v in (("green", l.g), ("yellow", l.y), ("red", l.r)) if v]
    check(f"{_state}{' + carrying' if _carry else ''}: only {_want} is lit", lit == [_want])

l = FakeLeds()
n = M.Nav(FakeDrive(), FakeVision([None]), l, _Open())
n.state = M.SEARCH; n._set_leds()
n._enter(M.DONE); n._done()
check("finished (DONE): everything is off", not (l.g or l.y or l.r))
l = FakeLeds()
n = M.Nav(FakeDrive(), FakeVision([None]), l, _Open())
n.step()
check("lit from the very first tick: yellow while it starts searching",
      l.y is True and not l.g and not l.r)

# a victim collected: green all through collecting, red the moment it's aboard
l = FakeLeds()
n = M.Nav(FakeDrive(), FakeVision([None]), l, _Open())
n._enter(M.AT_VICTIM)
check("stopped at the victim, collecting: green", l.g is True and not l.y and not l.r)
_t0 = M.time
_c = types.SimpleNamespace(t=10.0)
M.time = types.SimpleNamespace(monotonic=lambda: _c.t, sleep=lambda s: None)
try:
    n.arrived_at = _c.t
    n.rescue.start()
    _c.t += M.RESCUE_TIME_S + 0.1
    n._at_victim()
finally:
    M.time = _t0
check("the moment it's picked up (carrying): red, green and yellow off",
      n.carrying and l.r is True and not l.g and not l.y)
n.state = M.AT_BASE
n.release.start()
n._set_leds()
check("at base releasing it: still red", l.r is True and not l.g and not l.y)
n.carrying = False
n._enter(M.SEARCH)
check("released, back out to search: yellow", l.y is True and not l.g and not l.r)

print("33) the (mock) collection: 5 s stood still, all three LEDs flashing together, then home")
check("the mock collection lasts 5 seconds", M.RESCUE_TIME_S == 5.0)


class _Led3(FakeLeds):
    """Records all three LEDs after every call, to see them change together."""
    def __init__(self):
        super().__init__(); self.snap = []
    def _rec(self): self.snap.append((self.g, self.y, self.r))
    def green(self, on): super().green(on); self._rec()
    def yellow(self, on): super().yellow(on); self._rec()
    def red(self, on): super().red(on); self._rec()


_t0 = M.time
_c = types.SimpleNamespace(t=100.0)
M.time = types.SimpleNamespace(monotonic=lambda: _c.t, sleep=lambda s: None)
_o = M.STATUS.event; M.STATUS.event = lambda m: None
try:
    d = FakeDrive(); l = _Led3()
    n = M.Nav(d, FakeVision([None]), l, _Open())
    n._enter(M.AT_VICTIM)
    n.arrived_at = _c.t
    frames, t_end = [], None
    for _i in range(400):
        _c.t += 0.05
        n._at_victim()
        if n.state != M.AT_VICTIM:
            t_end = _c.t - 100.0
            break
        frames.append((l.g, l.y, l.r))
    check(f"it stays put and 'collects' for 5 s (took {t_end:.2f} s)", t_end is not None and 4.9 <= t_end <= 5.2)
    check("...with the drive stopped throughout", d.last == (0.0, 0.0))
    check("...all three LEDs always in step: all lit or all dark",
          all(f in ((True, True, True), (False, False, False)) for f in frames))
    check("...and they actually flash (both states, many times)",
          frames.count((True, True, True)) >= 20 and frames.count((False, False, False)) >= 20)
    flips = sum(1 for a, b in zip(frames, frames[1:]) if a != b)
    check(f"...at about {M.COLLECT_FLASH_HZ:g} Hz ({flips / 2 / 5:.1f} Hz)",
          abs(flips / 2 / 5 - M.COLLECT_FLASH_HZ) < 0.8)
    check("when it's done it has the victim and is heading home", n.carrying and n.state == M.RETURN)
    check("...red alone, solid, for the trip home",
          l.r is True and not l.g and not l.y)
finally:
    M.time = _t0
    M.STATUS.event = _o

print("34) ultrasonic pins: each name is where the sensor really sits")
# The sensors were found crossed: the real LEFT was on 23/24 (named "front"), the real RIGHT
# on 5/6 (named "left"), the real FRONT on 22/27 (named "right"). Pinned so it can't slip back.
check("front sensor is on 22/27", M.ULTRASONIC_PINS["front"] == (22, 27))
check("left sensor is on 23/24", M.ULTRASONIC_PINS["left"] == (23, 24))
check("right sensor is on 5/6", M.ULTRASONIC_PINS["right"] == (5, 6))
check("no two sensors share a pin (and none uses the LEDs' 16/20/21 or I2C's 2/3)",
      len({p for pair in M.ULTRASONIC_PINS.values() for p in pair}) == 6
      and not ({p for pair in M.ULTRASONIC_PINS.values() for p in pair} & {16, 20, 21, 2, 3}))
import ultrasonic_test as _UT
check("ultrasonic_test.py uses the same pins as main.py",
      all(_UT.SENSOR_PINS[k] == M.ULTRASONIC_PINS[k] for k in ("front", "left", "right")))

print("37) the victim's WALL MARKER: green, and go and look in the cell in front of it")
# The marker hangs on the wall, so the camera can't range it -- but it says a victim is near.
# With no victim object in view, nav places a victim half a cell in front of the marker's wall
# (wall range from the front sonar), lights green and goes to look (SEEK).
class MarkerVision(FakeVision):
    """FakeVision plus the victim's wall marker in view (bearing in deg), or not (None)."""
    def __init__(self, bearing=None, script=()):
        super().__init__(list(script))
        self.set(bearing)
    def set(self, bearing):
        self.marker = None if bearing is None else M.Victim(bearing, None, (200, 40, 60, 60), 3600)
        self.n_marker = 0 if bearing is None else 1


def marker_nav(bearing=0.0, front=0.80, script=()):
    v = MarkerVision(bearing, script)
    return M.Nav(FakeDrive(ticks=(0, 0)), v, FakeLeds(), FakeSonar(front=front)), v


def marker_cell(n, front):
    """Where the cue should put the victim, worked out from the geometry independently."""
    wall = front + (M.CELL_M / 2 - M.FRONT_WALL_AT_CENTRE_M)
    return n.map.cell_at(n.odo.x, n.odo.y + wall - M.MARKER_CELL_INSET_M)   # facing north

n, v = marker_nav(0.0, 0.80)
want = marker_cell(n, 0.80)
run(n, M.DETECTION_DEBOUNCE + 1)
check("marker confirmed, nothing on the floor -> SEEK", n.state == M.SEEK)
check("...green on, yellow off", n.leds.g is True and n.leds.y is False)
check(f"...a victim placed in the cell in front of the marker's wall {want}", n.victim_cell == want)
check("...and that cell remembered as the marker's", n.marker_cell == want)

n, v = marker_nav(0.0, 0.80)
run(n, M.DETECTION_DEBOUNCE - 1)
check("two frames is not enough (debounced)", n.state == M.SEARCH and n.leds.g is not True)

n, v = marker_nav(25.0, 0.80)
run(n, 6)
check("marker well off the nose -> the sonar isn't ranging its wall: no cue", n.state == M.SEARCH)

n, v = marker_nav(0.0, None)
run(n, 6)
check("no front sonar reading -> can't place it, no cue (and green stays off)",
      n.state == M.SEARCH and n.leds.g is not True)

n, v = marker_nav(0.0, 0.80)
n.done_cells.add(marker_cell(n, 0.80))
run(n, 6)
check("a cell already collected from is ignored (its marker is still on the wall)",
      n.state == M.SEARCH)

n, v = marker_nav(0.0, 0.80)
n.seek_tries[marker_cell(n, 0.80)] = M.SEEK_ATTEMPTS
run(n, 6)
check("a cell that has failed its sonar check SEEK_ATTEMPTS times is not tried again",
      n.state == M.SEARCH)

n, v = marker_nav(0.0, 0.80)
n.carrying = True
run(n, 6)
check("carrying a victim: markers are ignored", n.state != M.SEEK)

_enabled = M.MARKER_CUE_ENABLED
M.MARKER_CUE_ENABLED = False
n, v = marker_nav(0.0, 0.80)
run(n, 6)
M.MARKER_CUE_ENABLED = _enabled
check("MARKER_CUE_ENABLED = False turns it off", n.state == M.SEARCH)

n, v = marker_nav(0.0, 0.80, script=[V(bearing=0.0)] * 8)
run(n, M.DETECTION_DEBOUNCE + 1)
check("a victim OBJECT in view still wins: the usual path, not the marker's",
      n.state == M.SEEK and n.marker_cell is None)

n, v = marker_nav(0.0, 0.80)
n.victim_cell, n.marker_cell = (0, 2), (0, 3)
n._enter(M.AT_VICTIM)
n.rescue.started = M.time.monotonic() - 100.0
n._at_victim()
check("collecting records both the victim's cell and the marker's as done",
      {(0, 2), (0, 3)} <= n.done_cells and n.marker_cell is None)

print("37a) telling the marker from the object")
vv = M.VictimVision.__new__(M.VictimVision)            # no camera: just the geometry
vv.frame_h = 480
hy = M.horizon_row(480)
check("a marker-template hit is a marker",
      vv._is_marker({"class": "victim", "is_ground_object": False, "bbox": (10, 100, 50, 50)}))
check("a victim-coloured blob up the wall (base above the horizon) is a marker",
      vv._is_marker({"class": "victim", "is_ground_object": True, "bbox": (10, int(hy) - 90, 50, 50)}))
check("...but one standing on the floor is the OBJECT, not a marker",
      not vv._is_marker({"class": "victim", "is_ground_object": True, "bbox": (10, int(hy) + 60, 50, 50)}))
check("another class (the base marker, say) is not a victim marker",
      not vv._is_marker({"class": "base", "is_ground_object": False, "bbox": (10, 100, 50, 50)}))
import types as _types
check("the status line says when a marker is in view",
      "marker 1" in M.vision_summary(_types.SimpleNamespace(n_marker=1)))
check("...and says nothing when there isn't one",
      "marker" not in M.vision_summary(_types.SimpleNamespace(n_marker=0)))

print("37b) the victim goes in the cell BEFORE the marker's wall -- not on it, not past it")
# The geometry here is worked out from the maze, not from the code under test: the robot
# sits at the centre of the base cell facing north and the marker's wall is the k-th cell
# boundary ahead of it. The front sonar sits (CELL_M/2 - FRONT_WALL_AT_CENTRE_M) ahead of
# the turning centre, so it reads the wall distance minus that. The victim must end up in
# the cell just before the wall (k cells north of the base cell) even when the sonar reads
# a few centimetres long or short.
for k in range(0, 4):
    bad = []
    for err in (-0.05, 0.0, 0.05, 0.10, 0.15):
        wall_from_centre = M.CELL_M / 2 + k * M.CELL_M
        front = wall_from_centre - (M.CELL_M / 2 - M.FRONT_WALL_AT_CENTRE_M) + err
        n, v = marker_nav(0.0, front)
        run(n, M.DETECTION_DEBOUNCE + 1)
        want = (M.BASE_CELL[0], M.BASE_CELL[1] - k)
        if not (n.state == M.SEEK and n.victim_cell == want):
            bad.append(f"sonar {err*100:+.0f}cm -> {n.victim_cell} (want {want})")
    check(f"marker's wall {k} cell(s) ahead: victim in the cell before it, even if the sonar "
          f"is off by -5..+15cm" + (f"  [{'; '.join(bad)}]" if bad else ""), not bad)

print("37c) a victim OBJECT and the marker both in view: the object's cell is the placement")
# The camera range for the object (0.40 m) puts it 2 cells ahead of the base cell; the
# marker's wall (sonar 1.20 m) would put the victim 4 cells ahead. The object must win.
obj_cell = (M.BASE_CELL[0], M.BASE_CELL[1] - 2)
marker_only = marker_nav(0.0, 1.20)[0]
run(marker_only, M.DETECTION_DEBOUNCE + 1)
check("(setup) the marker alone would have put the victim somewhere else",
      marker_only.victim_cell not in (None, obj_cell))

n, v = marker_nav(0.0, 1.20, script=[V(dist=0.40)] * 8)
run(n, M.DETECTION_DEBOUNCE + 1)
check("both seen from the first frame -> victim placed where the object is",
      n.state == M.SEEK and n.victim_cell == obj_cell and n.marker_cell is None)

n, v = marker_nav(0.0, 1.20, script=[None, None] + [V(dist=0.40)] * 8)
run(n, M.DETECTION_DEBOUNCE + 3)
check("marker seen first, object a frame or two later -> the marker doesn't jump the gun",
      n.state == M.SEEK and n.victim_cell == obj_cell and n.marker_cell is None)

n, v = marker_nav(0.0, 1.20, script=[None] * 4)
run(n, 4)
first = n.victim_cell
check("(setup) the marker cue fired first, with no object in view",
      n.state == M.SEEK and n.marker_cell == first and first != obj_cell)
v.script.extend([V(dist=0.40)] * 5)
run(n, 3)
check("then the object appears while seeking -> its cell replaces the marker's",
      n.victim_cell == obj_cell)
check("...and the marker's cell is still remembered, so it is ignored after the rescue",
      n.marker_cell == first)

print("38) an OBSTACLE to one side of the corridor: slide to the other side and go past")
_s38_saved = (M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE)
M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE = True, False, 0.0   # plain proportional
# Field facts (measured): the obstacle is ~12 cm wide, the cell 28 cm, and the robot has ~2 cm
# either side of it when alongside -- so the obstacle stands against one wall and the robot has
# to run down the middle of the 16 cm gap, OBSTACLE_PASS_OFFSET_M = 6 cm off the centreline.
class ObVision(FakeVision):
    """FakeVision with a floor obstacle in view: bearing in deg (+ = right), camera range m."""
    def __init__(self, bearing=None, dist=0.25):
        super().__init__([])
        self.marker, self.n_marker = None, 0
        self.set(bearing, dist)
    def set(self, bearing, dist=0.25):
        self.obstacle = None if bearing is None else M.Victim(bearing, dist, (200, 150, 60, 60), 3600)


class GapSonar(FakeSonar):
    """Front + side sonars with FRESH side stamps (FakeSonar's counter would read as stale)."""
    def __init__(self, left=M.SIDE_WALL_AT_CENTRE_M, right=M.SIDE_WALL_AT_CENTRE_M, front=1.0):
        super().__init__(front=front)
        self.l, self.r = left, right
        self.sensors = {"front": 1, "left": 1, "right": 1}
    def get(self, name):
        return {"front": self.front, "left": self.l, "right": self.r}[name]
    def stamp(self, name):
        return self._n if name == "front" else time.monotonic()


def ob_nav(bearing=None, dist=0.25, left=M.SIDE_WALL_AT_CENTRE_M, right=M.SIDE_WALL_AT_CENTRE_M,
           front=1.0, ticks=8):
    v, s, d = ObVision(bearing, dist), GapSonar(left, right, front), FakeDrive(ticks=(0, 0))
    n = M.Nav(d, v, FakeLeds(), s)
    run(n, ticks)
    return n, v, s, d


def steer(d):
    return 0 if abs(d.last[1]) < 1e-6 else (1 if d.last[1] > 0 else -1)


n, v, s, d = ob_nav()
check("(setup) nothing in view, centred: drives dead straight", steer(d) == 0 and d.last[0] > 0)

n, v, s, d = ob_nav(bearing=30.0)
check("obstacle off to the RIGHT -> a sidestep is latched", n.mover.avoiding)
check(f"...to the LEFT by {M.OBSTACLE_PASS_OFFSET_M*100:.0f}cm (half the obstacle width)",
      abs(n.mover.avoid[0] + M.OBSTACLE_PASS_OFFSET_M) < 1e-9)
check("...and steers LEFT, away from it, while still driving forward", steer(d) == 1 and d.last[0] > 0)

n, v, s, d = ob_nav(bearing=-30.0)
check("obstacle off to the LEFT -> steers RIGHT", n.mover.avoiding and steer(d) == -1 and d.last[0] > 0)

n, v, s, d = ob_nav(bearing=0.0)
check("obstacle dead ahead (centred): too little room to squeeze past, so no sidestep",
      not n.mover.avoiding)
n, v, s, d = ob_nav(bearing=5.0)
check("...nor when it is only ~2cm off centre (inside OBSTACLE_MIN_SIDE_M)", not n.mover.avoiding)

n, v, s, d = ob_nav(bearing=30.0, dist=None)
check("camera can't range it -> an assumed range is used, and it still sidesteps",
      n.mover.avoiding and steer(d) == 1)

n, v, s, d = ob_nav(bearing=30.0, dist=M.OBSTACLE_SEE_MAX_M + 0.2)
check("an obstacle further off than OBSTACLE_SEE_MAX_M is not acted on yet", not n.mover.avoiding)

# Already on the line through the middle of the gap: nothing left to correct.
gap_l = M.SIDE_WALL_AT_CENTRE_M - M.OBSTACLE_PASS_OFFSET_M
gap_r = M.SIDE_WALL_AT_CENTRE_M + M.OBSTACLE_PASS_OFFSET_M
n, v, s, d = ob_nav(bearing=30.0, left=gap_l, right=gap_r)
check("obstacle on the right and already 6cm over to the left -> on target, drives straight",
      n.mover.avoiding and steer(d) == 0)

# Alongside: the obstacle-side sonar reads the obstacle at ~2cm, the far wall reads ~2cm.
n, v, s, d = ob_nav(bearing=30.0)
s.l = s.r = 0.02
v.set(None)                      # the camera has lost it by now
run(n, 3)
check("alongside it (both sonars ~2cm) -> the two-wall centring holds it in the gap, straight on",
      n.mover.avoiding and steer(d) == 0 and d.last[0] > 0)

# Past it: the sidestep is dropped, and the usual centring pulls back to the middle.
n, v, s, d = ob_nav(bearing=30.0)
shift, until = n.mover.avoid
n.mover._avoid_shift(until - 0.01)
check("once past the obstacle (plus OBSTACLE_CLEAR_M) the sidestep ends", not n.mover.avoiding)

# The front sonar sees the obstacle on the way in: that must not make it give up the cell.
n, v, s, d = ob_nav(bearing=30.0)
s.front = 0.15                    # the sonar now sees the obstacle (or the camera's seen it first)
run(n, 12)
check("sidestepping: the front sonar reading the obstacle at 15cm is not 'blocked'",
      n.mover.phase == M.Mover.DRIVE and d.last[0] > 0)
ahead_cell = (M.BASE_CELL[0], M.BASE_CELL[1] - 1)
check("(setup) so far the cell ahead is still open", ahead_cell not in n.map.blocked)
s.front = M.OBSTACLE_FRONT_STOP_M - 0.01
run(n, 12)
check("...but nearly touching it IS blocked: backs out rather than collide (at 6cm it reads "
      "like a wall at the cell edge, so that is what gets recorded)",
      ahead_cell in n.map.blocked or n.map.wall(M.BASE_CELL, 0) is True)

# The same obstacle with no sidestep (dead ahead): the old behaviour, blocked and detoured.
n, v, s, d = ob_nav(bearing=0.0)
s.front = 0.15
run(n, 12)
check("obstacle dead ahead at 15cm: still declared blocked (cell marked, detoured as before)",
      (M.BASE_CELL[0], M.BASE_CELL[1] - 1) in n.map.blocked)

# Switches.
_av = M.OBSTACLE_AVOID
M.OBSTACLE_AVOID = False
n, v, s, d = ob_nav(bearing=30.0)
M.OBSTACLE_AVOID = _av
check("OBSTACLE_AVOID = False (--no-avoid) turns it off", not n.mover.avoiding and steer(d) == 0)

# A sidestep belongs to one leg: the next move starts clean.
n, v, s, d = ob_nav(bearing=30.0)
n.mover.move_to(*n.map.centre((M.BASE_CELL[0], M.BASE_CELL[1] - 1)), 0.0)
check("a new move clears the old sidestep", not n.mover.avoiding)

# The vision wrapper picks the obstacle out of the detections: nearest, on the floor only.
class _Sys:                         # stands in for VisionSystem
    def __init__(self, dets): self.dets = dets
    def _blur_variance(self, f): return 1000.0
    def classify_frame(self, f): return self.dets

class _Cam:
    def read(self): return _Frame()
class _Frame:
    shape = (480, 640, 3)

def det(cls, y, h, ang, dist, ground=True):
    return {"class": cls, "bbox": (300, y, 60, h), "angle_deg": ang, "distance_m": dist,
            "is_ground_object": ground, "confidence": 0.8}

vv = M.VictimVision.__new__(M.VictimVision)
vv.camera, vv.system, vv._placeholder = _Cam(), None, False
vv.frame_w = vv.frame_h = None; vv.last_frame = None; vv.last_detections = []
vv.last_blur = None; vv.blur_rejected = False; vv.class_counts = {}
vv.n_victim_raw = vv.n_victim_kept = vv.n_marker = 0; vv.marker = vv.obstacle = None
floor_y = 480 // 2 + 60            # base well below the horizon (image centre row)
vv.system = _Sys([det("obstacle", floor_y, 60, 20.0, 0.30),
                  det("obstacle", floor_y - 40, 60, -10.0, 0.50),     # further: base higher
                  det("obstacle", 100, 60, 5.0, None),                # base above the horizon
                  det("ramp", floor_y, 60, 0.0, 0.2)])
vv.look()
check("VictimVision.obstacle: the NEAREST floor obstacle (lowest base), not a ramp, not a wall-high one",
      vv.obstacle is not None and abs(vv.obstacle.bearing_deg - 20.0) < 1e-9
      and abs(vv.obstacle.distance_m - 0.30) < 1e-9)
vv.system = _Sys([det("ramp", floor_y, 60, 0.0, 0.2)])
vv.look()
check("...and None when there is no obstacle in view", vv.obstacle is None)
M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE = _s38_saved

print("39) after collecting a victim: stand still, recheck the position against the walls")
_s39_saved = (M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE, M.MOCK_COLLECTION, M.TEST_MODE)
M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE = True, False, 0.0
M.MOCK_COLLECTION, M.TEST_MODE = False, False        # the victim really is aboard, not in front


class RecheckSonar(GapSonar):
    """Sonars whose readings only count as 'fresh' once .ready is set (to test the wait)."""
    def __init__(self, **kw):
        super().__init__(**kw); self.ready = True
    def since(self, name, t):
        return (self.ready, self.get(name)) if name in self.sensors else (False, None)


def recheck_nav(front, left=M.SIDE_WALL_AT_CENTRE_M, right=M.SIDE_WALL_AT_CENTRE_M,
                theta=None, x=None, y=None, cell=(0, 3)):
    """A Nav in `cell`, off the grid and facing north, whose odometry says (x, y)."""
    s = RecheckSonar(left=left, right=right, front=front)
    n = M.Nav(FakeDrive(ticks=(0, 0)), FakeVision([]), FakeLeds(), s)
    cx, cy = n.map.centre(cell)
    n.cell = cell
    n.odo.reset(cx if x is None else x, cy if y is None else y,
                M.HEADING_RAD[MZ.NORTH] if theta is None else theta)
    return n, s, cx, cy


# Cell (0,3): centre y = 0.98, its north wall at y = 1.12. A robot truly 4cm SOUTH of the
# centre has the wall 0.18 m from its centre, so the front sonar (8cm ahead of it) reads 0.10.
true_y, wall_y = 0.98 - 0.04, 1.12
front_read = wall_y - (true_y + (M.CELL_M / 2 - M.FRONT_WALL_AT_CENTRE_M))
n, s, cx, cy = recheck_nav(front=front_read, y=0.99)             # odometry is 5cm out
n._fix_position({"front": front_read})
check("wall ahead: the position along the corridor is pulled onto the truth (±1cm)",
      abs(n.odo.y - true_y) < 0.01 and abs(n.odo.x - cx) < 1e-9)

n, s, cx, cy = recheck_nav(front=0.06, y=0.98 + 0.05)           # at the centre; odometry 5cm long
n._fix_position({"front": 0.06})
check("a centred stop (front reads 6cm) puts it back on the centre", abs(n.odo.y - cy) < 0.01)

n, s, cx, cy = recheck_nav(front=0.17, y=0.98)                  # fits no edge: ~11cm from the nearest
n._fix_position({"front": 0.17})
check("a front reading that matches no cell edge is ignored (a victim, the mechanism)",
      abs(n.odo.y - 0.98) < 1e-9)

n, s, cx, cy = recheck_nav(front=1.0, y=0.98)
n._fix_position({"front": None})
check("no echo ahead: nothing to check against, position untouched",
      abs(n.odo.y - 0.98) < 1e-9 and abs(n.odo.x - cx) < 1e-9)

# Sideways: truly 3cm right of centre -> left wall reads 3cm further, right wall 3cm nearer.
n, s, cx, cy = recheck_nav(front=1.0)
l, r = M.SIDE_WALL_AT_CENTRE_M + 0.03, M.SIDE_WALL_AT_CENTRE_M - 0.03
n._fix_position({"front": None, "left": l, "right": r})
check("side walls: the position across the corridor is pulled onto the truth (+3cm right)",
      abs(n.odo.x - (cx + 0.03)) < 0.005 and abs(n.odo.y - cy) < 1e-9)
n, s, cx, cy = recheck_nav(front=1.0, x=0.14 + 0.2)               # absurd odometry: 20cm over
n._fix_position({"front": None, "left": l, "right": r})
check("a side correction bigger than RECHECK_SIDE_MAX_M is not believed",
      abs(n.odo.x - (n.map.centre((0, 3))[0] + 0.2)) < 1e-9)

# Skewed heading: the sonars say nothing useful about position along the grid.
n, s, cx, cy = recheck_nav(front=front_read, y=0.99, theta=M.HEADING_RAD[MZ.NORTH] + 0.5)
n._fix_position({"front": front_read, "left": l, "right": r})
check("pointing ~29 deg off a grid direction: left alone",
      abs(n.odo.y - 0.99) < 1e-9 and abs(n.odo.x - cx) < 1e-9)

# Facing east: the same maths in another direction. Cell (2,3): centre x = 0.70, east wall x = 0.84.
# Truly 3cm WEST of the centre (x = 0.67); the odometry says 0.72.
n, s, cx, cy = recheck_nav(front=1.0, cell=(2, 3), theta=M.HEADING_RAD[M.EAST], x=0.72)
east_read = 0.84 - (0.67 + (M.CELL_M / 2 - M.FRONT_WALL_AT_CENTRE_M))
n._fix_position({"front": east_read})
check("facing EAST: along-track fix uses x, and lands on the truth",
      abs(n.odo.x - 0.67) < 0.01 and abs(n.odo.y - cy) < 1e-9)
# Facing west the wall is at x = 0.56 (the west edge of column 2); truly 3cm EAST (x = 0.73);
# the odometry says 0.68.
n, s, cx, cy = recheck_nav(front=1.0, cell=(2, 3), theta=M.HEADING_RAD[M.WEST], x=0.68)
west_read = 0.73 - 0.56 - (M.CELL_M / 2 - M.FRONT_WALL_AT_CENTRE_M)
n._fix_position({"front": west_read})
check("facing WEST: the sign flips correctly (lands on the truth)",
      abs(n.odo.x - 0.73) < 0.01 and abs(n.odo.y - cy) < 1e-9)

# The mock collection leaves the victim in front of the sonar (10cm): that is not a wall.
M.MOCK_COLLECTION = True
n, s, cx, cy = recheck_nav(front=M.STOP_DISTANCE_M, y=0.98 + 0.02)
n._fix_position({"front": M.STOP_DISTANCE_M})
check("(mock collection) a front reading at the victim's stop distance is not used",
      abs(n.odo.y - 1.00) < 1e-9)
M.MOCK_COLLECTION = False

# The whole sequence through step(): pickup -> no movement until fresh readings -> corrected -> rejoin.
n, s, cx, cy = recheck_nav(front=front_read, y=0.99)
s.ready = False
n._enter(M.AT_VICTIM)
n.rescue.started = time.monotonic() - 100.0                      # collection time is up
n.step()
check("pickup done -> RETURN, and a recheck is pending", n.state == M.RETURN and n.rechecking)
d = n.drive
run(n, 5)
check("...no fresh readings yet: stands still, odometry untouched",
      n.rechecking and d.last == (0.0, 0.0) and abs(n.odo.y - 0.99) < 1e-9)
s.ready = True
n.step()
check("...fresh readings arrive: the position is rechecked (and corrected) before any move",
      not n.rechecking and abs(n.odo.y - true_y) < 0.01)
n.step(); n.step()
check("...and only then does the rejoin start", n.mover.busy or n.rejoin_stage is not None
      or d.last != (0.0, 0.0))

n, s, cx, cy = recheck_nav(front=front_read, y=0.99)
s.ready = False
n._enter(M.AT_VICTIM)
n.rescue.started = time.monotonic() - 100.0
n.step()
n.recheck_from = time.monotonic() - 10 * M.RECHECK_WAIT_S
n.step()
check("sonars that never answer: gives up after RECHECK_WAIT_S and carries on",
      not n.rechecking and abs(n.odo.y - 0.99) < 1e-9)

_r = M.RECHECK_AFTER_PICKUP
M.RECHECK_AFTER_PICKUP = False
n, s, cx, cy = recheck_nav(front=front_read, y=0.99)
n._enter(M.AT_VICTIM)
n.rescue.started = time.monotonic() - 100.0
n.step()
M.RECHECK_AFTER_PICKUP = _r
check("RECHECK_AFTER_PICKUP = False skips it", not n.rechecking)

# An off-grid move never drives into a wall the odometry didn't know about.
def rejoin_mover(front):
    sonar = FakeSonar(front=front)
    odo = M.Localiser(M.TICKS_PER_M, M.EFFECTIVE_TRACK_M, M.sonar_mounts())
    odo.reset(0.14, 0.14, M.HEADING_RAD[MZ.NORTH])
    mv = M.Mover(FakeDrive(ticks=(0, 0)), sonar, odo)
    mv.move_to(0.14, 0.40, M.HEADING_RAD[MZ.NORTH], grid=False)     # 26cm to go
    return mv
mv = rejoin_mover(front=0.05)
out = [mv.update() for _ in range(4)]
check("rejoin drive with a wall 5cm ahead and 26cm 'to go': stops instead of driving on",
      "arrived" in out and mv.drive.last == (0.0, 0.0))
mv = rejoin_mover(front=0.60)
out = [mv.update() for _ in range(4)]
check("...but with a clear 60cm ahead it drives on", "arrived" not in out and mv.drive.last[0] > 0)
M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE, M.MOCK_COLLECTION, M.TEST_MODE = _s39_saved


print("40) the HAZARD marker: the dead-end cell in front of it is marked out of bounds")
_s40_saved = (M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE)
M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE = True, False, 0.0


class HazVision(FakeVision):
    """FakeVision plus the hazard marker in view (bearing in deg), or not (None)."""
    def __init__(self, bearing=None):
        super().__init__([])
        self.marker, self.n_marker, self.obstacle, self.n_hazard = None, 0, None, 0
        self.set(bearing)
    def set(self, bearing):
        self.hazard = None if bearing is None else M.Victim(bearing, None, (200, 40, 60, 60), 3600)
        self.n_hazard = 0 if bearing is None else 1


def haz_nav(bearing=0.0, front=0.90, cell=None, theta=None):
    v = HazVision(bearing)
    n = M.Nav(FakeDrive(ticks=(0, 0)), v, FakeLeds(), FakeSonar(front=front))
    if cell is not None:
        cx, cy = n.map.centre(cell)
        n.cell = cell
        n.odo.reset(cx, cy, M.HEADING_RAD[MZ.NORTH] if theta is None else theta)
    return n, v


# From the base cell (0,6) facing north, a wall at y = 1.12 (the north edge of row 3) is 0.98
# from the turning centre -> front reads 0.90 (the sonar is 8cm ahead). The cell in front of
# that wall is (0,3).
n, v = haz_nav(0.0, 0.90)
run(n, M.DETECTION_DEBOUNCE + 1)
check("hazard marker confirmed: the cell in front of its wall, (0, 3), is blocked",
      (0, 3) in n.map.blocked and (0, 3) in n.hazard_cells)
check("...it is not mistaken for a victim (still exploring, yellow on, green off)",
      n.state == M.SEARCH and n.leds.y is True and n.leds.g is not True)
check("...the planners no longer reach it or count it as ground left to explore",
      (0, 3) not in n.map.flood([n.base_cell], True) and (0, 3) not in n.map.frontier_cells())
check("...and the base cell, the robot's own cell and the cells before the marker stay open",
      all(c not in n.map.blocked for c in ((0, 6), (0, 5), (0, 4))))

n, v = haz_nav(0.0, 0.90)
run(n, M.DETECTION_DEBOUNCE - 1)
check("two frames is not enough (debounced)", not n.map.blocked)

n, v = haz_nav(25.0, 0.90)
run(n, 6)
check("marker well off the nose -> the sonar isn't ranging its wall: nothing marked",
      (0, 3) not in n.map.blocked)

n, v = haz_nav(0.0, None)
run(n, 6)
check("no front sonar reading -> can't place it: nothing marked", not n.hazard_cells)

n, v = haz_nav(0.0, 0.90, theta=M.HEADING_RAD[MZ.NORTH] + 0.4)
n.odo.reset(*n.map.centre(M.BASE_CELL), M.HEADING_RAD[MZ.NORTH] + 0.4)
run(n, 6)
check("pointing ~23 deg off the grid -> not placed", not n.hazard_cells)

n, v = haz_nav(0.0, 0.06)
run(n, 6)
check("its cell is the base cell -> refused", M.BASE_CELL not in n.map.blocked and not n.hazard_cells)

n, v = haz_nav(0.0, 0.90)
n.victim_cell = (0, 3)
run(n, 6)
check("its cell has a victim in it -> refused", (0, 3) not in n.map.blocked)

_hz = M.HAZARD_CUE_ENABLED
M.HAZARD_CUE_ENABLED = False
n, v = haz_nav(0.0, 0.90)
run(n, 6)
M.HAZARD_CUE_ENABLED = _hz
check("HAZARD_CUE_ENABLED = False turns it off", not n.map.blocked)

# Standing in the dead end when it is recognised: remembered, blocked only once it has left.
n, v = haz_nav(0.0, 0.06, cell=(0, 3))
run(n, 6)
check("robot is IN the hazard cell: remembered but not blocked yet (it must be able to leave)",
      (0, 3) in n.hazard_cells and (0, 3) not in n.map.blocked)
n.cell = (0, 4)
n._apply_hazards()
check("...blocked as soon as it has left", (0, 3) in n.map.blocked)

# A hazard that would seal the robot in: refused. Robot in the NW corner (0,0), east wall,
# facing south; the only way out is through (0,1).
n, v = haz_nav(0.0, 0.34, cell=(0, 0), theta=M.HEADING_RAD[MZ.SOUTH])
n.map.set_wall((0, 0), M.EAST, True)
n.mover.stop()
run(n, 6)
check("blocking it would cut the robot off from the base -> refused",
      (0, 1) not in n.map.blocked and (0, 1) not in n.hazard_cells)

# Already driving into it: turn back to the cell it left, and never go in.
n, v = haz_nav(None, 0.34)                          # wall at y = 0.56 -> hazard cell (0, 5)
run(n, 3)
check("(setup) driving from the base cell towards (0, 5)",
      n.mover.busy and n.target_cell == (0, 5))
v.set(0.0)
for _ in range(M.DETECTION_DEBOUNCE + 2):
    n.step()
    if (0, 5) in n.map.blocked:
        break
check("hazard recognised on the way in: (0, 5) blocked, and the move is turned back to the base",
      (0, 5) in n.map.blocked and n.target_cell == M.BASE_CELL)
run(n, 40)
check("...and it never heads for the hazard cell afterwards", n.target_cell != (0, 5))

# The vision wrapper: hazard markers come from the template matcher, not colour blobs.
vv.system = _Sys([det("hazard", 100, 60, 4.0, None, ground=False),
                  det("hazard", floor_y, 60, 0.0, 0.3, ground=True),      # a colour blob: ignored
                  det("victim", 100, 60, 0.0, None, ground=False)])
vv.look()
check("VictimVision.hazard: the template-matched hazard marker, with its bearing",
      vv.hazard is not None and abs(vv.hazard.bearing_deg - 4.0) < 1e-9 and vv.n_hazard == 1)
vv.system = _Sys([det("victim", 100, 60, 0.0, None, ground=False)])
vv.look()
check("...a victim marker is not a hazard", vv.hazard is None and vv.n_hazard == 0)
M.STEER_WHILE_DRIVING, M.TRIM_ENABLED, M.STEER_MIN_RATE = _s40_saved

print(f"\n{len(fails)} failed")
sys.exit(1 if fails else 0)
