#!/usr/bin/env python3
"""Navigation main loop -- the final demo mission.

Explores the maze cell by cell, mapping walls with the three sonars (frontier
exploration: always head for the nearest unexplored cell). When vision confirms a
victim it drives at it, stops 10 cm short and collects it, then returns to base by
the shortest route through the map it has built (flood fill), releases it, and goes
back out for the next. Along every corridor it holds itself dead centre between
the walls, and holds heading from the encoders so neither track falls behind.

    yellow LED   exploring / navigating
    green LED    victim detected, approaching and collecting
    red LED      returning to base with a victim

    python3 main.py                      # full run: camera window + map window
    python3 main.py --no-camera          # explore and map only, no vision at all
    python3 main.py --camera-walls       # ALSO use the camera to see walls (off by default)
    python3 main.py --no-centring        # encoder heading hold only, no sonar centring
    python3 main.py --no-avoid           # don't sidestep obstacles the camera sees
    python3 main.py --test-mode          # demo: ONE victim, flash all LEDs 5 s, home, done
    python3 straight_test.py             # does it drive straight? motors + encoders only
    python3 main.py --base-cell 6,6 --start-heading W   # base in another corner
    python3 main.py --calibrate straight # measure TICKS_PER_M
    python3 main.py --calibrate turn     # measure EFFECTIVE_TRACK_M
    python3 main.py --calibrate sonar    # measure the wall readings at a cell centre
    python3 main.py --no-motors          # everything except driving
    python3 main.py --no-display         # headless / over SSH with no X
    python3 main.py --placeholder-vision # use the stand-in detector, NOT Kushal's
    python3 main.py --check              # wire everything up, report, exit
    python3 main.py --blur-threshold 100 # put Kushal's blur guard back on
    python3 main.py --sonar-test         # just the three ranges, live

While it runs you get four views of what it's doing:
  * a camera window with Kushal's detection overlay, the horizon line, the victim
    nav has locked onto, the live range and the sonar readout. q or ESC quits.
  * a map window: walls found, cells visited, the frontier, the planned route and
    the robot's pose. The final map is saved to runs/ when the program exits.
  * a status line rewriting in place: state, cell, victim bearing and range, all
    three sonar readings, which LEDs are lit, and the measured loop rate.
  * event lines above it whenever the state changes or a victim is reached.

Runs with pieces missing on purpose. The motor HAT doesn't enumerate yet and the real
victim detector isn't in the repo, so each subsystem degrades to a named stub and says
so at startup rather than refusing to start. What it will NOT do is invent a distance:
if the camera geometry below isn't measured, APPROACH is disabled rather than driving
at a made-up number.
"""

import argparse
import math
import os
import shutil
import sys
import time

# Kushal's detectors sit at the repo root but import camera_capture_v2_0 / 
# class_profile_v2_0 as top-level modules, which live in vision/. Put both on the
# path so they resolve either way.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VISION_DIR = os.path.join(_REPO_ROOT, "vision")
for _p in (_REPO_ROOT, _VISION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from maze import (EAST, HEADING_BY_NAME, HEADING_NAME, HEADING_RAD, LEFT_OF,  # noqa: E402
                  RIGHT_OF, WEST, Localiser, MazeMap, Odometry, nearest_heading, wrap)


# ===========================================================================
# CONFIG -- the only numbers you should need to touch
# ===========================================================================

# Moved 2026-10-19 from 16/20/21 to 13/19/26, freeing GPIO 20 for the start button
# (button_start.py). Physical pins: green 33, yellow 35, red 37.
GREEN_LED_BCM = 13          # victim detected
YELLOW_LED_BCM = 19         # exploring / navigating
RED_LED_BCM = 26            # returning to base with a victim

# Ultrasonics: (TRIG_BCM, ECHO_BCM). Front is the primary range to the victim;
# left/right are for walls. All reachable through the HAT's passthrough pins.
#
# *** ECHO IS A 5 V OUTPUT AND PI GPIO IS 3.3 V ONLY ***
# Wire each ECHO through a divider (1k in series, 2k to ground) or you will
# eventually damage the pin. TRIG is an input to the sensor and is fine direct.
# (Right was once on (20, 21) until the old LED pins claimed those.)
# The names are WHERE THE SENSOR SITS ON THE ROBOT, looking from behind it. They were once
# crossed -- fixed 2026-10-15 after the sensors were found mixed up: the sensor at the real
# front is on 22/27, the real left on 23/24, the real right on 5/6.
# Full pin map -- keep these disjoint:
#   I2C to the motor HAT   2, 3
#   LEDs                   13 green, 19 yellow, 26 red
#   start button           20 (button_start.py, button to GND)
#   ultrasonics            22/27 front, 23/24 left, 5/6 right
ULTRASONIC_PINS = {
    "front": (22, 27),
    "left":  (23, 24),
    "right": (5, 6),
}
# Only these are pinged. Pinging a sensor that isn't there costs a full echo timeout
# (~22 ms) every time its turn comes round, so take one out here if it's unplugged.
# The maze nav needs all three: front to stop at walls, sides to map and centre.
SONARS_FITTED = ("front", "left", "right")

US_MAX_RANGE_M = 1.0        # ignore anything past this -- a cell is 28 cm. Also caps
                            # how long a missed echo blocks the loop (~6 ms per metre)
US_MIN_TRIGGER_GAP_S = 0.03 # sensors are fired one at a time, and each ping waits out its
                            # own echo, so the gap only has to let reverberation die
                            # (<~20 ms at these ranges). Was 0.06, which at 20 Hz
                            # ticks meant a ping every 100 ms
US_STALE_AFTER_S = 0.5      # a reading older than this is discarded, not reused
US_TRUST_BEARING_DEG = 10.0 # only believe the front sonar is ranging the VICTIM
                            # when the victim is this close to centre -- the cone
                            # is ~15 deg and it reports whatever is nearest in it

# VisionSystem discards any frame whose Laplacian variance falls under this, which
# on the real camera -- 8 ms exposure, high gain, ordinary room light -- threw away
# effectively every frame and made victim detection impossible from main.py.
# 0 disables the guard entirely, which is what the robot actually runs with.
# --blur-threshold overrides it (100.0 is Kushal's original) if motion blur ever
# becomes the bigger problem.
BLUR_THRESHOLD = 0.0

VICTIM_CLASS_NAME = "victim"   # matches the profiles.pkl classes: obstacle/ramp/rubble/victim

# The demo target is the victim OBJECT on the floor. But the arena also carries the
# wall MARKER that depicts a victim, and Kushal's 'victim' colour profile is trained
# on marker images -- so it will happily match both, and nav would otherwise be free
# to drive at whichever happens to look bigger.
#
# They're separated geometrically. The camera is level at 10 cm, so the horizon is
# the image centre row: anything resting on the floor has its base BELOW that row,
# and a marker mounted partway up a wall sits at or above it. Filtering on that both
# discards the marker and guarantees the surviving detections are on the ground
# plane, which is exactly the assumption ground_distance_from_bbox_bottom needs --
# so the camera range estimator becomes valid at the same time.
VICTIM_ON_FLOOR = True
HORIZON_MARGIN_PX = 8       # slack for mounting tolerance and tilt error

STOP_DISTANCE_M = 0.10      # stop this far short of the victim (assessment: 10 cm)
DISTANCE_TOLERANCE_M = 0.02 # close enough -- stops hunting back and forth

# --- the maze -----------------------------------------------------------------
# The grid size, cell size and base corner are released before the demo -- CHECK
# these against the real field. Defaults are the 2026 sim scene: 7 x 7 cells of
# 280 mm on the 2 m table, base in the south-west corner facing north.
# --base-cell and --start-heading override the last two without editing.
MAZE_COLS = 7
MAZE_ROWS = 7
CELL_M = 0.28
BASE_CELL = (0, 6)          # (col, row); row 0 is the NORTH edge
START_HEADING = "N"         # which way the robot faces in the base cell
MISSION_TIME_S = 420.0      # the 7-minute demo -- motors stop when it's up
VICTIMS_TOTAL = 3           # stop once this many are home
TEST_MODE_VICTIMS = 1       # --test-mode: one victim, home, done
# When two open sides are equally good, which to try first. Straight on always
# beats both, because a turn costs time and heading accuracy.
EXPLORE_PREFER_RIGHT = True

# --- odometry: MEASURE with  python3 main.py --calibrate straight / turn -------
# Encoder ticks per metre of track travel. MEASURED 2026-10-15 with the NEW motors: 2.5 s
# straight drive, left 5874 / right 5920 ticks (mean 5897) over 56 cm -> 10530. (The old
# motors gave 11300 on 2026-10-01; the theoretical 18570 -- 1400 counts per output rev over a
# 24 mm sprocket -- was out by 1.6x.) Re-measure with python3 main.py --calibrate straight if
# the tracks, motors or the floor change.
TICKS_PER_M = 10530.0
# Track width the ODOMETRY uses for heading. Wider than the 123 mm sprocket spacing
# because the tracks skid sideways to rotate, and the encoders count the full track
# travel regardless. If 90 degree turns come out short or long, --calibrate turn.
# MEASURED 2026-10-15, new motors: the encoders counted 360 deg and the robot really turned
# 350, so 0.16 * 360 / 350 = 0.1646. (0.3291 from 2026-10-01 was the old, torque-starved
# motors with only one track really turning -- not comparable.)
EFFECTIVE_TRACK_M = 0.1646

# --- sonar geometry: MEASURE with  python3 main.py --calibrate sonar -----------
# What each sonar reads with the robot centred in a cell and a wall right there.
FRONT_WALL_AT_CENTRE_M = 0.060
SIDE_WALL_AT_CENTRE_M = 0.079
# A wall counts as present when the reading is within this of those. Deliberately
# tight rather than half a cell: a victim or rubble standing in the NEXT cell reads
# ~15 cm, and calling that a wall would seal off the very cell the victim is in.
WALL_PRESENT_WINDOW_M = 0.08
# ...except towards a cell a victim has been placed in, where anything past this is
# taken to be the victim rather than a wall.
VICTIM_WALL_WINDOW_M = 0.05
SENSE_TIMEOUT_S = 0.6       # max wait at a cell centre for fresh L/F/R readings

# --- driving one cell ---------------------------------------------------------
SEARCH_SPEED = 0.13         # m/s along a corridor
ARRIVE_SLOW_M = 0.06        # slow down this far from the cell centre...
ARRIVE_SPEED = 0.08         # ...to this, so it stops where it means to
ARRIVE_TOLERANCE_M = 0.01
REVERSE_SPEED = 0.10        # backing out of a cell that turned out to be blocked
DRIVE_TIMEOUT_S = 6.0       # no arrival in this long (stuck on a bump) = blocked
# Odometry sanity check: still driving after this long, the encoders should have
# counted at least this share of the distance the commanded speed implies. Falling
# short means the encoders aren't counting or TICKS_PER_M is wrong -- and then every
# cell "never arrives", which looks like blocked cells and an empty map.
ODO_CHECK_S = 2.0
ODO_MIN_SHARE = 0.4
# Something this much closer than the far side of the cell ahead is in the way (a
# victim, rubble, a wall the side sonar missed). Seen on this many fresh pings.
BLOCK_MARGIN_M = 0.05
BLOCK_CONFIRM = 3

# STEERING WHILE DRIVING. On the OLD motors it did nothing (straight_test.py step 5, 2026-10-01:
# the right track commanded 70 raw above the left, both tracks still ran at ~0.10 m/s and the
# robot turned ~4 deg/s -- the chassis locked them together), so it was off and the heading was
# set with pivot pulses at the start of each leg (TRIM below). The NEW motors steer (46% of a
# commanded turn, table below), so it's back ON; the leg-start pivot trim stays. Set False for
# pivot pulses only.
STEER_WHILE_DRIVING = True
# The NEW motors (straight_test.py step 5, 2026-10-15) DO steer by a speed difference, but with
# a dead band: under ~25 raw of difference the yaw is ~2 deg/s, then 6 deg/s at 30, 9 at 40, 15
# at 50, 30 at 70. So a small heading error asks for a correction the motors don't deliver.
# When steering while driving, any error worth correcting (over STEER_TOL_RAD) is given at least
# STEER_MIN_RATE, which clears the dead band, and smaller ones are left alone.
STEER_MIN_RATE = 0.5        # rad/s; 0 = off (plain proportional steering). ~31 raw of
                            # difference: past the dead band. Sim, measured response: wall
                            # clearance 13 cm vs 11 (pulses only) on a clean run, 11 vs 7-9 with slip.
STEER_TOL_RAD = 0.035       # ~2 deg
TRIM_ENABLED = True         # False: no pivot trim at the start of a leg
TRIM_TOL_RAD = 0.035        # ~2 deg: close enough to start the leg
TRIM_PULSE_MIN_S = 0.02     # shortest / longest single pivot pulse
TRIM_PULSE_MAX_S = 0.30     # a weak pivot needs a long pulse to move at all
TRIM_SETTLE_S = 0.06        # stand still this long after a pulse so the encoders catch up
TRIM_MAX_S = 3.0            # never spend longer than this squaring the heading up
TRIM_MAX_PULSES = 14
# A turn that stops well short is NOT finished: driving off then points the leg at a wall (a
# 90 deg corner left 58 deg undone, it drove into the wall, marked it blocked and lost the
# route home). Keep turning instead, up to this many more rounds.
TURN_RETRY_ERR_RAD = 0.35   # ~20 deg
TURN_MAX_RETRIES = 8
# Lean the leg towards the corridor centre: aim this far off the cardinal heading,
# atan(offset / lead), so it arrives on the centreline. Next leg squares it up again.
CENTRE_AIM_LEAD_M = 0.28
CENTRE_AIM_MIN_M = 0.02     # offsets smaller than this aren't worth a pulse (noise)
CENTRE_AIM_MAX_RAD = 0.15   # ~8.6 deg: never lean further than this

# Stopped short of a cell centre: if, when it stops to sense a cell, the front sonar sees a
# wall nearer than "not there yet" but not at the cell's own edge, it can't be where the
# odometry says. A wall ahead of a robot standing at a cell centre is half a cell away
# (and further walls whole cells beyond that), so a wall at half a cell PLUS a bit means
# it stopped that bit short -- wheel slip, a scale error in TICKS_PER_M, which adds up
# along a long corridor where no wall is in sonar range to correct it. Creep up to the
# wall, then call that the centre.
CREEP_ENABLED = True
CREEP_MIN_M = 0.035         # shorter than this is sonar noise, not worth moving for
CREEP_MAX_M = 0.17          # longer, and it can't be this cell's wall: leave it to the map...
CREEP_PATH_SHARE = 0.16     # ...unless it has driven far since a wall last pinned it down: an
                            # odometry that's out by up to this share of the distance driven
CREEP_ABS_MAX_M = 0.26      # can reach this far, no further: a wall a whole cell on from the
                            # centre reads 0.28, and "stopped 28 cm short" can't be told from it
CREEP_VICTIM_HOLD_S = 10.0  # a victim seen this recently might be what the sonar is reading
# Every time a wall pins down where it really is, the gap to where the odometry said is
# also a measurement of the odometry's distance scale (the error over the distance driven
# since the last such wall). The scale is nudged by this much of that each time, within
# limits, so a TICKS_PER_M that's out by 10% stops mattering after the first corridor.
SCALE_LEARN_RATE = 0.7
SCALE_MIN_PATH_M = 0.25     # a shorter run than this says nothing about the scale
SCALE_LIMITS = (0.6, 1.4)

# Heading hold: the encoders say which way it's pointing, so holding heading is
# the same thing as not letting either track fall behind the other.
HEADING_HOLD_GAIN = 4.0     # rad/s per rad of heading error
MAX_STEER_RATE = 1.0        # rad/s cap while driving
# Corridor centring: aim this many rad of heading back towards the middle per metre
# off centre, capped at MAX_CENTRE_ANGLE. 7.0 means 1 cm off -> 4 deg correction.
# The offset decays at about SEARCH_SPEED * CENTRE_GAIN per second (0.9/s here), so
# 4.0 took ~2 s -- most of a cell -- to pull in.
# TUNE: weaves -> lower CENTRE_GAIN; drifts into walls -> raise it.
CENTRE_GAIN = 7.0
MAX_CENTRE_ANGLE = 0.30     # rad, ~17 deg
# Offset from the side sonars directly, not via the filter: a filter that has settled
# believes itself and answers a new reading slowly. A reading older than this is
# ignored (the filter's estimate is used instead).
SIDE_FRESH_S = 0.35

# --- the camera as a second wall sensor (wallvision.py) ------------------------
# The camera sees both walls and the way ahead in one frame, 20 times a second, and
# catches what the sonars miss (a soft or glancing wall that gives no echo). The
# sonars cover what it can't (the last ~27 cm). They are used together:
#   * centring: sonar and camera offsets are blended while they agree, and either
#     one alone is used when the other has nothing;
#   * mapping: a wall seen by EITHER marks the edge (a missed wall costs a collision,
#     a phantom one only a detour), except towards a cell a victim is in.
# OFF until it's been checked against the real maze: on the robot it contradicted the
# sonar (phantom "wall ahead") and its yaw reading led the squaring-up astray. Turn it on
# with --camera-walls to try it; straight_test.py first if driving isn't straight.
USE_CAMERA_WALLS = False
# Steer back to the corridor centre from the side sonars. --no-centring turns it off,
# leaving encoder heading hold alone, to find out whether a veer is the motors or this.
CENTRING = True

# --- going round an obstacle that sits to one side of the corridor ------------------
# The camera sees a floor OBSTACLE ahead, off to one side. Rather than drive at it until
# the front sonar says "blocked" and give up the cell, steer to the OPPOSITE side of the
# corridor and slide past in the gap. Measured on the real field: the obstacle is ~12 cm
# wide and the robot has ~2 cm either side of it when alongside -- so the obstacle stands
# against one wall, the gap is CELL_M - OBSTACLE_WIDTH_M = 16 cm, and the robot (~12 cm)
# has to run down the MIDDLE OF THE GAP, which is OBSTACLE_WIDTH_M / 2 off the corridor
# centreline. An obstacle in the middle of the corridor (nearer than OBSTACLE_MIN_SIDE_M
# to the centreline) leaves a gap too small for the robot, so that is still treated as
# blocked and detoured round, as before.
# Alongside the obstacle both side sonars read it and the far wall at ~2 cm each, so the
# normal two-wall centring keeps it in the middle of the gap with no help from here.
OBSTACLE_AVOID = True
OBSTACLE_CLASS_NAME = "obstacle"   # matches the profiles.pkl class
OBSTACLE_WIDTH_M = 0.12
OBSTACLE_PASS_OFFSET_M = OBSTACLE_WIDTH_M / 2.0   # how far off the centreline to run
OBSTACLE_MIN_SIDE_M = 0.03     # obstacle centre must be at least this off the centreline
OBSTACLE_SEE_MAX_M = 0.50      # ignore obstacles the camera ranges further than this
OBSTACLE_ASSUMED_RANGE_M = 0.25  # used when the camera can't range it (no geometry)
OBSTACLE_CLEAR_M = 0.12        # keep sidestepping this far past it (the camera loses sight
                               # of it once it is under the lens, well before we're past)
OBSTACLE_ALONGSIDE_M = 0.05    # an obstacle-side sonar reading under this = we're beside it
OBSTACLE_FRONT_STOP_M = 0.07   # while sidestepping, only call it blocked this close
CAMERA_WALL_WEIGHT = 0.7    # camera's share of the centring offset when both agree
                            # (it updates ~20x a second, the sonars ~7x)
CAMERA_AGREE_M = 0.03       # ...agree = within this; further apart, trust the sonars
CAMERA_FRESH_S = 0.30       # a wall view older than this is ignored
CAMERA_WALL_TOLERANCE_M = 0.04  # camera wall is "this cell's" within half a cell + this
# The camera can ADD a wall the sonar missed (no echo, or a reading only just outside
# the wall window), but never over a sonar that sees clear space well beyond it: a
# front sonar reading 67 cm cannot have a wall 14 cm ahead. Past this slack beyond
# the wall window, the sonar's "open" stands and the disagreement is logged.
CAMERA_SONAR_SLACK_M = 0.06
CAMERA_DISAGREE_TICKS = 15  # camera says "wall right ahead" against a clear front
                            # sonar for this many frames in a row -> warn once
# The walls are on the maze grid, so the angle the camera measures against them IS the
# heading error -- measured, where the encoders only integrate it and drift. While
# driving, the heading estimate is pulled this far towards it per fresh frame:
CAMERA_YAW_NUDGE = 0.35
CAMERA_YAW_MAX_RAD = 0.20   # ignore a "yaw" bigger than ~11 deg: after a turn the encoders
                            # are never that far out, so that's a bad reading
# After every turn on the spot, square up to the walls before driving off: short
# pulses, each followed by a fresh frame, until it's parallel. The pivot floor is
# ~6 deg per control tick, far too coarse to creep up on the walls continuously.
ALIGN_AFTER_TURN = True
ALIGN_TOL_RAD = 0.02        # ~1.1 deg: square enough
ALIGN_SETTLE_S = 0.12       # let it stop and the camera catch up before measuring
ALIGN_WAIT_S = 0.35         # no wall angle to be had by then -> drive on without it
ALIGN_MAX_S = 1.5           # never spend longer than this squaring up
ALIGN_MAX_PULSE_S = 0.06    # first pulse length (~7 deg at the pivot floor if it bites)
# A pulse this short can fall inside the motors' dead band and turn nothing; the next
# one is made longer, up to ALIGN_PULSE_CAP_S. And if the yaw gets WORSE (or nothing
# helps), squaring up is abandoned rather than chasing a reading that isn't tracking the
# robot -- driving off as it was beats driving off further out.
ALIGN_PULSE_CAP_S = 0.30
ALIGN_PULSE_GROWTH = 1.6
ALIGN_NO_EFFECT_RAD = 0.008 # a pulse that changed the yaw by less than ~0.5 deg did nothing
ALIGN_DIVERGE_RAD = 0.012   # ...and one that made it ~0.7 deg worse went the wrong way
                            # (small, because the camera under-reads a growing angle)
ALIGN_GIVE_UP = 2           # this many bad pulses in a row (diverging) -> give up
ALIGN_NO_EFFECT_GIVE_UP = 5 # ...or this many that did nothing, even lengthened
CAMERA_SUSPECT_S = 10.0     # camera contradicts the sonar -> ignore it this long

# --- localisation: odometry corrected by the sonars against the map ------------
# Each sonar reading is compared with the range the map predicts from where the
# robot thinks it is, and the difference corrects position AND heading. These say
# how much to trust a reading and when to throw one out.
SONAR_SIGMA_M = 0.008       # HC-SR04 noise at these ranges
GATE_KNOWN_M = 0.05         # reading this far off a mapped wall = something else
GATE_MAYBE_M = 0.025        # ...tighter for an edge whose wall hasn't been seen yet
# Where the side sonars sit, forward of the turning centre. MEASURE. (Their sideways
# offset and the front sonar's forward offset come from the *_AT_CENTRE_M readings.)
SIDE_SONAR_FORWARD_M = 0.0

# --- turning on the spot ------------------------------------------------------
SEARCH_TURN_RATE = 2.4      # rad/s, the fastest it spins
TURN_GAIN = 3.0             # rad/s per rad left to turn, floored at MIN_TURN_RATE_PIVOT
TURN_TOLERANCE_RAD = 0.06   # ~3.4 deg: close enough, the drive's heading hold finishes it
TURN_TIMEOUT_S = 10.0       # a weak pivot is slow (a quarter turn took >4 s); keep turning
# Pivots in the maze (turning at corners, setting the heading) run at FULL power: the tracks
# scrub on carpet and the motors are short of torque, so a gentler pivot just stalls. 2.8
# rad/s is above the 2.65 that already saturates the board, i.e. raw 127. (The victim
# approach keeps its own, gentler rates above.)
GRID_PIVOT_RATE = 2.8

# --- rescue placeholders -- Roger's mechanism replaces these ------------------
SEEK_ATTEMPTS = 2           # tries at a victim cell whose sonar check fails
# STAND-IN for the collection mechanism (Roger's): on reaching a victim the robot stands
# still for RESCUE_TIME_S with ALL THREE LEDs flashing together, then it has the victim
# "aboard" and drives home with the red LED on. When the real mechanism arrives, replace
# RescueStub's start()/done() and the flashing in Nav._at_victim with its real calls.
RESCUE_TIME_S = 5.0
COLLECT_FLASH_HZ = 3.0      # all-LED flash while "collecting"
# MOCK COLLECTION (while there's no real mechanism): at a victim it stands still for
# RESCUE_TIME_S with all three LEDs flashing, then drives home with the victim "aboard".
# Leave True until Roger's mechanism is wired in; then set it False and put the real calls in
# RescueStub / Nav._at_victim. --test-mode forces the mock on whatever this says, so a demo can
# still run find-victim -> flash all LEDs -> return home once the real thing is the default.
MOCK_COLLECTION = True
TEST_MODE = False
RELEASE_TIME_S = 2.0        # "releasing" at base, red LED

# --- approach: hold the victim within HEADING_TOLERANCE_DEG -------------------
HEADING_TOLERANCE_DEG = 1.0   # aim to keep the victim inside this
# Beyond this it pivots; inside it, it corrects while driving. Set wide deliberately:
# correcting while rolling is both easier (no static friction to break) and smoother.
# A pivot is effectively bang-bang once the friction floor kicks in -- at 2.2 rad/s
# and 20 Hz it swings ~6 deg per tick -- so it needs plenty of margin to settle into,
# or it would hunt either side of the threshold instead of converging.
HEADING_COARSE_DEG = 20.0

# Heading lock. Once the victim is inside HEADING_LOCK_DEG the heading is committed
# and steering stops -- from there it drives straight. Bearings get noisy close in
# (the victim fills the frame, the bbox centre wanders), and chasing that noise
# steers the robot off a line that was already good enough.
#
# It only unlocks if the bearing drifts past HEADING_UNLOCK_DEG, which is well
# outside the noise: locking and unlocking on the same threshold would chatter.
HEADING_LOCK_DEG = 5.0
HEADING_UNLOCK_DEG = 15.0
# Two different floors, because pivoting and correcting are different physics.
# Stationary, a tracked chassis has to SKID both tracks sideways to rotate, and
# static friction across the whole contact patch is what was stalling turns under
# ~18 degrees -- the command was real, it just wasn't enough force. Already rolling,
# the tracks are moving and a much gentler differential steers fine.
MIN_TURN_RATE_PIVOT = 2.2     # rad/s floor when turning on the spot
MIN_TURN_RATE_MOVING = 0.3    # rad/s floor while driving forward
APPROACH_SPEED = 0.16       # m/s, closing speed once a victim is being tracked
CREEP_SPEED = 0.09          # m/s, inside CREEP_RANGE_M -- slow enough to stop cleanly
CREEP_RANGE_M = 0.25
HEADING_GAIN = 0.07         # rad/s per degree of bearing error
MAX_TURN_RATE = 2.5         # rad/s

DETECTION_DEBOUNCE = 3      # consecutive frames before believing a detection
LOST_GRACE_FRAMES = 8       # frames a victim may vanish for before we call it lost

# --- recheck the position after collecting a victim ---------------------------
# Chasing the victim takes the robot off the grid: it steers by eye, its heading wanders off
# the cardinal, and the cell it ends up in is usually one it has never mapped, so the
# localiser (which corrects only against walls already on the map) has nothing to hold on
# to. Left alone, the rejoin then drives "to the cell centre" on an odometry that may be
# several cm out -- straight into the wall the victim was beside -- and spins up against it.
# So, before moving at all, it stands still, takes a fresh set of sonar readings and
# re-anchors the odometry to the walls:
#   * ahead: a wall is always on a cell edge, so the reading fixes the position along the
#     way it faces (the nearest edge that fits, within RECHECK_FRONT_SNAP_M -- anything
#     else, e.g. the mechanism or the victim in front of the sonar, is ignored);
#   * sideways: the side walls give the offset from the corridor centre, as in driving.
# Readings only mean something when the robot points along the grid, so it needs to be
# within RECHECK_MAX_SKEW_DEG of a cardinal heading.
RECHECK_AFTER_PICKUP = True
RECHECK_WAIT_S = 0.8            # max wait for a fresh set of pings
RECHECK_MAX_SKEW_DEG = 12.0
RECHECK_FRONT_MAX_M = 0.45      # a front reading longer than this isn't used
RECHECK_FRONT_SNAP_M = 0.06     # ...and must match a cell edge to within this
RECHECK_SIDE_MAX_M = 0.07       # side-wall correction bigger than this is not believed
# A move that is not on the grid (the rejoin) has no wall-ahead check of its own, so it
# stops when the front sonar reads this close: it is at least as near the wall as a robot
# standing at a cell centre (FRONT_WALL_AT_CENTRE_M), and no centre is worth driving into one.
REJOIN_WALL_STOP_M = 0.07

# --- the victim's WALL MARKER as a cue -----------------------------------------
# The exposed-victim marker hangs on the wall (centre ~120 mm up), so it is NOT on the
# floor and the camera cannot range it by ground geometry -- driving at it as if it were
# the token would drive into the wall. But it does say a victim is near. So when the
# marker is confirmed and no victim object is in view, nav places a victim in the cell IN
# FRONT of the marker (wall range from the front sonar, minus half a cell and a small
# back-off, so it lands BEFORE the wall and never on it), lights the green LED and goes
# there to look (SEEK). The usual sonar check then decides whether there is really
# something to close in on. A victim OBJECT on the floor always wins: if one is in view,
# the victim is placed where the object is, not where the marker says.
MARKER_CUE_ENABLED = True
MARKER_CLASS_NAMES = (VICTIM_CLASS_NAME,)   # marker classes that mean "a victim is here":
                                            # the template folder under vision/templates/
MARKER_BACK_OFF_M = 0.05                    # extra nudge from the wall towards the robot, so a
                                            # sonar reading that comes out long still lands in
                                            # the cell BEFORE the wall, not in or past it
MARKER_CELL_INSET_M = CELL_M / 2.0 + MARKER_BACK_OFF_M   # wall -> just inside the cell in front

# --- the HAZARD marker: a dead end -----------------------------------------------
# The hazard marker hangs on the end wall of a dead end, and the dead-end cell is the one
# directly in front of it (assessment 5.2: continued exploration toward it "is unlikely to
# assist"). When it is confirmed the cell in front of its wall is marked BLOCKED -- the
# planners then never enter it, and it stops counting as ground left to explore -- found the
# same way as the victim marker's cell (wall range from the front sonar, minus half a cell and
# a small back-off). Never the base cell, a cell a victim is in, or the cell the robot is
# standing in (that one is blocked as soon as it has left it).
# The class name is the templates/<name>/ folder the hazard marker was captured into:
#     python3 capture_marker_template.py hazard
HAZARD_CUE_ENABLED = True
HAZARD_CLASS_NAMES = ("hazard",)
HAZARD_BACK_OFF_M = 0.05
HAZARD_CELL_INSET_M = CELL_M / 2.0 + HAZARD_BACK_OFF_M
HAZARD_HEADING_TOL_DEG = 10.0   # only place it while pointing along the grid

# --- the blind run-in ---------------------------------------------------------
# Close up, the victim drops out of the bottom of the frame: at CAMERA_HEIGHT_M with
# VERTICAL_FOV_DEG, an object on the floor stops being visible somewhere under
# ~camera_blind_range_m(). Losing sight of it there is expected, NOT a lost victim --
# so once we're lined up and this close, the approach commits and finishes on the
# front sonar alone.
CLOSING_TRIGGER_M = 0.30    # start trusting sonar alone below this
CLOSING_ALIGN_DEG = 12.0    # ...but only if the victim was this well centred
CLOSING_TIMEOUT_S = 5.0     # give up and go back to searching if it never arrives
CLOSING_JUMP_M = 0.05       # a front reading this much further than expected = lost it
CONTROL_HZ = 20

# --- camera ranging: measured on the robot 2026-09-09 -------------------------
# The front sonar is still the primary estimator; these make the camera a usable
# fallback for when the victim is off to one side, outside the sonar's cone.
CAMERA_HEIGHT_M = 0.10      # lens centre above the floor
CAMERA_TILT_DEG = 0.0       # mounted level
CAMERA_FORWARD_M = 0.06     # lens ahead of the turning centre -- MEASURE. Used to
                            # place a seen victim in the right cell of the map.
VERTICAL_FOV_DEG = 41.0     # Camera Module 3 standard lens

# Bearing FOV. NOTE: vision/camera_capture_v2_0.py defaults to 45.0, which does not
# match a 41 deg vertical -- the Camera Module 3 standard lens is 66 x 41, and 45 x 41
# is not a real 4:3 lens. A too-small FOV understates every bearing (a victim 15 deg
# off-axis reads as ~10), which both slows the steering correction and lets the
# US_TRUST_BEARING_DEG gate believe the sonar when the victim is further off-axis than
# it thinks. Nav passes this explicitly rather than taking that default; Kushal's file
# still needs fixing at source.
HORIZONTAL_FOV_DEG = 66.0

# Placeholder detector only (--placeholder-vision). NOT Kushal's calibrated bands.
# Taken from the victim entry in profiles.pkl (hue 78-102, green-cyan) so the
# stand-in at least looks for the right colour. Still not a vision system.
PLACEHOLDER_HSV_LOW = (78, 183, 135)
PLACEHOLDER_HSV_HIGH = (102, 255, 255)
PLACEHOLDER_MIN_AREA_PX = 400


# ===========================================================================
# Terminal status + live camera window
# ===========================================================================
WINDOW_NAME = "EGB320 nav"
MAP_WINDOW_NAME = "EGB320 map"

# What each state is actually doing, in words, for the terminal and the HUD.
STATE_LABEL = {
    "SEARCH":    "EXPLORING",
    "SEEK":      "GOING TO VICTIM",
    "APPROACH":  "APPROACHING VICTIM",
    "CLOSING":   "CLOSING IN (camera blind)",
    "AT_VICTIM": "COLLECTING VICTIM",
    "RETURN":    "RETURNING TO BASE",
    "AT_BASE":   "RELEASING AT BASE",
    "DONE":      "FINISHED",
}


class StatusLine:
    """One self-rewriting line of live telemetry, plus event lines above it.

    Events (state changes, warnings) have to clear the status line before printing or
    they land on top of it and leave fragments behind, so everything routes through
    here rather than calling print() directly.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._width = 0

    def event(self, message):
        if self.enabled and self._width:
            print("\r" + " " * self._width + "\r", end="")
            self._width = 0
        print(message, flush=True)

    def update(self, text):
        if not self.enabled:
            return
        try:
            text = text[:max(20, shutil.get_terminal_size().columns - 1)]   # never wrap
        except OSError:
            pass
        pad = max(0, self._width - len(text))
        print("\r" + text + " " * pad, end="", flush=True)
        self._width = len(text)

    def close(self):
        if self.enabled and self._width:
            print()
            self._width = 0


STATUS = StatusLine()


def status_text(nav, sonar, leds, hz):
    """The live line: what state we're in, what we can see, what the sonar reads."""
    state = STATE_LABEL.get(nav.state, nav.state)
    if nav.state == "DONE" and getattr(nav, "finish_reason", None):
        return f"FINISHED: {nav.finish_reason}"

    v = nav.victim
    if v is None:
        target = "victim: none"
    else:
        distance, source = nav.victim_range()
        rng = "range unknown" if distance is None else f"{distance*100:5.1f}cm ({source})"
        lock = " LOCK" if getattr(nav, "heading_locked", False) else "     "
        target = f"victim: {v.bearing_deg:+6.1f}deg{lock} {rng}"

    def cm(name):
        d = sonar.get(name)
        return "  ---" if d is None else f"{d*100:5.1f}"
    sonar_txt = f"sonar L{cm('left')} F{cm('front')} R{cm('right')} cm"
    if getattr(nav, "wallcam", None) is not None:
        view = nav._camera_view()
        if view is None:
            sonar_txt += " cam ---"
        else:
            def cv(x):
                return " --" if x is None else f"{x*100:3.0f}"
            front = "CLS" if view.front_close else cv(view.front_m)
            yaw = "" if view.yaw_rad is None else f" Y{math.degrees(view.yaw_rad):+.0f}"
            sonar_txt += f" cam L{cv(view.left_m)} F{front} R{cv(view.right_m)}{yaw}"
        ahead, src = nav.front_distance()
        sonar_txt += (" ahead ---" if ahead is None
                      else f" ahead {ahead*100:.0f}cm({src})")

    lit = "".join(c if getattr(leds, "_state", {}).get(n) else "-"
                  for n, c in (("green", "G"), ("yellow", "Y"), ("red", "R")))
    cell = getattr(nav, "cell", None)
    where = "" if cell is None else f"{cell}{HEADING_NAME[nav.facing]} "
    # Most useful first: StatusLine cuts to the terminal width, so what's last goes first.
    return (f"{state:<19}| {where}{target:<38}| {sonar_txt} | {lit} | {hz:4.1f}Hz | "
            f"{vision_summary(getattr(nav, 'vision', None))}")


def vision_summary(vision):
    """What vision did with the last frame -- the thing that was invisible before.

    A frame rejected by the blur guard produced no boxes and no message, which looks
    exactly like "nothing is there". It now says so, with the number it was judged on.
    """
    if vision is None:
        return "vis: --"
    if getattr(vision, "blur_rejected", False):
        blur = getattr(vision, "last_blur", None)
        thr = getattr(vision, "blur_threshold", None)
        return (f"vis: BLUR-REJECT {blur:.0f}<{thr:.0f}" if blur is not None and thr
                else "vis: BLUR-REJECT")
    counts = getattr(vision, "class_counts", {}) or {}
    blobs = ("none" if not counts else
             " ".join(f"{k}x{v}" for k, v in sorted(counts.items())))
    raw = getattr(vision, "n_victim_raw", 0)
    kept = getattr(vision, "n_victim_kept", 0)
    blur = getattr(vision, "last_blur", None)
    blur_txt = "" if blur is None else f" blur{blur:.0f}"
    n_marker = getattr(vision, "n_marker", 0)
    marker_txt = f" marker {n_marker}" if n_marker else ""
    n_hazard = getattr(vision, "n_hazard", 0)
    hazard_txt = f" HAZARD {n_hazard}" if n_hazard else ""
    return f"vis:{blur_txt} [{blobs}] victim {kept}/{raw}{marker_txt}{hazard_txt}"


class Display:
    """Live camera window with the vision overlays plus a nav HUD, and the map window
    beside it showing what nav has explored.

    Off automatically when there's no GUI to draw into -- opencv-python-headless has
    no imshow at all, and an SSH session without X forwarding has nowhere to put a
    window. Either case disables the window and says so, rather than killing the run.
    """

    def __init__(self, enabled=True, camera=True, show_map=True):
        self.ok = False
        self.cv2 = None
        self.camera = camera
        self.show_map = show_map
        if not enabled:
            return
        try:
            import cv2
            if not hasattr(cv2, "imshow"):
                raise RuntimeError("this OpenCV build has no GUI (opencv-python-headless)")
            if camera:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 800, 600)
            if show_map:
                cv2.namedWindow(MAP_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self.cv2 = cv2
            self.ok = True
            STATUS.event("[display] windows open -- q or ESC to quit")
        except Exception as exc:                       # noqa: BLE001
            STATUS.event(f"[display] no windows -- {type(exc).__name__}: {exc}\n"
                         "          (headless OpenCV, or no X display over SSH). "
                         "Everything else still runs; --no-display silences this.")

    def show(self, frame, detections, nav, sonar, draw_detections=None):
        """Draw and display one tick. Returns False if the user asked to quit."""
        if not self.ok:
            return True
        if self.camera and frame is not None:
            self.cv2.imshow(WINDOW_NAME, render_hud(frame, detections, nav, sonar,
                                                    draw_detections))
        if self.show_map:
            from map_view import render_map
            self.cv2.imshow(MAP_WINDOW_NAME, render_map(nav))
        return self.cv2.waitKey(1) & 0xFF not in (ord("q"), 27)

    def close(self):
        if self.ok:
            try:
                self.cv2.destroyAllWindows()
            except Exception:                          # noqa: BLE001
                pass


def render_hud(frame, detections, nav, sonar, draw_detections=None):
    """Overlay the vision output and nav's own state onto a copy of the frame.

    Separate from Display so it can be rendered and checked without a GUI.

    Draw order matters: the state banner goes down first, then Kushal's detection
    overlay on top of it, so his own readout isn't hidden behind our bar.
    """
    import cv2
    img = frame.copy()
    h, w = img.shape[:2]

    # --- state banner ---
    state = STATE_LABEL.get(nav.state, nav.state)
    colour = {"SEARCH": (200, 200, 200), "APPROACH": (0, 220, 255),
              "CLOSING": (0, 170, 255), "AT_VICTIM": (0, 255, 0), "SEEK": (0, 255, 0),
              "RETURN": (0, 0, 255), "AT_BASE": (0, 0, 255),
              "DONE": (200, 200, 200)}.get(nav.state, (255, 255, 255))
    cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(img, state, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    distance, source = nav.victim_range() if nav.victim is not None else (None, None)
    if distance is not None:
        txt = f"{distance*100:.1f} cm ({source})   stop at {STOP_DISTANCE_M*100:.0f}"
        cv2.putText(img, txt, (w - 330, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if distance > STOP_DISTANCE_M else (0, 0, 255), 2)

    # --- Kushal's overlay: boxes, class, confidence, angle, distance ---
    if draw_detections is not None and detections:
        try:
            draw_detections(img, detections)
        except Exception:                              # noqa: BLE001
            pass

    # --- horizon: below it is floor. This line is what separates the victim object
    # from its wall marker, so being able to see it is the point.
    hy = int(horizon_row(h))
    cv2.line(img, (0, hy), (w, hy), (80, 80, 255), 1)
    cv2.putText(img, "horizon", (6, max(12, hy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 255), 1)

    # --- the victim nav has actually locked onto, distinct from everything else ---
    if nav.victim is not None:
        x, y, bw, bh = nav.victim.bbox
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 220, 255), 3)
        cx = x + bw // 2
        cv2.line(img, (cx, y), (cx, y + bh), (0, 220, 255), 1)
        cv2.circle(img, (w // 2, h - 30), 4, (255, 255, 255), -1)
        cv2.line(img, (w // 2, h - 30), (cx, y + bh), (0, 220, 255), 1)

    # --- what vision found, second row ---
    vision = getattr(nav, "vision", None)
    if vision is not None:
        counts = getattr(vision, "class_counts", {}) or {}
        txt = ("blobs: none" if not counts else
               "blobs: " + "  ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
        txt += f"   victim kept {getattr(vision,'n_victim_kept',0)}" \
               f"/{getattr(vision,'n_victim_raw',0)}"
        marker = getattr(vision, "marker", None)
        if marker is not None:
            mx, my, mw, mh = marker.bbox
            cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 140, 255), 2)
            cv2.putText(img, "victim marker", (mx, max(12, my - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
            txt += f"   MARKER x{getattr(vision, 'n_marker', 1)}"
        hazard = getattr(vision, "hazard", None)
        if hazard is not None:
            hx, hy, hw, hh = hazard.bbox
            cv2.rectangle(img, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 2)
            cv2.putText(img, "hazard marker", (hx, max(12, hy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            txt += f"   HAZARD x{getattr(vision, 'n_hazard', 1)}"
        if getattr(nav, "heading_locked", False):
            txt += "   HEADING LOCKED"
        cv2.putText(img, txt, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)

    # --- sonar readout along the bottom ---
    cv2.rectangle(img, (0, h - 26), (w, h), (0, 0, 0), -1)
    parts = []
    for name in ("left", "front", "right"):
        d = sonar.get(name)
        parts.append(f"{name[0].upper()} {'---' if d is None else f'{d*100:.0f}cm'}")
    cv2.putText(img, "sonar  " + "   ".join(parts), (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    return img


# ===========================================================================
# LEDs
# ===========================================================================
class Leds:
    """The assessment's colours: yellow = exploring or navigating, green = victim
    detected and collection in progress, red = returning to base with a victim.

    Falls back to console printing when gpiozero isn't available or the pins can't be
    claimed, so the nav logic stays testable on a laptop.
    """

    def __init__(self, green_pin=GREEN_LED_BCM, yellow_pin=YELLOW_LED_BCM,
                 red_pin=RED_LED_BCM):
        self.real = False
        try:
            from gpiozero import LED
            self._green = LED(green_pin)
            self._yellow = LED(yellow_pin)
            self._red = LED(red_pin)
            self.real = True
            print(f"[leds] GPIO {green_pin} green, {yellow_pin} yellow, {red_pin} red")
        except Exception as exc:                       # noqa: BLE001
            print(f"[leds] CONSOLE ONLY -- {type(exc).__name__}: {exc}")
        self._state = {"green": None, "yellow": None, "red": None}

    def _set(self, name, obj, on):
        if self._state[name] == on:
            return                                     # only log real transitions
        self._state[name] = on
        if self.real:
            obj.on() if on else obj.off()
        else:
            print(f"[led] {name} {'ON' if on else 'off'}")

    def green(self, on):
        self._set("green", getattr(self, "_green", None), on)

    def yellow(self, on):
        self._set("yellow", getattr(self, "_yellow", None), on)

    def red(self, on):
        self._set("red", getattr(self, "_red", None), on)

    def all_off(self):
        try:
            self.green(False)
            self.yellow(False)
            self.red(False)
        except Exception:                              # noqa: BLE001
            pass


# ===========================================================================
# Ultrasonics
# ===========================================================================
class Ultrasonic:
    """One HC-SR04 / SRF05 on a trigger + echo pair.

    Timed in Python, so expect a centimetre or two of jitter -- fine against a 10 cm
    stop threshold, and no worse than gpiozero's own DistanceSensor, which does the
    same thing. Pinging is explicit rather than free-running precisely so the three
    sensors can be sequenced and never hear each other's echoes.
    """

    SPEED_OF_SOUND = 343.0     # m/s at ~20 C

    def __init__(self, trig_bcm, echo_bcm, max_range_m=US_MAX_RANGE_M):
        from gpiozero import DigitalInputDevice, DigitalOutputDevice
        self.trig = DigitalOutputDevice(trig_bcm)
        self.echo = DigitalInputDevice(echo_bcm)
        self.max_range_m = max_range_m
        # Longest an echo can legitimately take, plus margin. Past this there was
        # no echo -- nothing in range -- which is a normal answer, not a fault.
        self._timeout_s = (2.0 * max_range_m / self.SPEED_OF_SOUND) + 0.01

    def ping(self):
        """Distance in metres, or None if nothing answered within range."""
        self.trig.on()
        time.sleep(0.00001)                    # 10 us trigger pulse
        self.trig.off()

        deadline = time.monotonic() + self._timeout_s
        while not self.echo.value:             # wait for the echo to go high
            if time.monotonic() > deadline:
                return None
        rise = time.monotonic()

        deadline = rise + self._timeout_s
        while self.echo.value:                 # ...and for it to fall again
            if time.monotonic() > deadline:
                return None
        width = time.monotonic() - rise

        distance = width * self.SPEED_OF_SOUND / 2.0
        return distance if 0.0 < distance <= self.max_range_m else None


class Ultrasonics:
    """All three sensors, fired one per tick so they never overlap.

    All three are polled equally often: the sides drive corridor centring, so they
    can't be starved for the front. Readings go stale rather than lingering --
    a 2-second-old range is worse than admitting you don't know.
    """

    # One ping per tick, round robin: each side is read every 3rd tick (~7 Hz) rather
    # than every 4th at half the ping rate. Centring runs off the sides, so they must
    # not lag. Anything not in SONARS_FITTED is skipped, so with only the front fitted
    # it gets every tick.
    ORDER = ["front", "left", "right"]

    def __init__(self, pins=None, enabled=True):
        self.sensors = {}
        self.readings = {}                     # name -> (distance_m|None, timestamp)
        self._i = 0
        self._last_ping = 0.0
        if not enabled:
            print("[sonar] disabled (--no-sonar)")
            return
        for name, (trig, echo) in (pins or ULTRASONIC_PINS).items():
            if name not in SONARS_FITTED:
                continue
            try:
                self.sensors[name] = Ultrasonic(trig, echo)
            except Exception as exc:           # noqa: BLE001
                print(f"[sonar] {name} unavailable on TRIG {trig}/ECHO {echo} -- "
                      f"{type(exc).__name__}: {exc}")
        if self.sensors:
            print(f"[sonar] {', '.join(sorted(self.sensors))}")

    def update(self):
        """Fire at most one sensor. Call once per control tick."""
        if not self.sensors:
            return
        now = time.monotonic()
        if now - self._last_ping < US_MIN_TRIGGER_GAP_S:
            return                             # too soon; last echo may still be alive
        # Skip past anything not fitted rather than burning a tick on it.
        for _ in range(len(self.ORDER)):
            name = self.ORDER[self._i % len(self.ORDER)]
            self._i += 1
            if name in self.sensors:
                break
        else:
            return
        sensor = self.sensors.get(name)
        if sensor is None:
            return
        self._last_ping = now
        self.readings[name] = (sensor.ping(), now)

    def get(self, name):
        """Latest range in metres, or None if unknown, out of range, or stale."""
        entry = self.readings.get(name)
        if entry is None:
            return None
        distance, t = entry
        if time.monotonic() - t > US_STALE_AFTER_S:
            return None
        return distance

    def since(self, name, t):
        """(True, distance) if `name` pinged after time t, else (False, None).

        A fresh None distance is real information -- no echo, nothing near -- which
        is why this exists alongside get(), where None also means "don't know".
        """
        entry = self.readings.get(name)
        if entry is None or entry[1] < t:
            return False, None
        return True, entry[0]

    def stamp(self, name):
        """When `name` last pinged, so callers can tell a new reading from a repeat."""
        entry = self.readings.get(name)
        return None if entry is None else entry[1]

    @property
    def front(self):
        return self.get("front")

    @property
    def available(self):
        return "front" in self.sensors


# ===========================================================================
# Drive
# ===========================================================================
class Drive:
    """Differential drive on the unit's controller board (0x57, i2c-8).

    Thin wrapper over motors.MotorDriver so nav has one call -- set_velocity(v, w),
    v forward m/s, w yaw rad/s with POSITIVE = LEFT. If the board isn't there this
    becomes a no-op that records commands, so the rest of the loop still runs.
    """

    def __init__(self, enabled=True):
        self.driver = None
        self.last = (0.0, 0.0)
        if not enabled:
            STATUS.event("[drive] disabled (--no-motors)")
            return
        try:
            from motors import MotorDriver
            self.driver = MotorDriver()
        except Exception as exc:                   # noqa: BLE001
            STATUS.event(f"[drive] NO MOTORS -- {type(exc).__name__}: {exc}")

    @property
    def board(self):
        """Truthy when motors are live -- what the status line reports on."""
        return self.driver

    def set_velocity(self, v_mps, w_rps):
        self.last = (v_mps, w_rps)
        if self.driver is not None:
            self.driver.set_velocity(v_mps, w_rps)

    def read_encoders(self):
        return None if self.driver is None else self.driver.read_encoders()

    def stop(self):
        """Never raises -- called on every exit path."""
        self.last = (0.0, 0.0)
        if self.driver is not None:
            self.driver.stop()

    def close(self):
        if self.driver is not None:
            self.driver.close()


# ===========================================================================
# Grid motion: turn, drive one cell, back out
# ===========================================================================
def _clamp(v, limit):
    return max(-limit, min(limit, v))


class Mover:
    """Moves the robot between cell centres, one tick at a time.

    Start a move with move_to(), then call update() every tick until it returns an
    outcome -- "arrived", or "blocked" once it has backed out to where it started.
    A move turns on the spot to the heading first if it isn't already facing it.

    It steers on the Localiser's pose, which the sonars keep correcting against the
    walls on the map. So while driving along a corridor it:
      * holds heading -- heading comes from the difference between the tracks, so
        holding it is what stops one track falling behind the other;
      * steers back onto the corridor's centreline whenever it's off it -- dead
        centre, with the side walls telling the pose where that is;
      * stops at the cell centre, where a wall ahead pins down exactly how far;
      * backs out if something turns up in the way that the map said was open.
    """

    IDLE, TURN, ALIGN, TRIM, DRIVE, BACKOUT = ("IDLE", "TURN", "ALIGN", "TRIM", "DRIVE",
                                               "BACKOUT")

    def __init__(self, drive, sonar, odo, camera_view=None):
        self.drive, self.sonar, self.odo = drive, sonar, odo
        self.camera_view = camera_view      # () -> fresh wallvision.WallView or None
        self.trim_eff = 1.0                 # learned: real turn per commanded turn, per pulse
        self.turn_retries = 0               # rounds of "turn again" used on this move
        self.phase = self.IDLE
        self.lateral = None                         # m right of centre, for the HUD
        self.near_since = None
        self.avoid = None       # (shift m right of centre, remaining at which it's past)

    @property
    def avoiding(self):
        return self.avoid is not None

    def note_obstacle(self, side_m, ahead_m):
        """The camera sees a floor obstacle `ahead_m` in front, `side_m` right of the
        corridor centreline (negative = left). Called by Nav on every sighting.

        Latches a sidestep to the opposite side for the rest of the way past it: the
        camera loses it long before the robot is level with it, and the side sonars only
        pick it up once it is alongside. Does nothing off the grid, when not driving
        forwards, or when the obstacle is too near the middle for the robot to squeeze by.
        """
        if (not OBSTACLE_AVOID or self.phase != self.DRIVE or not self.grid
                or not self.forward or abs(side_m) < OBSTACLE_MIN_SIDE_M):
            return
        o = self.odo
        ux, uy = math.cos(self.heading), math.sin(self.heading)
        remaining = (self.target[0] - o.x) * ux + (self.target[1] - o.y) * uy
        shift = -math.copysign(OBSTACLE_PASS_OFFSET_M, side_m)    # away from it
        first = self.avoid is None
        self.avoid = (shift, remaining - ahead_m - OBSTACLE_CLEAR_M)
        if first:
            STATUS.event(f"[nav] obstacle {abs(side_m)*100:.0f}cm to the "
                         f"{'right' if side_m > 0 else 'left'} of centre, {ahead_m*100:.0f}cm "
                         f"ahead -- sliding {OBSTACLE_PASS_OFFSET_M*100:.0f}cm "
                         f"{'left' if side_m > 0 else 'right'} to go past")

    def _avoid_shift(self, remaining):
        """The sidestep in force (metres right of centre), 0.0 when none; ends it once
        the robot is past the obstacle."""
        if self.avoid is None:
            return 0.0
        shift, until = self.avoid
        if remaining <= until:
            self.avoid = None
            return 0.0
        return shift

    @property
    def busy(self):
        return self.phase != self.IDLE

    def stop(self):
        self.phase = self.IDLE
        self.drive.stop()

    def move_to(self, x, y, heading, grid=True, through=False):
        """Face `heading` (world rad), then drive straight to (x, y).

        grid=True means a cardinal move between cell centres, which turns on blocked
        detection. A rejoin move after a victim approach is grid=False.

        through=True rolls over the target at full speed without stopping, for a
        known cell the route goes straight on from.
        """
        self.target, self.heading, self.grid = (x, y), heading, grid
        self.through = through
        self.turn_retries = 0
        self._yaw_stamp = None          # last wall view the heading was nudged by
        self.near_since = None          # when it came within ARRIVE_SLOW_M
        self.block_count = 0
        self.block_info = None          # (front reading, distance to go) when blocked
        self._front_stamp = None
        self.avoid = None               # a sidestep belongs to one leg
        now = time.monotonic()
        self.phase_started = now
        if abs(wrap(heading - self.odo.theta)) > TURN_TOLERANCE_RAD:
            self.phase, self.turn_sign = self.TURN, None
        else:
            self._begin_leg(now)

    def _begin_leg(self, now):
        """Start a leg: set the heading first (grid moves), then drive."""
        if self.grid and TRIM_ENABLED:
            self._begin_trim(now)
        else:
            self._begin_drive(now)

    def _begin_trim(self, now):
        self.phase = self.TRIM
        self.phase_started = now
        self.trim_settle_until = now
        self.trim_pulses = 0
        self.trim_before = None                     # (error, commanded angle) of the last pulse
        # Aim a little towards the corridor centre if it's off to one side.
        lateral = self._blended_offset() if CENTRING else None
        lean = 0.0
        if lateral is not None and abs(lateral) >= CENTRE_AIM_MIN_M:
            lean = _clamp(math.atan2(lateral, CENTRE_AIM_LEAD_M), CENTRE_AIM_MAX_RAD)
        self.aim = wrap(self.heading + lean)

    def _trim(self):
        """Pivot in short pulses until the encoder heading is on the aim.

        Each pulse is sized from how much turn a pulse has actually been giving (trim_eff,
        learned from the encoders), then it stands still while the encoders settle, so the
        next measurement is clean. A pulse is far finer than a control tick of pivoting
        (~6 deg), which is why this isn't done tick by tick.
        """
        now = time.monotonic()
        if now < self.trim_settle_until:
            return None
        err = wrap(self.aim - self.odo.theta)
        if self.trim_before is not None:                   # what did the last pulse do?
            before, commanded = self.trim_before
            moved = (before - err) * math.copysign(1.0, before)
            observed = moved / commanded if commanded > 1e-6 else 1.0
            self.trim_eff = _clamp(0.5 * self.trim_eff + 0.5 * max(observed, 0.05), 3.0) \
                if observed > 0.05 else max(0.05, 0.5 * self.trim_eff)
        if abs(err) <= TRIM_TOL_RAD:
            self._begin_drive(now)
            return None
        if self.trim_pulses >= TRIM_MAX_PULSES or now - self.phase_started > TRIM_MAX_S:
            if abs(err) > TURN_RETRY_ERR_RAD and self.turn_retries < TURN_MAX_RETRIES:
                self.turn_retries += 1
                STATUS.event(f"[nav] the turn stopped {math.degrees(err):+.0f} deg short -- turning "
                             f"again ({self.turn_retries}/{TURN_MAX_RETRIES}), not driving off")
                self.phase, self.turn_sign = self.TURN, None
                self.phase_started = now
                return None
            STATUS.event(f"[nav] heading trim gave up: {math.degrees(err):+.1f} deg still off "
                         f"after {self.trim_pulses} pulses -- driving on")
            self._begin_drive(now)
            return None
        rate = GRID_PIVOT_RATE
        pulse = _clamp(abs(err) / (rate * max(self.trim_eff, 0.05)), TRIM_PULSE_MAX_S)
        pulse = max(TRIM_PULSE_MIN_S, pulse)
        self.drive.set_velocity(0.0, math.copysign(rate, err))
        time.sleep(pulse)
        self.drive.stop()
        self.trim_before = (err, rate * pulse)
        self.trim_pulses += 1
        self.trim_settle_until = time.monotonic() + TRIM_SETTLE_S
        return None

    def _begin_drive(self, now):
        self.phase = self.DRIVE
        self.phase_started = now
        self.start = (self.odo.x, self.odo.y)
        self._start_lr = (self.odo.left_m, self.odo.right_m)
        self._odo_warned = False
        # Which way the target lies at the start. A forward move that overshoots has
        # arrived; only a move that STARTED behind the target reverses to it.
        ux, uy = math.cos(self.heading), math.sin(self.heading)
        self.forward = ((self.target[0] - self.odo.x) * ux
                        + (self.target[1] - self.odo.y) * uy) >= 0

    def update(self):
        if self.phase == self.TURN:
            return self._turn()
        if self.phase == self.ALIGN:
            return self._align()
        if self.phase == self.TRIM:
            return self._trim()
        if self.phase == self.DRIVE:
            return self._drive()
        if self.phase == self.BACKOUT:
            return self._backout()
        return None

    # -- phases ----------------------------------------------------------------
    def _turn(self):
        now = time.monotonic()
        err = wrap(self.heading - self.odo.theta)
        if self.turn_sign is None:
            self.turn_sign = 1.0 if err >= 0 else -1.0
        # Crossing the target counts as done: at the pivot floor it moves ~6 deg a
        # tick, so it can't always land inside the tolerance. The drive's heading
        # hold takes out what's left. (Only near the target -- at 180 deg the error
        # flips sign by wrapping, not by crossing.)
        crossed = err * self.turn_sign < 0 and abs(err) < math.pi / 2
        if (abs(err) <= TURN_TOLERANCE_RAD or crossed
                or now - self.phase_started > TURN_TIMEOUT_S):
            self.drive.stop()
            if self.camera_view is not None and ALIGN_AFTER_TURN and self.grid:
                self._begin_align(now)
            else:
                self._begin_leg(now)
            return None
        self.drive.set_velocity(0.0, math.copysign(GRID_PIVOT_RATE, err))
        return None

    def _begin_align(self, now):
        self.phase = self.ALIGN
        self.phase_started = now
        self.align_ready = now + ALIGN_SETTLE_S     # frames older than this saw it moving
        self.align_seen = False
        self.align_first = None                     # yaw when it started, for the log
        self.align_min = 0.0                        # shortest pulse to use; grows if they do nothing
        self.align_last_pulse = 0.0
        self.align_last = None                      # yaw before the last pulse
        self.align_worse = self.align_idle = 0      # bad pulses in a row, by kind

    def _wall_yaw(self, newer_than=None):
        """Robot yaw against the walls from the latest wall view, rad, + = turned left;
        None if there isn't a trustworthy one."""
        view = self.camera_view() if self.camera_view else None
        if (view is None or view.yaw_rad is None or abs(view.yaw_rad) > CAMERA_YAW_MAX_RAD
                or (newer_than is not None and view.t < newer_than)):
            return None, None
        return view.yaw_rad, view.t

    def _align(self):
        """Square up to the walls after a turn, so the next leg starts truly parallel.

        Measure the angle to the walls (side walls of the corridor it's now facing, or
        the wall ahead if it can see one), pulse the tracks to cancel it, wait for a
        fresh frame, repeat. The walls are on the grid, so being parallel to them is
        being exactly on the cardinal heading -- and the heading estimate is snapped to
        it, wiping out whatever the encoders drifted by on the turn.
        """
        now = time.monotonic()
        elapsed = now - self.phase_started
        yaw, _ = self._wall_yaw(newer_than=self.align_ready)
        if yaw is None and self.align_seen:
            # A fresh frame it can't get an angle from (walls out of view, or turned past
            # what the camera can measure) is a reading that's stopped tracking: stop here.
            view = self.camera_view() if self.camera_view else None
            if view is not None and view.t >= self.align_ready:
                STATUS.event(f"[nav] squaring up lost the walls after {elapsed:.2f}s (last yaw "
                             f"{math.degrees(self.align_last or 0):+.1f} deg) -- driving on")
                self._begin_leg(now)
                return None
        if yaw is None:
            if (not self.align_seen and elapsed > ALIGN_WAIT_S) or elapsed > ALIGN_MAX_S:
                if not self.align_seen:
                    STATUS.event("[nav] turn done, but no wall angle in view -- not squared up")
                self._begin_leg(now)
            return None
        self.align_seen = True
        if self.align_first is None:
            self.align_first = yaw
        if abs(yaw) <= ALIGN_TOL_RAD:
            self.odo.theta = wrap(self.heading + yaw)
            STATUS.event(f"[nav] squared up to the walls: {math.degrees(self.align_first):+.1f}"
                         f" -> {math.degrees(yaw):+.1f} deg off in {elapsed:.2f}s")
            self._begin_leg(now)
            return None
        # Did the last pulse help? Judge it by how the yaw moved.
        if self.align_last is not None:
            gain = abs(self.align_last) - abs(yaw)            # + = got closer
            if gain < -ALIGN_DIVERGE_RAD:
                self.align_worse += 1
                self.align_idle = 0
            elif gain < ALIGN_NO_EFFECT_RAD:
                self.align_idle += 1
                self.align_worse = 0
                # that pulse did nothing: it was probably inside the motors' dead band, so
                # the next must be longer than it (and than any proportional pulse)
                self.align_min = min(ALIGN_PULSE_CAP_S,
                                     max(self.align_min, self.align_last_pulse) * ALIGN_PULSE_GROWTH)
            else:
                self.align_worse = self.align_idle = 0
        if (self.align_worse >= ALIGN_GIVE_UP or self.align_idle >= ALIGN_NO_EFFECT_GIVE_UP
                or elapsed > ALIGN_MAX_S):
            why = ("the yaw got worse" if self.align_worse >= ALIGN_GIVE_UP else
                   "the pulses turned nothing" if self.align_idle >= ALIGN_NO_EFFECT_GIVE_UP
                   else "out of time")
            STATUS.event(f"[nav] squaring up gave up ({why}): {math.degrees(self.align_first):+.1f}"
                         f" -> {math.degrees(yaw):+.1f} deg after {elapsed:.2f}s -- driving on, "
                         "heading left as the encoders have it")
            self._begin_leg(now)
            return None
        # turned left (+) -> pulse right, and vice versa
        rate = MIN_TURN_RATE_PIVOT
        pulse = min(ALIGN_PULSE_CAP_S,
                    max(self.align_min, min(ALIGN_MAX_PULSE_S, max(0.01, abs(yaw) / rate))))
        self.align_last_pulse = pulse
        self.align_last = yaw
        self.drive.set_velocity(0.0, -math.copysign(rate, yaw))
        time.sleep(pulse)
        self.drive.stop()
        self.align_ready = time.monotonic() + ALIGN_SETTLE_S
        return None

    def _drive(self):
        now = time.monotonic()
        o = self.odo
        tx, ty = self.target
        ux, uy = math.cos(self.heading), math.sin(self.heading)
        rx, ry = uy, -ux                          # unit vector to the robot's right
        remaining = (tx - o.x) * ux + (ty - o.y) * uy

        if self.grid and self.forward:
            # Something much closer than the far side of the cell ahead is in the way.
            front, stamp = self._front_reading()
            fresh = stamp is not None and stamp != self._front_stamp
            self._front_stamp = stamp
            # Sidestepping an obstacle the camera has already seen: the front sonar will
            # read it (the cone is wide at first), but we're going round, so only call it
            # blocked once it's nearly touching.
            going_round = (front is not None and self.avoid is not None
                           and front >= OBSTACLE_FRONT_STOP_M)
            if (front is not None and not going_round
                    and front - FRONT_WALL_AT_CENTRE_M < remaining - BLOCK_MARGIN_M):
                if fresh:
                    self.block_count += 1
                if self.block_count >= BLOCK_CONFIRM:
                    STATUS.event(f"[nav] blocked: {front*100:.0f}cm ahead with "
                                 f"{remaining*100:.0f}cm to go -- backing out")
                    self.block_info = (front, remaining)
                    return self._begin_backout(now)
            elif fresh:
                self.block_count = 0

        if not self.grid and self.forward and remaining > ARRIVE_TOLERANCE_M:
            # Off the grid there is no wall-ahead check above, and the odometry this target
            # was worked out from may be out. A wall this close is as near as any cell centre
            # gets (FRONT_WALL_AT_CENTRE_M): stop here rather than drive into it.
            wall = self.sonar.front
            if wall is not None and wall <= REJOIN_WALL_STOP_M:
                STATUS.event(f"[nav] wall {wall*100:.0f}cm ahead with {remaining*100:.0f}cm "
                             "still to go -- stopping here, not driving into it")
                return self._arrive()

        if remaining <= ARRIVE_SLOW_M and self.near_since is None:
            self.near_since = now       # side walls read from here are this cell's
        if abs(remaining) <= ARRIVE_TOLERANCE_M or (remaining < 0 and self.forward):
            return self._arrive()
        if now - self.phase_started > DRIVE_TIMEOUT_S:
            STATUS.event("[nav] no arrival in time -- treating the cell as blocked")
            self.block_info = None
            return self._begin_backout(now)

        # Where the walls say it's pointing beats where the encoders think it is: pull
        # the heading estimate towards it on every fresh frame.
        if self.grid:
            yaw, stamp = self._wall_yaw()
            if yaw is not None and stamp != self._yaw_stamp:
                self._yaw_stamp = stamp
                o.theta = wrap(o.theta + CAMERA_YAW_NUDGE * wrap(self.heading + yaw - o.theta))

        # Off the centreline to the right -> aim a little left of the corridor, and
        # vice versa. The pose already carries what the side walls say.
        self.lateral = self._blended_offset() if CENTRING else 0.0
        if self.lateral is None:
            self.lateral = (o.x - tx) * rx + (o.y - ty) * ry
        steer_off = self.lateral             # self.lateral stays where the robot really is
        shift = self._avoid_shift(remaining)
        if shift and not self._beside_obstacle(shift):
            # Aim for the middle of the gap, not the middle of the corridor: being "shift"
            # off the centreline is on target. (Once the obstacle is alongside, both
            # sonars read it and the far wall at about the same range, so the plain
            # two-wall offset already is the middle of the gap.)
            steer_off -= shift
        if STEER_WHILE_DRIVING or not self.grid:
            aim = self.heading + _clamp(CENTRE_GAIN * steer_off, MAX_CENTRE_ANGLE)
            err = wrap(aim - o.theta)
            w = _clamp(HEADING_HOLD_GAIN * err, MAX_STEER_RATE)
            if STEER_MIN_RATE > 0.0:
                w = (math.copysign(max(abs(w), STEER_MIN_RATE), err)
                     if abs(err) > STEER_TOL_RAD else 0.0)
        else:
            w = 0.0                  # a speed difference can't steer this chassis (see above)
        v = SEARCH_SPEED if remaining > ARRIVE_SLOW_M or self.through else ARRIVE_SPEED
        if remaining < 0:
            v = -ARRIVE_SPEED                    # target is behind: back straight up
        self._check_odometry(now, ux, uy)
        self.drive.set_velocity(v, w)
        return None

    def _front_reading(self):
        """The wall ahead as (a front-sonar-equivalent reading, when it was taken).

        The sonar is the one that knows how far the wall is up close -- exactly where
        the camera is blind (its view of a wall's foot starts ~27 cm out) -- so it's
        used whenever it has an echo. With no echo (a soft or glancing wall) the camera
        steps in, if the wall is far enough for it to range. Both are expressed as what
        the front sonar would read, so the callers' thresholds don't care which it was.
        """
        front = self.sonar.front
        if front is not None:
            stamp = self.sonar.stamp("front") if hasattr(self.sonar, "stamp") else None
            return front, stamp
        view = self.camera_view() if self.camera_view else None
        if view is not None and not view.front_close and view.front_m is not None:
            from_centre = view.front_m - CAMERA_FORWARD_M
            return max(0.0, from_centre - (CELL_M / 2.0 - FRONT_WALL_AT_CENTRE_M)), view.t
        return None, None

    def _camera_offset(self):
        """Metres right of centre from the camera's wall view, or None. Walls sit on
        the cell boundaries, so a wall at the centre is half a cell away."""
        view = self.camera_view() if self.camera_view else None
        if view is None:
            return None
        half = CELL_M / 2.0
        if view.left_m is not None and view.right_m is not None:
            return (view.left_m - view.right_m) / 2.0
        if view.right_m is not None:
            return half - view.right_m
        if view.left_m is not None:
            return view.left_m - half
        return None

    def _blended_offset(self):
        """Sonar and camera offsets together: blended while they agree, the sonars
        alone when they don't (they're the better-proven of the two), and whichever
        one has anything when only one does."""
        sonar, cam = self._side_offset(), self._camera_offset()
        if cam is None:
            return sonar
        if sonar is None:
            return cam
        if abs(sonar - cam) <= CAMERA_AGREE_M:
            return CAMERA_WALL_WEIGHT * cam + (1.0 - CAMERA_WALL_WEIGHT) * sonar
        return sonar

    def _side_offset(self):
        """Metres right of the corridor centre read straight off the side sonars, or
        None if there's nothing usable (then the filter's pose is used).

        Both walls close: half the difference. Only one: its distance against where a
        wall reads from the centre. A reading too far to be this corridor's wall is
        an opening, and is ignored. Valid only while pointing along the corridor,
        which is when this is called.
        """
        reads = self._side_reads()
        if reads is None:
            return None
        return self._offset_from_reads(reads)

    @staticmethod
    def _offset_from_reads(reads):
        """Metres right of the corridor centre from {"left"|"right": wall distance}."""
        if "left" in reads and "right" in reads:
            return (reads["left"] - reads["right"]) / 2.0
        if "right" in reads:
            return SIDE_WALL_AT_CENTRE_M - reads["right"]
        if "left" in reads:
            return reads["left"] - SIDE_WALL_AT_CENTRE_M
        return None

    def _side_reads(self):
        """{"left"|"right": metres} for each side sonar with a fresh reading that is
        close enough to be this corridor's wall; None if the sonars can't say at all."""
        get = getattr(self.sonar, "get", None)
        stamp = getattr(self.sonar, "stamp", None)
        if get is None or stamp is None:
            return None
        now = time.monotonic()
        near = SIDE_WALL_AT_CENTRE_M + WALL_PRESENT_WINDOW_M
        reads = {}
        for name in ("left", "right"):
            t, d = stamp(name), get(name)
            if t is not None and d is not None and now - t <= SIDE_FRESH_S and d < near:
                reads[name] = d
        return reads

    def _beside_obstacle(self, shift):
        """Is the obstacle alongside? The sonar on ITS side (opposite the sidestep) reads
        far closer than any wall could be from a robot sitting in the gap."""
        reads = self._side_reads() or {}
        d = reads.get("right" if shift < 0 else "left")
        return d is not None and d < OBSTACLE_ALONGSIDE_M

    def _check_odometry(self, now, ux, uy):
        """Say so, once, if the encoders aren't keeping up with the drive command."""
        elapsed = now - self.phase_started
        if self._odo_warned or elapsed < ODO_CHECK_S or not self.forward:
            return
        o = self.odo
        gone = (o.x - self.start[0]) * ux + (o.y - self.start[1]) * uy
        expected = SEARCH_SPEED * (elapsed - 0.3)        # allow for the spin-up
        if gone >= ODO_MIN_SHARE * expected:
            return
        self._odo_warned = True
        counted = abs(o.left_m - self._start_lr[0]) + abs(o.right_m - self._start_lr[1])
        if counted < 0.005:
            STATUS.event(f"[odo] WARNING: {elapsed:.1f}s of driving and the encoders counted "
                         "NOTHING -- check the encoder wiring / board, or run with "
                         "--calibrate straight")
        else:
            rough = TICKS_PER_M * max(gone, 1e-3) / expected
            STATUS.event(f"[odo] WARNING: {elapsed:.1f}s at {SEARCH_SPEED*100:.0f}cm/s should "
                         f"be ~{expected*100:.0f}cm but the encoders say {gone*100:.0f}cm -- "
                         f"TICKS_PER_M ({TICKS_PER_M:.0f}) is probably too big, roughly "
                         f"{rough:.0f}. Measure it: python3 main.py --calibrate straight")

    def _arrive(self):
        if not self.through:
            self.drive.stop()
        self.phase = self.IDLE
        return "arrived"

    def _begin_backout(self, now):
        self.drive.stop()
        self.phase = self.BACKOUT
        self.phase_started = now
        return None

    def _backout(self):
        o = self.odo
        ux, uy = math.cos(self.heading), math.sin(self.heading)
        ahead = (o.x - self.start[0]) * ux + (o.y - self.start[1]) * uy
        if ahead <= ARRIVE_TOLERANCE_M or time.monotonic() - self.phase_started > DRIVE_TIMEOUT_S:
            self.drive.stop()
            self.phase = self.IDLE
            return "blocked"
        w = _clamp(HEADING_HOLD_GAIN * wrap(self.heading - o.theta), MAX_STEER_RATE)
        self.drive.set_velocity(-REVERSE_SPEED, w)
        return None


def sonar_mounts():
    """Where each sonar sits on the robot: (forward m, left m, direction rad).

    Derived from the calibrated centre readings, taking the walls as sitting on the
    cell boundaries: a sonar reading FRONT_WALL_AT_CENTRE_M to a wall half a cell
    away is mounted half a cell minus that ahead of the centre.
    """
    side = CELL_M / 2 - SIDE_WALL_AT_CENTRE_M
    return {"front": (CELL_M / 2 - FRONT_WALL_AT_CENTRE_M, 0.0, 0.0),
            "left": (SIDE_SONAR_FORWARD_M, side, math.pi / 2),
            "right": (SIDE_SONAR_FORWARD_M, -side, -math.pi / 2)}


class RescueStub:
    """Stands in for Roger's collection mechanism: 'succeeds' after a fixed time.

    Replace start()/done() with the real calls. done() returning True is what moves
    nav on -- to RETURN after collecting, back to exploring after releasing.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = None

    def start(self):
        self.started = time.monotonic()

    def done(self):
        return self.started is not None and time.monotonic() - self.started >= self.seconds


def heading_correction(bearing_deg, pivoting=False):
    """Yaw rate to put the victim on the nose, targeting HEADING_TOLERANCE_DEG.

    Inside the tolerance the correction is exactly zero -- without that deadband a
    proportional controller hunts either side of centre forever and never settles.

    Outside it, the magnitude is floored at MIN_TURN_RATE. That matters because the
    deadband compensation in motors.to_raw() turns anything smaller into a stop: a
    2 degree error would otherwise command a yaw the tracks never actually execute,
    and the robot would sit there off-heading believing it was correcting.

    Vision reports +ve to the RIGHT, drive takes +ve as LEFT, so the sign flips here.
    This is the one place the two conventions meet.
    """
    if abs(bearing_deg) <= HEADING_TOLERANCE_DEG:
        return 0.0
    floor = MIN_TURN_RATE_PIVOT if pivoting else MIN_TURN_RATE_MOVING
    w = math.copysign(max(abs(HEADING_GAIN * bearing_deg), floor), -bearing_deg)
    return max(-MAX_TURN_RATE, min(MAX_TURN_RATE, w))


def camera_blind_range_m(height_m=None, tilt_deg=None, vfov_deg=None):
    """How close a floor object can get before its base leaves the bottom of frame.

    Below this the camera cannot range it -- or see it at all -- which is why the
    last stretch of the approach has to run on sonar. Returns None if the camera
    geometry isn't known.
    """
    h = CAMERA_HEIGHT_M if height_m is None else height_m
    tilt = CAMERA_TILT_DEG if tilt_deg is None else tilt_deg
    vfov = VERTICAL_FOV_DEG if vfov_deg is None else vfov_deg
    if None in (h, tilt, vfov):
        return None
    edge_deg = tilt + vfov / 2.0        # depression angle at the bottom row
    if edge_deg <= 0:
        return None
    return h / math.tan(math.radians(edge_deg))


def horizon_row(frame_h, tilt_deg=None, vfov_deg=None):
    """Image row where the ground plane meets the horizon.

    Level camera -> the centre row. Tilted down -> the horizon rises up the frame.
    Everything below this row is ground; everything above is wall, marker or sky.
    """
    tilt = CAMERA_TILT_DEG if tilt_deg is None else tilt_deg
    vfov = VERTICAL_FOV_DEG if vfov_deg is None else vfov_deg
    if tilt is None or vfov is None:
        return frame_h / 2.0                      # best guess: assume level
    half = frame_h / 2.0
    # inverse of pixel_y_to_depression_angle at a depression of -tilt
    offset = math.tan(math.radians(-tilt)) / math.tan(math.radians(vfov / 2.0))
    return half * (1.0 + offset)


def is_on_floor(bbox, frame_h, margin_px=None):
    """True if this bbox's base is below the horizon, i.e. it's standing on the floor.

    This is what tells the victim object apart from the wall marker of the same colour.
    """
    margin = HORIZON_MARGIN_PX if margin_px is None else margin_px
    _, y, _, h = bbox
    return (y + h) > horizon_row(frame_h) + margin



class Victim:
    """One victim detection, already converted out of pixel space."""

    def __init__(self, bearing_deg, distance_m, bbox, area_px):
        self.bearing_deg = bearing_deg     # +ve = RIGHT of centre (vision's convention)
        self.distance_m = distance_m       # None when camera geometry isn't measured
        self.bbox = bbox                   # (x, y, w, h)
        self.area_px = area_px

    def __repr__(self):
        d = "?" if self.distance_m is None else f"{self.distance_m:.3f}m"
        return f"<Victim {self.bearing_deg:+.1f}deg {d} area={self.area_px}>"


UNUSABLE = object()     # frame carried no usable information (blurred, or no frame)


# The vision module's blur guard has been renamed by the vision team before, so look for
# whichever name it has now rather than assuming one. The old one is an absolute Laplacian
# variance (0 = off, which is what BLUR_THRESHOLD below sets); the newer one is relative
# and nav doesn't know its scale, so that one is left alone unless --blur-threshold is given.
BLUR_SETTING_NAMES = ("BLUR_VARIANCE_THRESHOLD", "BLUR_RELATIVE_THRESHOLD")


def blur_setting_name(vs):
    """The name of the blur guard setting in vision_system, or None if there isn't one."""
    for name in BLUR_SETTING_NAMES:
        if hasattr(vs, name):
            return name
    return None


class VictimVision:
    """Kushal's VisionSystem, narrowed to the one thing nav needs: the victim object.

    Two things about classify_frame() that nav has to respect:

    * It returns None when the frame is too motion-blurred to trust, which is
      deliberately NOT an empty list. Empty means "looked, nothing there"; None means
      "no information this cycle". Collapsing them would make a fast pan look exactly
      like the victim vanishing. look() passes that through as UNUSABLE.

    * Its is_ground_object flag means "a colour detection rather than a marker-template
      one" -- so the wall MARKER, which shares the victim's colour, comes back as
      is_ground_object=True too. Verified: a frame with the victim on the floor and the
      same image up a wall yields two detections, both class=victim, both ground=True.
      So we gate on geometry as well, via is_on_floor().
    """

    def __init__(self, use_placeholder=False):
        from camera_capture_v2_0 import (CAMERA_HEIGHT_M as CH, CAMERA_TILT_DEG as CT,
                                         VERTICAL_FOV_DEG as VF, CameraCapture)
        self.camera = CameraCapture()
        self.frame_w = self.frame_h = None
        self.geometry_ok = None not in (CH, CT, VF)
        self.system = None
        self._placeholder = use_placeholder
        # kept for the live window; nav itself only needs the Victim
        self.last_frame = None
        self.last_detections = []
        self.draw_detections = None
        # diagnostics, so a frame going nowhere is visible instead of silent
        self.last_blur = None
        self.blur_rejected = False
        self.blur_threshold = None
        self.class_counts = {}
        self.n_victim_raw = 0
        self.n_victim_kept = 0
        self.marker = None          # the victim WALL MARKER in view (a Victim, no range)
        self.n_marker = 0
        self.obstacle = None        # the nearest floor OBSTACLE in view (a Victim), or None
        self.hazard = None          # the HAZARD (dead-end) wall marker in view (a Victim, no range)
        self.n_hazard = 0

        if use_placeholder:
            print("[vision] PLACEHOLDER detector -- not the real vision system")
            return

        from vision_system_v2_0 import VisionSystem
        profiles = os.path.join(_VISION_DIR, "profiles.pkl")
        if not os.path.exists(profiles):
            raise SystemExit(f"profiles.pkl not found at {profiles}")
        cwd = os.getcwd()
        try:
            # VisionSystem and MarkerDetector both open paths relative to the cwd
            # (profiles.pkl, templates/), so run their construction from vision/.
            os.chdir(_VISION_DIR)
            self.system = VisionSystem("profiles.pkl")
        finally:
            os.chdir(cwd)
        self.draw_detections = type(self.system).draw_detections
        import vision_system_v2_0 as _vs
        name = blur_setting_name(_vs)
        self.blur_threshold = getattr(_vs, name) if name else None
        print(f"[vision] VisionSystem, filtering for {VICTIM_CLASS_NAME!r}"
              f"{' on the floor' if VICTIM_ON_FLOOR else ''}")

    def _is_target(self, det):
        """Is this detection the victim OBJECT, as opposed to its wall marker?"""
        if det.get("class") != VICTIM_CLASS_NAME:
            return False
        if not det.get("is_ground_object", False):
            return False                       # a marker-template hit
        if VICTIM_ON_FLOOR and not is_on_floor(det["bbox"], self.frame_h):
            return False                       # colour hit on the wall marker
        return True

    def _is_marker(self, det):
        """Is this detection the victim's WALL MARKER? The mirror of _is_target: a
        marker-template hit, or a victim-coloured blob whose base is up the wall."""
        if det.get("class") not in MARKER_CLASS_NAMES:
            return False
        if not det.get("is_ground_object", False):
            return True                        # a marker-template hit
        return not is_on_floor(det["bbox"], self.frame_h)

    def look(self):
        """The nearest victim, None if there isn't one, UNUSABLE if we can't tell.
        The victim's wall marker, if one is in view, is left in self.marker."""
        frame = self.camera.read()
        self.last_frame = frame
        if frame is None:
            self.last_detections = []
            return UNUSABLE
        self.frame_h, self.frame_w = frame.shape[:2]

        if self._placeholder:
            boxes = self._placeholder_boxes(frame)
            dets = [{"class": VICTIM_CLASS_NAME, "bbox": b, "is_ground_object": True,
                     "angle_deg": self._angle(b), "distance_m": None} for b in boxes]
        else:
            # Measure the same thing classify_frame gates on, so a rejection can be
            # reported with its number rather than silently dropping the frame.
            try:
                self.last_blur = float(self.system._blur_variance(frame))
            except Exception:                  # noqa: BLE001
                self.last_blur = None
            dets = self.system.classify_frame(frame)
            if dets is None:
                self.blur_rejected = True
                self.last_detections = []
                self.class_counts = {}
                self.n_victim_raw = self.n_victim_kept = 0
                return UNUSABLE                # too blurred to conclude anything
            self.blur_rejected = False

        self.last_detections = dets
        counts = {}
        for d in dets:
            counts[d.get("class", "?")] = counts.get(d.get("class", "?"), 0) + 1
        self.class_counts = counts

        markers = [d for d in dets if MARKER_CUE_ENABLED and self._is_marker(d)]
        self.n_marker = len(markers)
        if markers:
            m = max(markers, key=lambda d: d["bbox"][2] * d["bbox"][3])
            self.marker = Victim(m["angle_deg"], None, m["bbox"], m["bbox"][2] * m["bbox"][3])
        else:
            self.marker = None

        # The hazard marker is a wall placard: always a template hit, never a colour blob.
        hazards = [d for d in dets if HAZARD_CUE_ENABLED
                   and d.get("class") in HAZARD_CLASS_NAMES
                   and not d.get("is_ground_object", False)]
        self.n_hazard = len(hazards)
        if hazards:
            hz = max(hazards, key=lambda d: d["bbox"][2] * d["bbox"][3])
            self.hazard = Victim(hz["angle_deg"], None, hz["bbox"], hz["bbox"][2] * hz["bbox"][3])
        else:
            self.hazard = None

        # The nearest obstacle standing on the floor (a wall-mounted one is no use here).
        obstacles = [d for d in dets if OBSTACLE_AVOID
                     and d.get("class") == OBSTACLE_CLASS_NAME
                     and d.get("is_ground_object", False)
                     and is_on_floor(d["bbox"], self.frame_h)]
        if obstacles:
            # nearest = lowest base in the image; with no range that is the best proxy
            o = max(obstacles, key=lambda d: d["bbox"][1] + d["bbox"][3])
            self.obstacle = Victim(o["angle_deg"], o.get("distance_m"), o["bbox"],
                                   o["bbox"][2] * o["bbox"][3])
        else:
            self.obstacle = None

        raw = [d for d in dets if d.get("class") == VICTIM_CLASS_NAME]
        victims = [d for d in raw if self._is_target(d)]
        self.n_victim_raw, self.n_victim_kept = len(raw), len(victims)
        if not victims:
            return None

        # Largest box = nearest. Fine while one victim is in frame at a time.
        best = max(victims, key=lambda d: d["bbox"][2] * d["bbox"][3])
        x, y, w, h = best["bbox"]
        return Victim(best["angle_deg"], best["distance_m"], best["bbox"], w * h)

    def _angle(self, bbox):
        from camera_capture_v2_0 import pixel_x_to_angle
        x, _, w, _ = bbox
        return pixel_x_to_angle(x + w / 2.0, self.frame_w)

    def _placeholder_boxes(self, frame_bgr):
        """STAND-IN ONLY -- a crude HSV blob finder. Not a vision system."""
        import cv2
        import numpy as np
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(PLACEHOLDER_HSV_LOW), np.array(PLACEHOLDER_HSV_HIGH))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours
                if cv2.contourArea(c) >= PLACEHOLDER_MIN_AREA_PX]

    def close(self):
        try:
            self.camera.release()
        except Exception:                              # noqa: BLE001
            pass


class NullVision:
    """--no-camera: never sees a victim. For testing the maze nav on its own."""

    geometry_ok = False
    last_frame = None
    last_detections = []
    draw_detections = None
    marker = None
    n_marker = 0
    obstacle = None
    hazard = None
    n_hazard = 0

    def look(self):
        return None

    def close(self):
        pass


# ===========================================================================
# Nav loop
# ===========================================================================
SEARCH, SEEK, APPROACH, CLOSING, AT_VICTIM, RETURN, AT_BASE, DONE = (
    "SEARCH", "SEEK", "APPROACH", "CLOSING", "AT_VICTIM", "RETURN", "AT_BASE", "DONE")


class Nav:
    """The mission: explore the maze, approach and collect a victim, carry it home.

        SEARCH     frontier exploration, one cell at a time, mapping as it goes
        SEEK       a victim is confirmed: place it in a cell, drive there along the
                   corridors, face it from the next cell
        CLOSING    the last few cm on the front sonar -- the camera can't see the
                   floor that close
        APPROACH   only when a victim can't be placed (no range): steer at it by eye
        AT_VICTIM  stopped 10 cm short, collecting
        RETURN     flood fill home over the explored map; red LED while carrying
        AT_BASE    releasing, then back to SEARCH for the next victim
        DONE       all victims home, maze exhausted, or time up

    The map survives the whole run, so every trip out resumes from the frontier it
    left, and every trip home takes the shortest route through cells it has seen.
    """

    def __init__(self, drive, vision, leds, sonar=None, base_cell=None,
                 start_heading=None, wall_camera=None):
        self.drive, self.vision, self.leds = drive, vision, leds
        self.wallcam = wall_camera       # wallvision.WallCamera, or None for sonar only
        self.cam_walls = None            # latest wallvision.WallView
        self._cam_disagree = 0           # consecutive frames the camera contradicted the sonar
        self._cam_suspect_until = 0.0    # camera ignored until then (it contradicted the sonar)
        self._cam_warned_at = -1e9
        self._wall_frame = None          # the frame it was computed from
        self.sonar = sonar if sonar is not None else Ultrasonics(enabled=False)
        self.state = SEARCH
        self.hits = 0            # consecutive frames with a victim
        self.misses = 0          # consecutive frames without one
        self.victim = None
        self.arrived_at = None
        self._announced = False
        self.aligned = False        # was the victim well centred when last seen?
        self.closing_since = None
        self.heading_locked = False # committed to a heading; stop steering
        self.ticks = None           # latest cumulative encoder ticks

        # --- the maze ---
        self.base_cell = tuple(base_cell or BASE_CELL)
        start = HEADING_BY_NAME[(start_heading or START_HEADING).upper()]
        self.map = MazeMap(MAZE_COLS, MAZE_ROWS, CELL_M)
        self.odo = Localiser(TICKS_PER_M, EFFECTIVE_TRACK_M, sonar_mounts(),
                             sonar_sigma=SONAR_SIGMA_M, gate_known=GATE_KNOWN_M,
                             gate_maybe=GATE_MAYBE_M)
        self.odo.reset(*self.map.centre(self.base_cell), HEADING_RAD[start])
        self._loc_stamps = {}
        self.mover = Mover(drive, self.sonar, self.odo,
                           camera_view=self._camera_view if wall_camera is not None else None)
        self.cell = self.base_cell   # cell it's in (or leaving, mid-move)
        self.facing = start          # cardinal heading when last on the grid
        self.move_heading = None     # heading of the move in progress
        self.target_cell = None
        self.route = None            # planned cells, for the map window
        self.off_grid = False        # left the grid to chase a victim
        self.rejoin_stage = None     # "point" then "align" on the way back on
        self.sensing = True          # at a cell centre, reading the walls
        self.sense_from = None       # sensor -> readings newer than this count
        self.carrying = False
        self.victim_cell = None      # where the victim being fetched is
        self.seek_tries = {}         # victim cell -> failed sonar checks
        self.block_tries = {}        # (cell, heading) -> drives that were blocked
        self.closing_heading = None  # heading CLOSING holds
        self.rescued = 0
        self.marker_hits = 0         # consecutive frames the victim's wall marker was in view
        self.done_cells = set()      # cells a victim was collected from (their marker stays up)
        self.marker_cell = None      # the cell the current marker cue placed a victim in
        self.hazard_hits = 0         # consecutive frames the hazard marker was in view
        self.hazard_cells = set()    # dead-end cells a hazard marker put out of bounds
        self._hazard_refused = None  # the last cell a hazard marker named but was not allowed to block
        self.rechecking = False      # standing still after a pickup, re-anchoring the odometry
        self.recheck_from = None     # sonar readings newer than this count
        self._mock_announced = None       # arrived_at of the victim the mock collection announced
        self._edge_warned = set()         # (cell, heading) map-edge openings already reported
        self._creeping = False           # a short creep up to the front wall is under way
        self._creeped_cell = None
        self._victim_seen_at = -1e9      # when the camera last saw a victim
        self._anchor_path = 0.0          # odometry path length when a wall last pinned it down
        self.victim_points = []      # where victims were picked up, for the map
        self.rescue = RescueStub(RESCUE_TIME_S)
        self.release = RescueStub(RELEASE_TIME_S)
        self.mission_start = None
        self.finish_reason = None    # why it entered DONE -- always set with it

    # -- transitions ------------------------------------------------------
    def _finish(self, reason):
        """The only way into DONE. Says why, loudly, so a stop is never a mystery."""
        if self.state == DONE:
            return
        self.finish_reason = reason
        where = f"cell {self.cell}, {len(self.map.visited)} visited, {self.rescued} rescued"
        STATUS.event(f"[nav] *** FINISHED: {reason} ({where}, "
                     f"{(self.elapsed() or 0.0):.0f}s in) ***")
        self._enter(DONE)

    def _enter(self, state):
        if state == self.state:
            return
        STATUS.event(f"[nav] {self.state} -> {state}")
        self.state = state
        if state in (APPROACH, CLOSING, AT_VICTIM, DONE):
            self._leave_grid()
        if state == AT_VICTIM:
            self.arrived_at = time.monotonic()
            self.drive.stop()
            self.rescue.start()      # TODO: Roger's collect() goes here
        if state == CLOSING:
            self.closing_since = time.monotonic()
            self.closing_from = (self.odo.x, self.odo.y)
            self.closing_range = None
            if self.closing_heading is None:
                self.closing_heading = self.odo.theta   # straight on from here
            # Drop the stale detection: it's behind/underneath us now, and leaving it
            # around would draw a box where the victim no longer is.
            self.victim = None
        if state == AT_BASE:
            self.drive.stop()
            self.release.start()     # TODO: Roger's release() goes here
        if state == DONE:
            self.drive.stop()
        if state in (SEARCH, RETURN):
            self.closing_heading = None
        if state == SEARCH:
            self.victim_cell = self.marker_cell = None
            self._announced = False
            self.aligned = False
            self.heading_locked = False
            self.victim = None
            self.hits = 0
        if state == APPROACH:
            self.heading_locked = False
        self._set_leds()

    def _localise(self):
        """Correct the pose with every sonar reading that's new since last tick."""
        stamp = getattr(self.sonar, "stamp", None)
        if stamp is None:
            return
        for name in ("front", "left", "right"):
            t = stamp(name)
            if t is None or t == self._loc_stamps.get(name):
                continue
            self._loc_stamps[name] = t
            self.odo.correct(self.map, name, self.sonar.get(name))

    def _set_leds(self):
        """The assessment's colours, decided in one place from the state."""
        green = self.state in (SEEK, APPROACH, CLOSING, AT_VICTIM)
        red = self.state in (RETURN, AT_BASE) and self.carrying
        yellow = self.state in (SEARCH, RETURN) and not red
        self.leds.green(green)
        self.leds.yellow(yellow)
        self.leds.red(red)

    def _leave_grid(self):
        """Stop any grid move; the next grid state starts by getting back on."""
        if self.mover.busy:
            self.mover.stop()
        self.off_grid = True
        self.rejoin_stage = None
        self.route = None

    def elapsed(self):
        return None if self.mission_start is None else time.monotonic() - self.mission_start

    def state_label(self):
        label = STATE_LABEL.get(self.state, self.state)
        if self.state == RETURN and not self.carrying:
            label = "HEADING HOME (maze explored)"
        return label

    @property
    def can_range(self):
        """Either estimator will do. Without one, we refuse to approach."""
        return self.sonar.available or self.vision.geometry_ok

    def victim_range(self):
        """Best available distance to the tracked victim, and where it came from.

        The front sonar is the primary estimator, but it ranges whatever is nearest
        in its cone -- not specifically the victim. So it's only trusted while the
        victim is near centre; off to one side, the camera estimate (if calibrated)
        is the honest source, and if neither applies we return None rather than
        guessing. Approach steers the victim towards centre anyway, so the sonar
        becomes valid exactly when it matters most -- the last few centimetres.
        """
        if self.state == CLOSING:
            d = self.sonar.front
            return (d, "sonar") if d is not None else (None, None)

        v = self.victim
        if v is None:
            return None, None

        # A locked heading is an assertion that we are pointed at the victim, so the
        # front sonar is ranging it whatever the bearing currently reads. Without
        # this, a noisy bearing past US_TRUST_BEARING_DEG would drop the range and
        # send it pivoting -- exactly the behaviour the lock exists to stop.
        if self.heading_locked:
            d = self.sonar.front
            if d is not None:
                return d, "sonar"

        if abs(v.bearing_deg) <= US_TRUST_BEARING_DEG:
            d = self.sonar.front
            if d is not None:
                return d, "sonar"
        if v.distance_m is not None and VICTIM_ON_FLOOR:
            return v.distance_m, "camera"
        return None, None

    def step(self):
        if self.mission_start is None:
            self.mission_start = time.monotonic()
            self._set_leds()
        self.sonar.update()                    # one ping per tick, before deciding
        reader = getattr(self.drive, "read_encoders", None)
        self.ticks = reader() if reader else None
        self.odo.update(self.ticks)
        self._localise()

        if self.state != DONE and self.elapsed() >= MISSION_TIME_S:
            self._finish(f"mission time up ({MISSION_TIME_S:.0f}s)")

        seen = self.vision.look()
        self._look_at_walls()

        # Debounce both ways: a single frame should neither trigger an approach
        # nor abandon one. Cheap insurance against detector flicker.
        if seen is UNUSABLE:
            # Blurred frame or dropped capture -- no evidence either way. Hold the
            # counters and keep acting on what we already believe, rather than
            # letting a fast pan read as the victim disappearing.
            pass
        elif seen is not None:
            self.hits, self.misses = self.hits + 1, 0
            self.victim = seen
            self._victim_seen_at = time.monotonic()
        else:
            self.misses, self.hits = self.misses + 1, 0
        if seen is not UNUSABLE:
            marker = getattr(self.vision, "marker", None)
            self.marker_hits = self.marker_hits + 1 if marker is not None else 0
            hazard = getattr(self.vision, "hazard", None)
            self.hazard_hits = self.hazard_hits + 1 if hazard is not None else 0

        if seen is not UNUSABLE:
            self._note_obstacle()
            self._hazard_cue()

        confirmed = self.hits >= DETECTION_DEBOUNCE and self._wanted(self.victim)
        lost = self.misses > LOST_GRACE_FRAMES

        if self.state == SEARCH:
            self._search(confirmed)
        elif self.state == SEEK:
            self._seek()
        elif self.state == APPROACH:
            self._approach(lost)
        elif self.state == CLOSING:
            self._closing()
        elif self.state == AT_VICTIM:
            self._at_victim()
        elif self.state == RETURN:
            self._return()
        elif self.state == AT_BASE:
            self._at_base()
        elif self.state == DONE:
            self._done()

    def _note_obstacle(self):
        """Hand a floor obstacle the camera can see to the mover, as (metres right of the
        corridor centreline, metres ahead), so it can slide to the other side of the
        corridor to pass it. Only while driving a grid leg -- the mover ignores it
        otherwise -- and only for ones near enough to matter."""
        ob = getattr(self.vision, "obstacle", None)
        if not OBSTACLE_AVOID or ob is None or not self.mover.busy:
            return
        rng = ob.distance_m if ob.distance_m is not None else OBSTACLE_ASSUMED_RANGE_M
        if rng > OBSTACLE_SEE_MAX_M:
            return
        bearing = math.radians(ob.bearing_deg)           # +ve = right of the nose
        ahead = rng * math.cos(bearing)
        own = self.mover.lateral or 0.0                  # where we are in the corridor
        self.mover.note_obstacle(own + rng * math.sin(bearing), ahead)

    def _look_at_walls(self):
        """Run the wall detector on the frame vision just grabbed (once per frame)."""
        if self.wallcam is None:
            return
        frame = getattr(self.vision, "last_frame", None)
        if frame is None or frame is self._wall_frame:
            return
        self._wall_frame = frame
        try:
            self.cam_walls = self.wallcam.look(frame)
            self._check_camera_agrees()
        except Exception as exc:                       # noqa: BLE001
            STATUS.event(f"[walls] camera wall detection failed, sonar only -- "
                         f"{type(exc).__name__}: {exc}")
            self.wallcam = self.cam_walls = None

    def _check_camera_agrees(self):
        """A wall right in front (closer than ~21 cm) can't coexist with a front sonar
        that reads far. If the camera keeps claiming it, its picture is wrong -- most
        likely something bright in the bottom of the frame (the robot's own chassis, a
        glare patch, a white floor marking) -- so say so once."""
        view, front = self.cam_walls, self.sonar.front
        if view is not None and view.front_close and front is not None and front > 0.35:
            self._cam_disagree += 1
        else:
            self._cam_disagree = 0
        if self._cam_disagree >= CAMERA_DISAGREE_TICKS:
            now = time.monotonic()
            self._cam_suspect_until = now + CAMERA_SUSPECT_S      # ignore it for a while
            if now - self._cam_warned_at > 30.0:
                self._cam_warned_at = now
                STATUS.event(f"[walls] WARNING: the camera says a wall is right ahead but the "
                             f"front sonar reads {front*100:.0f}cm -- the camera view is probably "
                             "catching something bright at the bottom of the frame (chassis? "
                             "glare?). Ignoring the camera's walls for "
                             f"{CAMERA_SUSPECT_S:.0f}s; run  python3 wall_vision_test.py --save  "
                             "facing down the corridor and check runs/*_mask.png")

    def _camera_summary(self):
        view = self._camera_view()
        if view is None:
            return "no wall view"

        def cm(v):
            return "--" if v is None else f"{v*100:.0f}"
        return (f"cam: front {'CLOSE' if view.front_close else cm(view.front_m)}, "
                f"L {cm(view.left_m)}, R {cm(view.right_m)} cm, {view.columns} boundary pts")

    def _camera_view(self):
        """The latest wall view if it's recent enough to act on."""
        view = self.cam_walls
        if view is None or view.age() > CAMERA_FRESH_S:
            return None
        if time.monotonic() < self._cam_suspect_until:
            return None                  # it's been contradicting the sonar: don't act on it
        return view

    def front_distance(self):
        """(metres from the turning centre to the wall ahead, "cam" | "sonar"), or
        (None, None) if neither can tell.

        The camera ranges a wall further out than ~27 cm from the lens; any closer it
        only knows "close" -- and that's when the front sonar is the one to ask.
        """
        view = self._camera_view()
        if view is not None and not view.front_close and view.front_m is not None:
            return max(0.0, view.front_m - CAMERA_FORWARD_M), "cam"
        front = self.sonar.front
        if front is not None:
            return front + (CELL_M / 2.0 - FRONT_WALL_AT_CENTRE_M), "sonar"
        return None, None

    def _camera_sees_wall(self, name):
        """Does the camera see a wall on this side of the cell? None if it can't say.
        Only ever vouches FOR a wall -- not seeing one proves nothing."""
        view = self._camera_view()
        if view is None:
            return None
        reach = CELL_M / 2.0 + CAMERA_WALL_TOLERANCE_M
        if name == "front":
            return bool(view.front_close or (view.front_m is not None
                                            and view.front_m - CAMERA_FORWARD_M <= reach))
        dist = view.left_m if name == "left" else view.right_m
        return dist is not None and dist <= reach

    def _wanted(self, v):
        """Is this a victim still to be rescued?

        Not while one is on board, not one lying in the base cell -- that's a victim
        already delivered, and chasing it would 'rescue' it again -- and not one in a
        cell that has already failed the sonar check SEEK_ATTEMPTS times.
        """
        if v is None or self.carrying:
            return False
        delivered = self.rescued > 0          # only then is there one lying in base
        spot = self._victim_spot(v)
        if spot is None:
            here = self.map.cell_at(self.odo.x, self.odo.y)
            return not (delivered and here == self.base_cell)
        cell = self.map.cell_at(*spot)
        if delivered and cell == self.base_cell:
            return False
        return self.seek_tries.get(cell, 0) < SEEK_ATTEMPTS

    def _victim_spot(self, v):
        """World (x, y) of a detection, or None if it can't be ranged."""
        distance = v.distance_m
        if distance is None and abs(v.bearing_deg) <= US_TRUST_BEARING_DEG:
            distance = self.sonar.front
        if distance is None:
            return None
        o = self.odo
        cx = o.x + CAMERA_FORWARD_M * math.cos(o.theta)
        cy = o.y + CAMERA_FORWARD_M * math.sin(o.theta)
        a = o.theta - math.radians(v.bearing_deg)          # bearing is +ve RIGHT
        return cx + distance * math.cos(a), cy + distance * math.sin(a)

    def _search(self, confirmed):
        if confirmed:
            self.leds.green(True)                      # green the moment we commit
            self.leds.yellow(False)
            if not self._announced:
                STATUS.event("[nav] VICTIM DETECTED -- green LED on")
                self._announced = True
            spot = self._victim_spot(self.victim)
            if spot is not None:
                self.victim_cell = self.map.cell_at(*spot)
                self.map.blocked.discard(self.victim_cell)
                STATUS.event(f"[nav] victim placed in cell {self.victim_cell}")
                self._enter(SEEK)
            elif self.can_range:
                self._enter(APPROACH)
            else:
                # Refuse to drive at a victim we cannot range at all. Holding with
                # green lit is the honest behaviour -- fit the front sonar, or
                # measure the camera constants.
                self._leave_grid()
                self.drive.stop()
            return
        if self._marker_cue():
            return
        self._set_leds()
        self._grid(self._plan_explore)

    def _hazard_cue(self):
        """The hazard marker is in view: the cell in front of its wall is a dead end, so
        take it out of the map -- never to be entered, no longer ground left to explore.

        Placed like the victim marker's cell: the front sonar ranges the marker's wall (only
        believed while the marker is near the nose and the robot points along the grid), and
        the cell is HAZARD_CELL_INSET_M back from it. It is refused, with the reason in the
        log, if it is the base cell, a cell a victim is in, or if blocking it would cut the
        robot off from the way home. The cell the robot is standing in is remembered and
        blocked as soon as it has left it (a blocked start cell has no route out).
        """
        hz = getattr(self.vision, "hazard", None)
        if (not HAZARD_CUE_ENABLED or hz is None or self.hazard_hits < DETECTION_DEBOUNCE
                or self.state not in (SEARCH, SEEK, RETURN)
                or abs(hz.bearing_deg) > US_TRUST_BEARING_DEG):
            return
        o = self.odo
        if abs(wrap(o.theta - HEADING_RAD[nearest_heading(o.theta)])) > math.radians(
                HAZARD_HEADING_TOL_DEG):
            return                     # the sonar isn't ranging along a corridor
        front = self.sonar.front
        if front is None:
            return
        to_wall = front + (CELL_M / 2.0 - FRONT_WALL_AT_CENTRE_M)
        reach = max(0.0, to_wall - HAZARD_CELL_INSET_M)
        a = o.theta - math.radians(hz.bearing_deg)             # bearing is +ve RIGHT
        cell = self.map.cell_at(o.x + reach * math.cos(a), o.y + reach * math.sin(a))
        if cell in self.hazard_cells:
            return
        if cell == self.base_cell or cell == self.victim_cell:
            if self._hazard_refused != cell:
                self._hazard_refused = cell
                STATUS.event(f"[nav] hazard marker seen, but its cell {cell} is the "
                             f"{'base' if cell == self.base_cell else 'victim'} cell -- "
                             "not blocking it")
            return
        if cell != self.cell:
            # Would blocking it cut the robot off from the way home? Check before committing.
            self.map.blocked.add(cell)
            cut_off = self.cell not in self.map.flood([self.base_cell], True)
            self.map.blocked.discard(cell)
            if cut_off:
                if self._hazard_refused != cell:
                    self._hazard_refused = cell
                    STATUS.event(f"[nav] hazard marker seen, but blocking {cell} would cut the "
                                 "way back to base -- not blocking it")
                return
        self.hazard_cells.add(cell)
        STATUS.event(f"[nav] HAZARD MARKER seen (wall {to_wall*100:.0f}cm ahead) -- cell {cell}, "
                     f"the dead end in front of it, is out of bounds (robot is in {self.cell})")
        if cell == self.cell:
            STATUS.event("[nav] ...the robot is standing in it; it will be blocked once it has left")
            return
        self.map.blocked.add(cell)
        if self.mover.busy and self.target_cell == cell:
            # Already on its way in: turn back to the centre of the cell it left.
            STATUS.event(f"[nav] heading into the hazard cell {cell} -- turning back to {self.cell}")
            self.target_cell = self.cell
            self.mover.move_to(*self.map.centre(self.cell), HEADING_RAD[self.move_heading])

    def _apply_hazards(self):
        """Block every remembered hazard cell the robot is no longer standing in."""
        for hc in self.hazard_cells:
            if hc != self.cell:
                self.map.blocked.add(hc)

    def _marker_cue(self):
        """The victim's wall marker is in view and there is no victim object to chase:
        place a victim in the cell BEFORE the marker, light green and go and look
        there. True if it committed (state is now SEEK).

        The cell comes from the front sonar, which is only believed while the marker is
        near the nose (it then ranges the marker's own wall). From the turning centre the
        wall is the sonar reading plus the sonar's offset; the victim goes MARKER_CELL_INSET_M
        back from there (half a cell and a little more), so it lands in the cell in front of
        the wall rather than on it. SEEK then drives there, and the usual sonar check at the
        last step decides whether anything is really standing in it.

        A victim OBJECT in view always wins: while one is being confirmed this waits, and
        the object's own spot becomes the placement (see _search and _seek).
        """
        marker = getattr(self.vision, "marker", None)
        if (not MARKER_CUE_ENABLED or marker is None or self.carrying
                or self.marker_hits < DETECTION_DEBOUNCE
                or abs(marker.bearing_deg) > US_TRUST_BEARING_DEG):
            return False
        if self.hits > 0 and self._wanted(self.victim):
            return False               # the object is in view: let it place the victim
        front = self.sonar.front
        if front is None:
            return False
        o = self.odo
        to_wall = front + (CELL_M / 2.0 - FRONT_WALL_AT_CENTRE_M)
        reach = max(0.0, to_wall - MARKER_CELL_INSET_M)
        a = o.theta - math.radians(marker.bearing_deg)        # bearing is +ve RIGHT
        cell = self.map.cell_at(o.x + reach * math.cos(a), o.y + reach * math.sin(a))
        if (cell in self.done_cells or self.seek_tries.get(cell, 0) >= SEEK_ATTEMPTS
                or (self.rescued > 0 and cell == self.base_cell)):
            return False
        self.victim_cell = self.marker_cell = cell
        self.map.blocked.discard(cell)
        self.leds.green(True)                          # green the moment we commit
        self.leds.yellow(False)
        self._announced = True
        STATUS.event(f"[nav] VICTIM MARKER seen (wall {to_wall*100:.0f}cm ahead) -- green LED "
                     f"on, victim placed in cell {cell}, the cell before the marker "
                     f"(robot is in {self.map.cell_at(o.x, o.y)})")
        self._enter(SEEK)
        return True

    # -- the grid ------------------------------------------------------------
    def _grid(self, plan):
        """One tick of cell-to-cell travel. `plan` picks the next heading at each
        cell centre, or returns None having dealt with there being nowhere to go."""
        if self.mover.busy:
            outcome = self.mover.update()
            if outcome == "arrived":
                self._arrived()
            elif outcome == "blocked":
                self._blocked(self.mover.block_info)
            return

        if self.off_grid:
            self._rejoin()
            return

        if self.sensing:
            now = time.monotonic()
            self.drive.stop()
            if self.sense_from is None:
                self.sense_from = {n: now for n in ("front", "left", "right")}
            fitted = [n for n in ("front", "left", "right")
                      if n in getattr(self.sonar, "sensors", {})]
            fresh = all(self.sonar.since(n, self.sense_from[n])[0] for n in fitted)
            if not fresh and now - self.sense_from["front"] < SENSE_TIMEOUT_S:
                return                             # wait for a full set of pings
            self._trace_stop(fitted)
            if self._front_creep(fitted):
                return                             # creeping up to the front wall first
            self._record_walls(fitted)
            self.sensing = False

        heading = plan()
        if heading is None:
            return
        self.move_heading = heading
        self.target_cell = self.map.neighbour(self.cell, heading)
        # Straight on through a cell it already knows? Then don't stop in it.
        r = self.route
        through = (r is not None and len(r) > 2 and r[1] == self.target_cell
                   and self.target_cell in self.map.visited
                   and self.map.heading_between(r[1], r[2]) == heading)
        self.mover.move_to(*self.map.centre(self.target_cell), HEADING_RAD[heading],
                           through=through)

    def _blocked(self, info):
        """A drive the map said was open wasn't. Work out what was in the way.

        Past the boundary, it's something standing IN the next cell -- rubble, or a
        victim the camera was too close to see -- so the cell is marked, not the
        wall. At the boundary it's a wall the sonar missed: locked in for good.
        """
        self.facing = self.move_heading
        edge = (self.cell, self.move_heading)
        self.block_tries[edge] = self.block_tries.get(edge, 0) + 1
        if self.target_cell in self.map.visited and self.block_tries[edge] < 2:
            # It's been in that cell, so the way is (or was) open -- more likely a
            # bad reading from off-centre. Re-read the walls here and try again.
            STATUS.event(f"[nav] blocked into {self.target_cell}, which it has been "
                         "in -- re-checking")
            self.sensing = True
            self.sense_from = None
            return
        if info is not None:
            front, along = info
            in_cell = front - FRONT_WALL_AT_CENTRE_M > along - CELL_M + 0.03
            if in_cell and self.state == SEEK:
                # Going to a victim and something is standing in the next cell: that's
                # it, closer than the camera can see. Face it and close in.
                STATUS.event(f"[nav] the victim is in {self.target_cell}")
                self.victim_cell = self.target_cell
                return
            if in_cell and self.target_cell not in self.map.visited:
                if self.target_cell != self.victim_cell:
                    self.map.blocked.add(self.target_cell)
                STATUS.event(f"[nav] something in {self.target_cell} -- marked blocked")
                return
        STATUS.event(f"[nav] {self.cell}->{HEADING_NAME[self.move_heading]} "
                     "is walled -- marked on the map")
        self.map.set_wall(self.cell, self.move_heading, True, lock=True)

    def _check_edge_is_open(self, heading, d, window_edge):
        """Warn, once per edge, when the sonar sees clear space where the map says the maze ends.

        The map's outer edge is a locked wall, so an opening there is ignored. If the robot
        is genuinely looking out of the maze, it's the base cell / heading that's wrong.
        """
        if self.map.in_bounds(self.map.neighbour(self.cell, heading)):
            return
        if d is not None and d < window_edge + 0.10:
            return                                 # a wall, or close enough to be one
        key = (self.cell, heading)
        if key in self._edge_warned:
            return
        self._edge_warned.add(key)
        seen = "no echo within range" if d is None else f"{d*100:.0f}cm of open space"
        STATUS.event(f"[map] WARNING: {self.cell} {HEADING_NAME[heading]} is the edge of the map "
                     f"(treated as a solid wall) but the sonar sees {seen}. If that's a real "
                     f"opening the base cell or heading is wrong: current base {self.base_cell} "
                     f"facing {HEADING_NAME[self.facing]}; try --base-cell col,row "
                     "--start-heading N/E/S/W")

    def _record_walls(self, fitted):
        """Map the three sides it can see from this cell centre."""
        for name, heading in (("front", self.facing), ("left", LEFT_OF[self.facing]),
                              ("right", RIGHT_OF[self.facing])):
            if name not in fitted:
                continue
            fresh, d = self.sonar.since(name, self.sense_from[name])
            if not fresh:
                continue
            at_centre = FRONT_WALL_AT_CENTRE_M if name == "front" else SIDE_WALL_AT_CENTRE_M
            window = WALL_PRESENT_WINDOW_M
            if self.map.neighbour(self.cell, heading) == self.victim_cell:
                # Towards the victim, a reading that's short but not wall-short is the
                # victim standing just beyond the boundary -- don't seal it off.
                window = VICTIM_WALL_WINDOW_M
            present = d is not None and d < at_centre + window
            self._check_edge_is_open(heading, d, at_centre + window)
            if (not present and self._camera_sees_wall(name)
                    and self.map.neighbour(self.cell, heading) != self.victim_cell):
                shown = "no echo" if d is None else f"{d*100:.0f}cm"
                if d is None or d < at_centre + window + CAMERA_SONAR_SLACK_M:
                    STATUS.event(f"[walls] {self.cell} {name}: sonar says open ({shown}), "
                                 "camera sees a wall -- marking it")
                    present = True
                else:
                    STATUS.event(f"[walls] {self.cell} {name}: camera sees a wall but the "
                                 f"sonar reads {shown} of clear space -- trusting the sonar "
                                 f"[{self._camera_summary()}]")
            self.map.set_wall(self.cell, heading, present)
        self.map.visited.add(self.cell)

    def _trace_stop(self, fitted):
        """One line per cell stop: what the sonars read and where the odometry thinks it is.

        `off` is the odometry's position against the cell centre, along the way it faces
        and across it; if the robot is physically further short than that says, the front
        reading shows it (a wall ahead reads ~6cm from a centred stop).
        """
        if getattr(self, "_traced_cell", None) == self.cell:
            return
        self._traced_cell = self.cell
        parts = []
        for name in ("front", "left", "right"):
            if name in fitted:
                ok, d = self.sonar.since(name, self.sense_from[name])
                parts.append(f"{name[0].upper()} {'--' if not ok or d is None else f'{d*100:.0f}'}")
        cx, cy = self.map.centre(self.cell)
        h = HEADING_RAD[self.facing]
        along = (self.odo.x - cx) * math.cos(h) + (self.odo.y - cy) * math.sin(h)
        across = (self.odo.x - cx) * math.sin(h) - (self.odo.y - cy) * math.cos(h)
        STATUS.event(f"[stop] {self.cell}{HEADING_NAME[self.facing]}  sonar {' '.join(parts)} cm  "
                     f"odo off centre {along*100:+.0f}cm along {across*100:+.0f}cm across  "
                     f"scale {getattr(self.odo, 'dscale', 1.0):.3f}  "
                     f"heading {math.degrees(wrap(self.odo.theta - h)):+.1f}deg")

    def _learn_scale(self, short):
        """Fold one "it was `short` metres short of the wall" into the distance scale.

        short > 0: the odometry went further than the robot did, so its scale is too big.
        Needs a decent run since the last anchor, or a few cm of noise looks like a scale.
        """
        o = self.odo
        path = o.path_m - self._anchor_path
        self._anchor_path = o.path_m
        if path < SCALE_MIN_PATH_M or not hasattr(o, "dscale"):
            return
        # An ordinary stop is always a centimetre or so short (it arrives within a tolerance
        # of the target). Learn from that and a perfect robot slowly decides its scale is
        # wrong; only a gap bigger than that is a scale error.
        if abs(short) < CREEP_MIN_M:
            return
        factor = (path - short) / path                  # real distance / odometry's
        new = o.dscale * (1.0 - SCALE_LEARN_RATE * (1.0 - factor))
        new = max(SCALE_LIMITS[0], min(SCALE_LIMITS[1], new))
        if abs(new - o.dscale) > 0.004:
            STATUS.event(f"[odo] distance scale {o.dscale:.3f} -> {new:.3f} "
                         f"({short*100:+.0f}cm out over {path*100:.0f}cm driven)")
        o.dscale = new

    def _front_creep(self, fitted):
        """If it stopped short of this cell's centre, start a creep up to the front wall.

        True if a creep was started (the cell is then sensed again on arrival).
        """
        if (not CREEP_ENABLED or "front" not in fitted or self._creeped_cell == self.cell
                or self.state in (SEEK, APPROACH, CLOSING, AT_VICTIM)):
            return False
        if time.monotonic() - self._victim_seen_at < CREEP_VICTIM_HOLD_S:
            return False       # the sonar is likely reading the victim, not a wall
        ok, reading = self.sonar.since("front", self.sense_from["front"])
        if not ok or reading is None:
            return False
        # How far short: where the wall is minus where it would be from the centre.
        short = reading - FRONT_WALL_AT_CENTRE_M
        # The further it has driven since a wall last told it where it is, the more error
        # it can have built up, so the further short it may plausibly be.
        path = self.odo.path_m - self._anchor_path
        reach = min(CREEP_ABS_MAX_M, max(CREEP_MAX_M, CREEP_PATH_SHARE * path))
        if -CREEP_MAX_M / 3 <= short <= reach:
            self._learn_scale(short)         # a wall within this cell: a real measurement
        if not CREEP_MIN_M <= short <= reach:
            if short > reach and short < 0.32:
                STATUS.event(f"[nav] {self.cell}: front wall {short*100:.0f}cm beyond a centred "
                             f"stop -- too far to be this cell's wall (reach {reach*100:.0f}cm "
                             f"after {path*100:.0f}cm driven), not creeping")
            return False
        if (self.map.neighbour(self.cell, self.facing) == self.victim_cell
                or self.map.neighbour(self.cell, self.facing) in self.map.blocked):
            return False                     # that's a victim/rubble in the next cell, not a wall
        self._creeped_cell = self.cell
        self._creeping = True
        ux, uy = math.cos(HEADING_RAD[self.facing]), math.sin(HEADING_RAD[self.facing])
        STATUS.event(f"[nav] {self.cell}: stopped {short*100:.0f}cm short of the centre (front wall "
                     f"at {reading*100:.0f}cm, should be {FRONT_WALL_AT_CENTRE_M*100:.0f}cm) -- "
                     "creeping up to it")
        self.move_heading = self.facing
        self.target_cell = self.cell
        self.sense_from = None
        self.mover.move_to(self.odo.x + short * ux, self.odo.y + short * uy,
                           HEADING_RAD[self.facing])
        return True

    def _arrived(self):
        if self._creeping:
            # The sonar says it's at the wall, which is the cell edge: this IS the centre.
            # Pull the odometry's along-track position onto it so the error doesn't carry on.
            self._creeping = False
            cx, cy = self.map.centre(self.cell)
            if self.facing in (EAST, WEST):
                self.odo.x = cx
            else:
                self.odo.y = cy
            self.sense_from = None
            self.sensing = True
            return
        if self.rejoin_stage == "point":
            # At the cell centre; now square up to the nearest cardinal heading so
            # the walls can be read.
            self.rejoin_stage = "align"
            self.facing = nearest_heading(self.odo.theta)
            self.mover.move_to(*self.map.centre(self.cell), HEADING_RAD[self.facing])
            return
        now = time.monotonic()
        if self.rejoin_stage == "align":
            self.rejoin_stage = None
            self.off_grid = False
            self.sense_from = None
        else:
            self.cell = self.target_cell
            self.facing = self.move_heading
            # The side walls were read on the way in; the front has to be read from
            # here. A cell already mapped isn't re-read at all.
            near = self.mover.near_since or now
            self.sense_from = {"front": now, "left": near, "right": near}
        self.sensing = self.cell not in self.map.visited
        self._apply_hazards()

    def _rejoin(self):
        """Back onto the grid after chasing a victim: go to the centre of the cell it's
        in, then face the nearest cardinal heading."""
        self.cell = self.map.cell_at(self.odo.x, self.odo.y)
        cx, cy = self.map.centre(self.cell)
        dx, dy = cx - self.odo.x, cy - self.odo.y
        STATUS.event(f"[nav] rejoining the grid at {self.cell}")
        facing = nearest_heading(self.odo.theta)
        u = HEADING_RAD[facing]
        sideways = abs(dx * math.sin(u) - dy * math.cos(u))
        if math.hypot(dx, dy) < 0.03 or sideways < 0.04:
            # On (or near) the cell's centreline already -- square up and roll
            # straight forward or back onto the centre, no about-turn.
            self.rejoin_stage = "align"
            self.facing = facing
            self.mover.move_to(cx, cy, u)
        else:
            self.rejoin_stage = "point"
            self.mover.move_to(cx, cy, math.atan2(dy, dx), grid=False)

    def _seek(self):
        """Drive along the corridors to the victim's cell, then face it."""
        self._set_leds()
        if self.misses == 0 and self.victim is not None:
            spot = self._victim_spot(self.victim)     # seen again: refine the cell
            # After a marker cue only a victim still wanted may move the placement; the
            # marker's guess is replaced by where the object really is.
            if spot is not None and (self.marker_cell is None or self._wanted(self.victim)):
                cell = self.map.cell_at(*spot)
                if self.marker_cell is not None and cell != self.victim_cell:
                    STATUS.event(f"[nav] victim OBJECT seen in {cell} -- using that instead of "
                                 f"the marker's cell {self.victim_cell}")
                self.victim_cell = cell
        self._grid(self._plan_victim)

    def _plan_victim(self):
        vc = self.victim_cell
        if self.cell == vc:
            self._start_closing()                     # it's in this very cell
            return None
        h = self.map.heading_between(self.cell, vc)
        if h is not None and self.map.wall(self.cell, h) is not True:
            if self.facing != h:
                # Next door: turn to face it without moving.
                self.move_heading, self.target_cell = h, self.cell
                self.mover.move_to(*self.map.centre(self.cell), HEADING_RAD[h])
                return None
            self._start_closing()
            return None
        heading = self.map.next_heading(self.cell, [vc], True, self.facing,
                                        EXPLORE_PREFER_RIGHT)
        if heading is None:
            STATUS.event(f"[nav] no way to the victim at {vc} -- exploring on")
            self.seek_tries[vc] = SEEK_ATTEMPTS
            self._enter(SEARCH)
            return None
        self.route = self.map.path(self.cell, [vc], True, self.facing, EXPLORE_PREFER_RIGHT)
        return heading

    def _start_closing(self):
        """Facing the victim's cell from the centre of the next one. The sonar must
        agree there's something standing in it before closing in -- a wall, or
        nothing at all, means the victim was placed in the wrong cell."""
        front = self.sonar.front
        at = FRONT_WALL_AT_CENTRE_M
        if self.cell == self.victim_cell:
            ok = front is not None and front < at + CELL_M / 2
        else:
            ok = front is not None and at + VICTIM_WALL_WINDOW_M < front < at + CELL_M + 0.03
        if not ok:
            vc = self.victim_cell
            self.seek_tries[vc] = self.seek_tries.get(vc, 0) + 1
            STATUS.event(f"[nav] nothing in {vc} on the sonar "
                         f"({'no echo' if front is None else f'{front*100:.0f}cm'}) -- "
                         "exploring on")
            self._enter(SEARCH)
            return
        self.closing_heading = HEADING_RAD[self.facing]
        self._enter(CLOSING)

    def _plan_explore(self):
        """Nearest frontier, through walls we haven't seen assumed open."""
        frontier = self.map.frontier_cells()
        heading = None
        if frontier:
            heading = self.map.next_heading(self.cell, frontier, True, self.facing,
                                            EXPLORE_PREFER_RIGHT)
        if heading is None:
            STATUS.event(f"[nav] no frontier left: {len(self.map.visited)} cells visited, "
                         f"{len(self.map.blocked)} blocked -- calling the maze explored "
                         "and heading home")
            self._enter(RETURN)
            return None
        self.route = self.map.path(self.cell, frontier, True, self.facing,
                                   EXPLORE_PREFER_RIGHT)
        return heading

    def _plan_home(self):
        """Flood fill to base, through edges it has actually seen open. Falls back to
        assuming unseen walls are open if the known map doesn't connect -- e.g. after
        a victim chase left it in a cell it never mapped."""
        if self.cell == self.base_cell:
            self.route = None
            if self.carrying:
                self._enter(AT_BASE)
            else:
                self._finish("back at base with no victim aboard -- nothing left to "
                             "collect (maze counted as explored: no frontier cells)"
                             if not self.map.frontier_cells() else
                             "back at base with no victim aboard")
            return None
        goal = [self.base_cell]
        for optimistic in (False, True):
            heading = self.map.next_heading(self.cell, goal, optimistic, self.facing,
                                            EXPLORE_PREFER_RIGHT)
            if heading is not None:
                self.route = self.map.path(self.cell, goal, optimistic, self.facing,
                                           EXPLORE_PREFER_RIGHT)
                return heading
        if self.map.blocked:
            # Last resort: something marked blocked is sealing it in. Forget the
            # blocked cells rather than give up -- a wrong one costs a detour, giving
            # up costs the rescue.
            STATUS.event("[nav] no route home -- clearing blocked cells and retrying")
            self.map.blocked.clear()
            return self._plan_home()
        self._finish(f"no route from {self.cell} to base {self.base_cell} on the map, "
                     "even assuming unseen walls are open and clearing blocked cells")
        return None

    def _approach(self, lost):
        if lost:
            # Losing sight of the victim this close is expected, not a failure: it has
            # gone under the camera. If we were lined up on it, commit and finish on
            # the sonar rather than turning away and hunting for it again.
            front = self.sonar.front
            if self.aligned and front is not None and front <= CLOSING_TRIGGER_M:
                STATUS.event(f"[nav] victim under the camera at {front*100:.0f}cm -- "
                             "closing on sonar")
                self._enter(CLOSING)
            else:
                self.victim = None
                self._enter(SEARCH)
            return

        v = self.victim
        self.leds.green(True)

        self.aligned = abs(v.bearing_deg) <= CLOSING_ALIGN_DEG
        self.update_heading_lock(v.bearing_deg)
        distance, source = self.victim_range()

        if distance is None:
            # Can't range it from here: rotate to centre it (which is what makes the
            # sonar trustworthy) but don't close on an unknown distance.
            self.drive.set_velocity(0.0, heading_correction(v.bearing_deg, pivoting=True))
            return

        if distance - STOP_DISTANCE_M <= DISTANCE_TOLERANCE_M:
            STATUS.event(f"[nav] VICTIM REACHED -- stopping at "
                         f"{distance*100:.1f} cm ({source})")
            self._enter(AT_VICTIM)
            return

        # Past HEADING_COARSE_DEG, turn on the spot: driving on while badly off
        # heading travels further off course than the turn recovers.
        #
        # Close in, pivot until properly lined up too. The camera loses the victim a
        # few cm further on, and CLOSING only takes over from an ALIGNED approach --
        # rolling forward while still correcting would drop it under the camera
        # unaligned, give up, and go round again. That's what happens when a victim
        # is first seen from the next cell, turning into a junction.
        close = distance < CLOSING_TRIGGER_M + 0.05
        if not self.heading_locked and (abs(v.bearing_deg) > HEADING_COARSE_DEG
                                        or (close and not self.aligned)):
            self.drive.set_velocity(0.0, heading_correction(v.bearing_deg, pivoting=True))
            return

        # Locked: drive dead straight and ignore the bearing entirely.
        w = 0.0 if self.heading_locked else heading_correction(v.bearing_deg)
        speed = CREEP_SPEED if distance < CREEP_RANGE_M else APPROACH_SPEED
        self.drive.set_velocity(speed, w)

    def update_heading_lock(self, bearing_deg):
        """Latch the heading once we're lined up; only let go if it really drifts."""
        if self.heading_locked:
            if abs(bearing_deg) > HEADING_UNLOCK_DEG:
                STATUS.event(f"[nav] heading unlocked -- victim drifted to "
                             f"{bearing_deg:+.1f} deg")
                self.heading_locked = False
        elif abs(bearing_deg) <= HEADING_LOCK_DEG:
            STATUS.event(f"[nav] heading LOCKED at {bearing_deg:+.1f} deg -- "
                         "driving straight from here")
            self.heading_locked = True

    def _closing(self):
        """Final run-in with the victim out of frame. Straight ahead, sonar only.

        No steering: there's nothing to steer on, and a blind correction would be a
        guess. It ran in aligned, so it drives straight and stops on range.
        """
        self.leds.green(True)
        elapsed = time.monotonic() - self.closing_since

        if elapsed > CLOSING_TIMEOUT_S:
            STATUS.event(f"[nav] blind run-in gave up after {elapsed:.1f}s -- searching")
            self.drive.stop()
            if self.victim_cell is not None:
                self.seek_tries[self.victim_cell] = self.seek_tries.get(self.victim_cell, 0) + 1
            self._enter(SEARCH)
            return

        # Close in, a victim a few cm off the centreline drops out of the sonar's
        # cone, and the sonar reports the wall behind it instead -- which would drive
        # straight over it. So remember the range, and when the reading suddenly jumps
        # further away, finish on the odometry: range then, minus distance since.
        front = self.sonar.front
        travelled = math.hypot(self.odo.x - self.closing_from[0],
                               self.odo.y - self.closing_from[1])
        expected = None if self.closing_range is None else self.closing_range - travelled
        if front is not None and (expected is None or front <= expected + CLOSING_JUMP_M):
            self.closing_range = front + travelled          # still on it
            distance, source = front, "sonar"
        elif expected is not None and self.ticks is not None:
            distance, source = expected, "odometry"         # lost it: dead reckon
        else:
            # No vision and no range is no information at all -- hold rather than
            # drive forward on faith.
            self.drive.stop()
            return

        if distance - STOP_DISTANCE_M <= DISTANCE_TOLERANCE_M:
            STATUS.event(f"[nav] VICTIM REACHED -- stopping at {distance*100:.1f} cm ({source})")
            self._enter(AT_VICTIM)
            return

        # Straight on, held by the encoders -- still no steering on the victim itself.
        w = 0.0
        if self.closing_heading is not None and self.ticks is not None:
            w = _clamp(HEADING_HOLD_GAIN * wrap(self.closing_heading - self.odo.theta),
                       MAX_STEER_RATE)
        self.drive.set_velocity(CREEP_SPEED, w)

    def _at_victim(self):
        """Stopped 10 cm short. The (mock) collection: stand still RESCUE_TIME_S with all
        three LEDs flashing together, then it has the victim and heads home (red on).
        Without the mock (MOCK_COLLECTION off, no --test-mode) the LEDs stay as the state
        set them (green) while the real mechanism does its work."""
        self.drive.stop()
        if not self.rescue.done():
            if MOCK_COLLECTION or TEST_MODE:
                # The mock "collecting": all three LEDs flash together, in step.
                if self._mock_announced != self.arrived_at:
                    self._mock_announced = self.arrived_at
                    STATUS.event(f"[nav] {'TEST MODE: ' if TEST_MODE else ''}at the victim -- "
                                 f"mock collection: flashing all LEDs for {RESCUE_TIME_S:g}s, "
                                 "then heading home")
                on = int(time.monotonic() * COLLECT_FLASH_HZ * 2) % 2 == 0
                self.leds.green(on)
                self.leds.yellow(on)
                self.leds.red(on)
            return
        # Remember where it was, for the map: just ahead of where we stopped.
        reach = STOP_DISTANCE_M + CELL_M / 2 - FRONT_WALL_AT_CENTRE_M
        self.victim_points.append((self.odo.x + reach * math.cos(self.odo.theta),
                                   self.odo.y + reach * math.sin(self.odo.theta)))
        self.carrying = True
        if self.victim_cell is not None:
            self.map.blocked.discard(self.victim_cell)   # it's on board now
            # It closed in on the victim through this edge, so it's open -- even if
            # the victim standing just beyond it once read as a wall.
            if self.map.heading_between(self.cell, self.victim_cell) == self.facing:
                self.map.set_wall(self.cell, self.facing, False)
        for done in (self.victim_cell, self.marker_cell):
            if done is not None:
                self.done_cells.add(done)                # its wall marker is still up: ignore it
        self.victim_cell = self.marker_cell = None
        STATUS.event(f"[nav] victim collected after "
                     f"{time.monotonic() - self.arrived_at:.1f}s -- returning to base")
        if RECHECK_AFTER_PICKUP:
            self.rechecking = True               # find out where it really is before moving
            self.recheck_from = time.monotonic()
        self._enter(RETURN)

    def _return(self):
        self._set_leds()
        if self.rechecking and self._recheck_position():
            return                               # still waiting for readings: don't move yet
        self._grid(self._plan_home)

    def _recheck_position(self):
        """Stand still after a pickup and re-anchor the odometry to the walls, before the
        rejoin drives anywhere. True while it is still waiting for fresh readings."""
        self.drive.stop()
        now = time.monotonic()
        fitted = [n for n in ("front", "left", "right")
                  if n in getattr(self.sonar, "sensors", {})]
        reads = {}
        for n in fitted:
            fresh, d = self.sonar.since(n, self.recheck_from)
            if fresh:
                reads[n] = d                     # d is None for "no echo"
        if len(reads) < len(fitted) and now - self.recheck_from < RECHECK_WAIT_S:
            return True
        self.rechecking = False
        self._fix_position(reads)
        return False

    def _fix_position(self, reads):
        """Pull the odometry onto what the sonars say about the walls round it.

        Ahead: a wall is on a cell edge, so the front reading puts the robot a known
        distance from the nearest edge -- that fixes the position along the way it faces.
        Sideways: the side walls give the offset from the corridor centre. Each is used
        only if it agrees with the odometry to within a plausible error; a reading that
        fits no wall (the victim, the collection mechanism) is left alone.
        """
        o = self.odo
        facing = nearest_heading(o.theta)
        skew = wrap(o.theta - HEADING_RAD[facing])
        cell = self.map.cell_at(o.x, o.y)
        if abs(skew) > math.radians(RECHECK_MAX_SKEW_DEG):
            STATUS.event(f"[nav] position recheck: pointing {math.degrees(skew):+.0f} deg off a "
                         "grid direction, too skewed for the sonars to say where it is -- "
                         "rejoining on the odometry")
            return
        c = math.cos(skew)
        ux, uy = round(math.cos(HEADING_RAD[facing])), round(math.sin(HEADING_RAD[facing]))
        rx, ry = uy, -ux                          # unit vector to the robot's right
        notes = []

        front = reads.get("front")
        if front is not None and front <= RECHECK_FRONT_MAX_M:
            victim_ahead = ((MOCK_COLLECTION or TEST_MODE)
                            and abs(front - STOP_DISTANCE_M) <= 0.025)
            if victim_ahead:
                notes.append(f"front {front*100:.0f}cm is the (mock) victim still in front -- "
                             "not used")
            else:
                reach = (front + CELL_M / 2.0 - FRONT_WALL_AT_CENTRE_M) * c   # centre -> wall
                sign = ux + uy                    # +1 facing E/N, -1 facing W/S
                est = o.x * ux + o.y * uy + reach            # wall's coordinate, heading frame
                edge = sign * round(sign * est / CELL_M) * CELL_M
                delta = edge - est
                if abs(delta) <= RECHECK_FRONT_SNAP_M:
                    o.x += delta * ux
                    o.y += delta * uy
                    self._anchor_path = o.path_m
                    notes.append(f"{delta*100:+.0f}cm along (wall {front*100:.0f}cm ahead)")
                else:
                    notes.append(f"front {front*100:.0f}cm matches no cell edge "
                                 f"({abs(delta)*100:.0f}cm out) -- not used")

        near = SIDE_WALL_AT_CENTRE_M + WALL_PRESENT_WINDOW_M
        sides = {n: reads[n] * c for n in ("left", "right")
                 if reads.get(n) is not None and reads[n] * c < near}
        offset = Mover._offset_from_reads(sides)
        if offset is not None:
            cx, cy = self.map.centre(cell)
            delta = offset - ((o.x - cx) * rx + (o.y - cy) * ry)
            if abs(delta) <= RECHECK_SIDE_MAX_M:
                o.x += delta * rx
                o.y += delta * ry
                notes.append(f"{delta*100:+.0f}cm across (walls say {offset*100:+.0f}cm off centre)")
            else:
                notes.append(f"side walls say {offset*100:+.0f}cm off centre, "
                             f"{abs(delta)*100:.0f}cm from the odometry -- not used")
        STATUS.event(f"[nav] position rechecked after pickup in {cell}: "
                     + (", ".join(notes) if notes else "no wall in range to check against"))

    def _at_base(self):
        """Home with a victim. Release it, then go back out for the next one."""
        self.drive.stop()
        if not self.release.done():
            return
        self.carrying = False
        self.rescued += 1
        STATUS.event(f"[nav] VICTIM RESCUED -- {self.rescued} home")
        target = TEST_MODE_VICTIMS if TEST_MODE else VICTIMS_TOTAL
        if self.rescued >= target:
            self._finish(f"{'TEST MODE: ' if TEST_MODE else ''}all {target} "
                         f"victim{'s' if target != 1 else ''} rescued")
        else:
            self._enter(SEARCH)

    def _done(self):
        """Terminal. Motors off, LEDs off, nothing else changes.

        Deliberately has no way out: the run is over and the robot should sit still
        rather than wander off. Restart main.py to go again.
        """
        self.drive.stop()
        self.leds.green(False)
        self.leds.yellow(False)
        self.leds.red(False)


def main():
    global CENTRING, TEST_MODE, OBSTACLE_AVOID
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--no-motors", action="store_true", help="run without driving")
    ap.add_argument("--no-sonar", action="store_true", help="run without ultrasonics")
    ap.add_argument("--sonar-test", action="store_true",
                    help="print all three ranges continuously and exit on Ctrl-C")
    ap.add_argument("--placeholder-vision", action="store_true",
                    help="stand-in HSV detector instead of the real one")
    ap.add_argument("--no-display", action="store_true",
                    help="don't open the camera window (headless / over SSH)")
    ap.add_argument("--blur-threshold", type=float, default=None,
                    help=f"override the blur guard (default {BLUR_THRESHOLD:g} = off). "
                         "Frames scoring below it are discarded whole")
    ap.add_argument("--vision-debug", action="store_true",
                    help="print a line per frame: blur, blobs by class, victims kept")
    ap.add_argument("--no-status", action="store_true",
                    help="don't print the live status line")
    ap.add_argument("--check", action="store_true", help="wire up, report, exit")
    ap.add_argument("--no-camera", action="store_true",
                    help="no vision at all -- explore and map the maze only")
    ap.add_argument("--no-map", action="store_true", help="don't open the map window")
    ap.add_argument("--camera-walls", action="store_true",
                    help="also use the camera to see walls (off by default)")
    ap.add_argument("--no-camera-walls", action="store_true",
                    help="(the default) walls from the sonars alone")
    ap.add_argument("--test-mode", action="store_true",
                    help="demo run: find ONE victim, flash all LEDs for 5 s (mock collection), "
                         "return home and finish")
    ap.add_argument("--no-centring", action="store_true",
                    help="don't steer back to the corridor centre -- encoder heading hold only")
    ap.add_argument("--no-avoid", action="store_true",
                    help="don't sidestep obstacles the camera sees -- a blocked cell is "
                         "detoured round instead")
    ap.add_argument("--base-cell", default=None,
                    help=f"base cell as col,row (default {BASE_CELL[0]},{BASE_CELL[1]})")
    ap.add_argument("--start-heading", choices=["N", "E", "S", "W"], default=None,
                    help=f"which way it faces at the start (default {START_HEADING})")
    ap.add_argument("--calibrate", choices=["straight", "turn", "sonar"], default=None,
                    help="measure TICKS_PER_M, EFFECTIVE_TRACK_M or the sonar geometry")
    args = ap.parse_args()

    # Bring-up aid: no camera, no motors, just the sensors. Use it to check wiring
    # and that each sensor is where you think it is -- wave a hand in front of one.
    if args.sonar_test:
        sonar = Ultrasonics()
        if not sonar.sensors:
            raise SystemExit("no ultrasonics came up -- check ULTRASONIC_PINS wiring")
        print("Ctrl-C to stop. Wave a hand in front of each sensor in turn.")
        try:
            while True:
                sonar.update()
                cells = []
                for name in ("left", "front", "right"):
                    d = sonar.get(name)
                    cells.append(f"{name}: {'--- ' if d is None else f'{d*100:5.1f}'}cm")
                print("  ".join(cells), end="\r", flush=True)
                time.sleep(1.0 / CONTROL_HZ)
        except KeyboardInterrupt:
            print("\nstopped")
        return

    if args.calibrate:
        calibrate(args.calibrate)
        return

    base_cell = None
    if args.base_cell:
        try:
            base_cell = tuple(int(v) for v in args.base_cell.split(","))
            assert len(base_cell) == 2
        except (ValueError, AssertionError):
            raise SystemExit("--base-cell wants col,row -- e.g. 0,6")

    if args.test_mode:
        TEST_MODE = True
        STATUS.event(f"[nav] --test-mode: collect {TEST_MODE_VICTIMS} victim (all LEDs flash for "
                     f"{RESCUE_TIME_S:g}s), return home, then finish")
    if args.no_centring:
        CENTRING = False
        STATUS.event("[nav] --no-centring: encoder heading hold only")
    if args.no_avoid:
        OBSTACLE_AVOID = False
        STATUS.event("[nav] --no-avoid: obstacles are not sidestepped")
    STATUS.enabled = not args.no_status
    leds = Leds()
    drive = Drive(enabled=not args.no_motors)
    sonar = Ultrasonics(enabled=not args.no_sonar)
    blur = BLUR_THRESHOLD if args.blur_threshold is None else args.blur_threshold
    if args.no_camera:
        vision = NullVision()
        STATUS.event("[vision] --no-camera: exploring and mapping only")
    else:
        blur_managed = False
        if not args.placeholder_vision:
            import vision_system_v2_0 as _vs
            blur_name = blur_setting_name(_vs)
            if blur_name is None:
                STATUS.event("[vision] no blur guard setting found in vision_system -- "
                             "leaving the vision team's behaviour as it is")
            elif blur_name == "BLUR_VARIANCE_THRESHOLD" or args.blur_threshold is not None:
                old = getattr(_vs, blur_name)
                if old != blur:
                    STATUS.event(f"[vision] {blur_name} {old:g} -> "
                                 f"{blur:g}{' (off)' if blur <= 0 else ''}")
                setattr(_vs, blur_name, blur)
                blur_managed = True
            else:
                STATUS.event(f"[vision] {blur_name} = {getattr(_vs, blur_name):g} (the vision "
                             "team's value, left as is; --blur-threshold overrides it)")
        vision = VictimVision(use_placeholder=args.placeholder_vision)
        if blur_managed or args.placeholder_vision:
            vision.blur_threshold = blur
    wallcam = None
    if ((USE_CAMERA_WALLS or args.camera_walls) and not args.no_camera
            and not args.no_camera_walls and not args.placeholder_vision):
        wallcam = make_wall_camera()
    nav = Nav(drive, vision, leds, sonar, base_cell=base_cell,
              start_heading=args.start_heading, wall_camera=wallcam)
    display = Display(enabled=not args.no_display, camera=not args.no_camera,
                      show_map=not args.no_map)
    STATUS.event(f"[nav] {MAZE_COLS}x{MAZE_ROWS} maze of {CELL_M*1000:.0f}mm cells, "
                 f"base {nav.base_cell} facing {HEADING_NAME[nav.facing]}")

    ranging = ("front sonar" if sonar.available else
               "camera only" if vision.geometry_ok else "NONE -- approach disabled")
    STATUS.event(f"[status] motors={'yes' if drive.board else 'NO'}  "
          f"leds={'GPIO' if leds.real else 'console'}  "
                 f"sonar={len(sonar.sensors)}/3  "
                 f"ranging={ranging}")

    if args.check:
        for _ in range(12):                    # a few ticks so each sensor reports
            sonar.update()
            time.sleep(US_MIN_TRIGGER_GAP_S)
        for name in ("left", "front", "right"):
            d = sonar.get(name)
            print(f"  sonar {name:>5}: {'no reading' if d is None else f'{d*100:.1f} cm'}")
        drive.stop()
        leds.all_off()
        vision.close()
        display.close()
        return

    period = 1.0 / CONTROL_HZ
    hz = float(CONTROL_HZ)
    try:
        while True:
            t0 = time.monotonic()
            nav.step()

            # The window draws the frame nav just decided on, so what you see is
            # exactly what it acted on -- not a fresh capture that may differ.
            if not display.show(vision.last_frame, vision.last_detections,
                                nav, sonar, vision.draw_detections):
                STATUS.event("[display] quit requested")
                break

            if args.vision_debug:
                STATUS.event(f"[vision] {vision_summary(vision)}")
            STATUS.update(status_text(nav, sonar, leds, hz))
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
            # measured, not the target -- if vision is slow this is where it shows
            hz = 0.8 * hz + 0.2 / max(time.monotonic() - t0, 1e-6)
    except KeyboardInterrupt:
        STATUS.event("\ninterrupted")
    finally:
        STATUS.close()
        drive.stop()
        leds.all_off()
        vision.close()
        display.close()
        save_map(nav)
        print("stopped")


def make_wall_camera():
    """The camera wall detector, or None if it can't be set up (no numpy/OpenCV, or
    the camera geometry isn't measured) -- the robot then runs on the sonars alone."""
    try:
        from camera_capture_v2_0 import (CAMERA_HEIGHT_M, CAMERA_TILT_DEG,
                                         HORIZONTAL_FOV_DEG, VERTICAL_FOV_DEG)
        from wallvision import WallCamera
    except Exception as exc:                           # noqa: BLE001
        STATUS.event(f"[walls] camera walls off -- {type(exc).__name__}: {exc}")
        return None
    if None in (CAMERA_HEIGHT_M, CAMERA_TILT_DEG, HORIZONTAL_FOV_DEG, VERTICAL_FOV_DEG):
        STATUS.event("[walls] camera walls off -- camera geometry not set in "
                     "camera_capture_v2_0.py")
        return None
    STATUS.event("[walls] camera + sonar wall sensing")
    return WallCamera(CAMERA_HEIGHT_M, CAMERA_TILT_DEG, HORIZONTAL_FOV_DEG, VERTICAL_FOV_DEG)


def save_map(nav):
    """Write the final map to Robot_Code/runs/ -- handy for the report."""
    try:
        import cv2
        from map_view import render_map
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("map_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, render_map(nav))
        print(f"[map] saved {path}")
    except Exception as exc:                           # noqa: BLE001
        print(f"[map] not saved -- {type(exc).__name__}: {exc}")


def calibrate(what):
    """Measure the numbers the maze nav depends on. Robot on the floor, in the maze."""
    period = 1.0 / CONTROL_HZ

    if what == "sonar":
        sonar = Ultrasonics()
        print("Put the robot dead centre in a cell with walls ahead, left and right.")
        input("Enter to start... ")
        samples = {n: [] for n in ("front", "left", "right")}
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            sonar.update()
            for n in samples:
                d = sonar.get(n)
                if d is not None:
                    samples[n].append(d)
            time.sleep(period)
        med = {n: (sorted(v)[len(v) // 2] if v else None) for n, v in samples.items()}
        for n, d in med.items():
            print(f"  {n:>5}: {'no reading' if d is None else f'{d*100:.1f} cm'}")
        if med["front"] is not None:
            print(f"\nFRONT_WALL_AT_CENTRE_M = {med['front']:.3f}")
        sides = [d for d in (med["left"], med["right"]) if d is not None]
        if sides:
            print(f"SIDE_WALL_AT_CENTRE_M = {sum(sides) / len(sides):.3f}")
        return

    drive = Drive()
    if drive.driver is None:
        raise SystemExit("no motors -- can't calibrate driving")
    odo = Odometry(TICKS_PER_M, EFFECTIVE_TRACK_M)
    odo.update(drive.read_encoders())
    start = drive.read_encoders()
    try:
        if what == "straight":
            print("Drives straight for 2.5 s. Mark where the robot starts.")
            input("Enter to start... ")
            t_end = time.monotonic() + 2.5
            while time.monotonic() < t_end:
                odo.update(drive.read_encoders())
                w = _clamp(HEADING_HOLD_GAIN * wrap(0.0 - odo.theta), MAX_STEER_RATE)
                drive.set_velocity(SEARCH_SPEED, w)
                time.sleep(period)
            drive.stop()
            time.sleep(0.3)
            end = drive.read_encoders()
            dl, dr = end[0] - start[0], end[1] - start[1]
            print(f"  left {dl} ticks, right {dr} ticks")
            if dl * dr < 0:
                print("\n  !! The two counts have OPPOSITE signs, so driving forward made one "
                      "encoder count down.\n     Flip that side's ENCODER_LEFT_SIGN / "
                      "ENCODER_RIGHT_SIGN in motors.py and run this again --\n     "
                      "otherwise the odometry thinks the robot is spinning, not driving.")
            ticks = (abs(dl) + abs(dr)) / 2.0
            cm = float(input("How far did it actually go, in cm? "))
            print(f"\nTICKS_PER_M = {ticks / (cm / 100.0):.0f}")
        else:
            print("Spins one full turn to the LEFT, by the encoders' reckoning. "
                  "Note which way it faces first.")
            input("Enter to start... ")
            turned = 0.0
            last = odo.theta
            t_end = time.monotonic() + 15.0
            while turned < 2 * math.pi and time.monotonic() < t_end:
                odo.update(drive.read_encoders())
                turned += wrap(odo.theta - last)
                last = odo.theta
                drive.set_velocity(0.0, MIN_TURN_RATE_PIVOT)
                time.sleep(period)
            drive.stop()
            print("  the encoders say that was 360 deg")
            deg = float(input("How many degrees did it ACTUALLY turn? (e.g. 330, 380) "))
            print(f"\nEFFECTIVE_TRACK_M = {EFFECTIVE_TRACK_M * 360.0 / deg:.4f}")
    finally:
        drive.stop()
        drive.close()


if __name__ == "__main__":
    main()
