#!/usr/bin/env python3
# -*-coding:utf8-*-
# Full station camera calibration in one run: the top camera from the breadboard holes
# (top_camera_calibration.py), then the wrist camera from one ArUco tag lying on the board
# (wrist_camera_calibration.py run), with everything brought up by piper_ros
# start_station_calibration.launch.py.
#
#   ros2 launch piper start_station_calibration.launch.py
#   python3 multi_camera_calibration.py                   # top, then wrist
#   python3 multi_camera_calibration.py --cameras wrist   # wrist only, with the saved top_calibration.json
#
# 1. Top: the board must be clear of everything (tag included). The window shows the hole fit;
#    once the hole count and jitter settle, press e. A fit over --max-extrinsic-rms is refused
#    (r resets the stats, then e again). q aborts.
# 2. Wrist: place the tag on the board, press Enter in the terminal. Default (--wrist-mode auto):
#    the arm visits planned views of the tag by itself until the fit converges (wrist_camera_auto.py;
#    asks before the first move; q in the window stops). --wrist-mode manual: teach the views in
#    teaching mode as in wrist_camera_calibration.py run (Enter capture, d undo, t re-measure the
#    tag, q fit + finish).
#
# Results: <static-dir>/top_calibration.json and wrist_calibration.json, with the time they were
# made inside ('timestamp'). Existing ones are only replaced after a prompt at the start. The
# timestamped copies and samples still go under --data-save-path as before ('history_file').
#
# The station launch puts every node under /station<N> on ROS_DOMAIN_ID N, N from the hostname
# (robotics-education-lab<N>); this script does the same, so --station-id is only needed off-station.

import argparse
import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import json
import os
import re
import socket
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from top_camera_calibration import BreadboardHoleTest, add_board_args, to_bgr
from wrist_camera_calibration import add_run_args, cmd_run, load_top_calibration, pose_to_T
import wrist_camera_auto

STATIC_FILES = {'top': 'top_calibration.json', 'wrist': 'wrist_calibration.json'}


def hostname_station():
    # Same rule as start_station_calibration.launch.py: robotics-education-lab<N>, else 0
    m = re.search(r'robotics-education-lab(\d+)(?:\..*)?$', socket.gethostname())
    return int(m.group(1)) if m else 0


def now_iso():
    try:
        tz = ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        tz = datetime.timezone(datetime.timedelta(hours=-5))
    return datetime.datetime.now(tz).isoformat(timespec='seconds')


def use_station(args):
    """Topic defaults under /station<N>, and the launch file's ROS_DOMAIN_ID (set before rclpy.init)."""
    ns = f"/station{args.station_id}"
    defaults = {'top_topic': f"{ns}/intel_realsense_d435i_top/color/image_raw",
                'top_info': f"{ns}/intel_realsense_d435i_top/color/camera_info",
                'wrist_topic': f"{ns}/intel_realsense_d435i_wrist/color/image_raw",
                'wrist_info': f"{ns}/intel_realsense_d435i_wrist/color/camera_info",
                'joint_topic': f"{ns}/joint_states_single"}
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.station_id:
        cur = os.environ.get('ROS_DOMAIN_ID')
        if cur not in (None, str(args.station_id)):
            print(f"NOTE: ROS_DOMAIN_ID was {cur}; using {args.station_id} like the station launch")
        os.environ['ROS_DOMAIN_ID'] = str(args.station_id)


def static_path(args, cam):
    return os.path.join(args.static_dir, STATIC_FILES[cam])


def confirm_overwrite(args, cams):
    """Ask once, before anything runs, whether existing calibration files may be replaced."""
    if not os.path.isdir(args.static_dir):
        raise SystemExit(f"{args.static_dir} does not exist (--static-dir)")
    found = []
    for cam in cams:
        p = static_path(args, cam)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    made = json.load(f).get('timestamp', 'no timestamp')
            except (OSError, ValueError):
                made = 'unreadable'
            found.append(f"  {p}  ({made})")
    if not found:
        return
    print("existing calibration files:\n" + "\n".join(found))
    while True:
        try:
            a = input("[o]verwrite or [c]ancel? ").strip().lower()
        except EOFError:
            a = 'c'
        if a in ('o', 'overwrite'):
            return
        if a in ('c', 'cancel'):
            raise SystemExit("cancelled, nothing changed")


def write_static(args, cam, result):
    out = dict(result)
    history = out.pop('file', None)
    out = {'timestamp': now_iso(), 'camera': cam, 'station_id': args.station_id, **out,
           'history_file': os.path.abspath(history) if history else None}
    p = static_path(args, cam)
    # Write then rename, so a crash never leaves a half-written calibration behind
    with open(p + '.tmp', 'w') as f:
        json.dump(out, f, indent=2)
    os.replace(p + '.tmp', p)
    print(f"wrote {p}")
    return p


def calibrate_top(args):
    """Live hole fit on the top camera until e gives an extrinsic good enough to save; None on q."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo

    app = BreadboardHoleTest(args)
    rclpy.init()
    node = Node("multi_camera_calibration_top")
    latest = {}
    node.create_subscription(Image, args.top_topic, lambda m: latest.__setitem__('img', m), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, args.top_info, lambda m: app.set_camera_info(
        np.array(m.k).reshape(3, 3), np.array(m.d)), qos_profile_sensor_data)
    print(f"\n=== top camera ({args.top_topic})\nboard clear of everything; when holes and jitter "
          f"settle, press e in the window (r reset, q abort)")
    vis, result = None, None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            msg = latest.pop('img', None)               # newest frame only
            if msg is not None:
                vis = app.process(to_bgr(msg))
            if vis is not None:
                cv2.imshow("top camera calibration", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 255:
                continue
            if k == ord('e'):
                result = app.extrinsic()
                if result is not None:
                    break
            elif not app.key(k, vis):
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        cv2.waitKey(1)                                  # lets the window actually close
    return result


def main():
    # 'resolve': both steps define --board-z and --data-save-path with the same meaning
    ap = argparse.ArgumentParser(description=__doc__, conflict_handler='resolve',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cameras', choices=['both', 'top', 'wrist'], default='both')
    ap.add_argument('--static-dir', default='/armlab_ws/static_data',
                    help="where top_calibration.json / wrist_calibration.json go")
    ap.add_argument('--station-id', type=int, default=hostname_station(),
                    help="station number (default from the hostname); topics under /station<N>, ROS_DOMAIN_ID N")
    add_board_args(ap)
    ap.add_argument('--save-path', default='./data/breadboard', help="hole snapshots (s key)")
    add_run_args(ap)
    ap.add_argument('--wrist-mode', choices=['auto', 'manual'], default='auto',
                    help="auto: the arm moves through planned views; manual: teach them by hand")
    wrist_camera_auto.add_auto_args(ap)
    # Topics default to the station launch's, filled in from --station-id
    ap.set_defaults(top_topic=None, top_info=None, wrist_topic=None, wrist_info=None, joint_topic=None)
    args = ap.parse_args()
    use_station(args)
    cams = ['top', 'wrist'] if args.cameras == 'both' else [args.cameras]
    print(f"station {args.station_id} (ROS_DOMAIN_ID {os.environ.get('ROS_DOMAIN_ID', 'unset')}), "
          f"calibrating: {', '.join(cams)}")
    confirm_overwrite(args, cams)
    if 'wrist' in cams and args.wrist_mode == 'auto':
        # Read the planning mount now: wrist_calibration.json may be replaced by this run
        guess = args.mount_guess or (static_path(args, 'wrist') if os.path.exists(static_path(args, 'wrist'))
                                     else wrist_camera_auto.NOMINAL_MOUNT)
        X_guess = wrist_camera_auto.load_mount(guess)
        print(f"planning mount guess: {guess}")

    if 'top' in cams:
        top = calibrate_top(args)
        if top is None:
            raise SystemExit("top camera not calibrated, nothing written")
        args.top_calibration = write_static(args, 'top', top)
        T_base_top = pose_to_T(top['position'], top['orientation'])
    else:
        args.top_calibration = static_path(args, 'top')
        if not os.path.exists(args.top_calibration):
            raise SystemExit(f"{args.top_calibration} missing: the wrist step needs the top camera calibrated first")
        T_base_top = load_top_calibration(args.top_calibration)

    if 'wrist' in cams:
        print(f"\n=== wrist camera ({args.wrist_topic})")
        try:
            input(f"place the tag (id {args.marker_id}, {args.marker_size:g} mm) flat on the board where the "
                  f"top camera sees it, then press Enter... ")
        except EOFError:
            raise SystemExit("no terminal input; wrist camera not calibrated")
        if args.wrist_mode == 'auto':
            rig = wrist_camera_auto.RosRig(args, T_base_top)
            wrist = wrist_camera_auto.run_auto(args, rig, X_guess, log=rig.log)
        else:
            wrist = cmd_run(args, T_base_top)
        if wrist is None:
            raise SystemExit("not enough wrist views, wrist_calibration.json not written")
        if not wrist.get('auto', {}).get('converged', True):
            raise SystemExit(f"automated wrist calibration ended '{wrist['auto']['end']}' without converging, "
                             f"wrist_calibration.json not written (result kept at {wrist['file']}); "
                             f"--target-loo-mm or --max-views to loosen")
        if wrist['joint_offsets_at_bound']:
            raise SystemExit(f"joint offsets {wrist['joint_offsets_at_bound']} hit the bound, wrist_calibration.json "
                             f"not written (result kept at {wrist['file']})")
        write_static(args, 'wrist', wrist)


if __name__ == '__main__':
    main()
