#!/usr/bin/env python3
# -*-coding:utf8-*-
# Eye-in-hand (wrist camera) calibration from one ArUco tag lying on the breadboard, with the
# tag's pose in the base measured by the calibrated top camera, so the tag needs no precise
# placement. multi_camera_calibration.py runs `run` as its second step, after the top camera.
# `run` does it all with both cameras live; the older one-camera-at-a-time steps are:
#
#   1. tag     Top camera bringup (mode:=eye_to_hand). Averages /aruco_single/pose and maps it to
#              the base with the top-camera calibration (top_camera_calibration.py extrinsic). The
#              tag lies flat on the board, so only its x, y and yaw come from the camera: its
#              position is where the viewing ray to the tag center meets the board plane (no
#              dependence on the printed tag size), its normal is the board normal.
#              -> data/eye_in_hand/<ts>_tag_in_base.json
#   2. record  Wrist camera bringup (mode:=eye_in_hand), arm in teaching mode, tag NOT moved.
#              Enter captures a view (averaged tag pose + joints); each view's implied camera
#              mount is printed against the earlier views right away, so a bad view shows at once.
#              -> data/eye_in_hand/<ts>_tag_samples.json (rewritten after every capture)
#   3. fit     Offline. Camera mount (flange -> camera) from all views, plus the j1..j5 zero
#              offsets once there are ~8+ views with varied arm configurations (--offsets).
#              -> data/eye_in_hand/<ts>_tag_calibration.json (same layout as the other results)
#
#   python3 wrist_camera_calibration.py tag --top-calibration data/eye_to_hand/<ts>_breadboard_calibration.json
#   python3 wrist_camera_calibration.py record --tag data/eye_in_hand/<ts>_tag_in_base.json
#   python3 wrist_camera_calibration.py fit data/eye_in_hand/<ts>_tag_samples.json [--offsets on|off]
#
# The marker size must be the printed tag's (black square side): ArUco distances scale with it,
# and a 2% error costs ~7 mm. The top camera also estimates the size (ArUco distance vs the
# board-plane distance to the same tag) and prints it as a check, but with a 100 mm tag at ~1 m
# that estimate read 2-3% high on the real setup, so it is only applied with --size-correction
# (for a tag whose size you can't measure).

import argparse
import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import json
import os
import sys
import threading
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from breadboard_model import fk, inv_T

SIGMA_T, SIGMA_R = 1.0, np.radians(1.0)     # mm, rad: weights of a view's tag pose in the fit


def timestamp():
    try:
        now = datetime.datetime.now(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5)))
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def pose_to_T(position_m, quat_xyzw):
    return to_T(Rotation.from_quat(quat_xyzw).as_matrix(), np.asarray(position_m) * 1000.0)


def average_T(Ts):
    """Mean of rigid transforms + spread (mm, deg) of the samples around it."""
    Ts = np.asarray(Ts)
    R = Rotation.from_matrix(Ts[:, :3, :3])
    Rm = R.mean()
    t = Ts[:, :3, 3].mean(0)
    dt = np.linalg.norm(Ts[:, :3, 3] - t, axis=1)
    dr = np.degrees((Rm.inv() * R).magnitude())
    return to_T(Rm.as_matrix(), t), dt, dr


def tag_in_base(T_cam_tags, T_base_cam, board_z, thickness=0.0):
    """Tag pose in the base from top-camera tag poses (camera <- tag, mm). Position: the ray from
    the camera through the measured tag center, intersected with the board plane (the ray
    direction doesn't depend on the marker size; the distance along it does). Orientation: the
    measured one, with the tag axis closest to vertical snapped onto the board normal."""
    T_ct, dt, dr = average_T(T_cam_tags)
    o = T_base_cam[:3, 3]
    d = T_base_cam[:3, :3] @ (T_ct[:3, 3] / np.linalg.norm(T_ct[:3, 3]))
    z = board_z + thickness
    s = (z - o[2]) / d[2]
    p = o + s * d
    raw = (T_base_cam @ T_ct)[:3, 3]
    size_ratio = np.linalg.norm(raw - o) / np.linalg.norm(p - o)   # >1: marker_size set too big
    R = T_base_cam[:3, :3] @ T_ct[:3, :3]
    k = int(np.argmax(np.abs(R[2, :])))                             # tag axis closest to vertical
    a = R[:, k]
    target = np.array([0.0, 0.0, np.sign(a[2])])
    # Smallest rotation taking that axis onto vertical; keeps the tag's own axis convention
    v = np.cross(a, target)
    snap = Rotation.from_rotvec(v / max(np.linalg.norm(v), 1e-12) * np.arcsin(min(np.linalg.norm(v), 1.0)))
    R_flat = snap.as_matrix() @ R
    tilt = np.degrees(np.arccos(min(abs(a[2]), 1.0)))
    return to_T(R_flat, p), dict(raw_position_mm=raw.tolist(), size_ratio=float(size_ratio),
                                 tilt_removed_deg=float(tilt), spread_mm=float(dt.max()),
                                 spread_deg=float(dr.max()))


def mount(q, T_cam_tag, T_base_tag, delta=None):
    """Flange -> camera implied by one view: X = FK(q + delta)^-1 * T_base_tag * T_cam_tag^-1."""
    qq = np.asarray(q) + (0 if delta is None else delta)
    return inv_T(fk(qq[None])[0]) @ T_base_tag @ inv_T(T_cam_tag)


def pose_residual(T_pred, T_meas):
    dt = (T_pred[:3, 3] - T_meas[:3, 3]) / SIGMA_T
    dr = Rotation.from_matrix(T_meas[:3, :3].T @ T_pred[:3, :3]).as_rotvec() / SIGMA_R
    return np.r_[dt, dr]


def fit_views(q, T_cam_tags, T_base_tag, joints=(), max_offset_deg=10.0):
    """Camera mount (+ optional joint offsets) from tag views. Without offsets the mount is the
    average of the per-view mounts; with offsets, least squares on the predicted tag pose in the
    camera. Returns (X, delta)."""
    X0, _, _ = average_T([mount(qk, Tk, T_base_tag) for qk, Tk in zip(q, T_cam_tags)])
    if not joints:
        return X0, np.zeros(6)
    joints = list(joints)
    lim = np.radians(max_offset_deg)

    def unpack(p):
        delta = np.zeros(6)
        delta[joints] = p[:len(joints)]
        X = to_T(Rotation.from_rotvec(p[len(joints):len(joints) + 3]).as_matrix(), p[len(joints) + 3:])
        return X, delta

    def res(p):
        X, delta = unpack(p)
        out = []
        for qk, Tk in zip(q, T_cam_tags):
            T_pred = inv_T(fk((qk + delta)[None])[0] @ X) @ T_base_tag     # camera <- tag
            out.append(pose_residual(T_pred, Tk))
        return np.concatenate(out)

    p0 = np.r_[np.zeros(len(joints)), Rotation.from_matrix(X0[:3, :3]).as_rotvec(), X0[:3, 3]]
    lo = np.r_[np.full(len(joints), -lim), np.full(6, -np.inf)]
    hi = np.r_[np.full(len(joints), lim), np.full(6, np.inf)]
    sol = least_squares(res, p0, bounds=(lo, hi), x_scale='jac', loss='soft_l1', f_scale=2.0)
    return unpack(sol.x)


def view_errors(X, delta, q, T_cam_tags, T_base_tag):
    """Per view: tag position error (mm) and rotation error (deg) of the prediction."""
    out = []
    for qk, Tk in zip(q, T_cam_tags):
        T_pred = inv_T(fk((qk + delta)[None])[0] @ X) @ T_base_tag
        out.append((np.linalg.norm(T_pred[:3, 3] - Tk[:3, 3]),
                    np.degrees(np.linalg.norm(Rotation.from_matrix(Tk[:3, :3].T @ T_pred[:3, :3]).as_rotvec()))))
    return np.array(out)


# ---------------------------------------------------------------- ROS steps (tag, record)

def ros_collect(node, sub_state, count, timeout, rclpy):
    """Spin until `count` new tag poses arrived (or timeout); returns them and the joint
    messages received meanwhile."""
    sub_state['tags'], sub_state['joints'] = [], []
    start = time.time()
    while len(sub_state['tags']) < count and time.time() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
    return list(sub_state['tags']), list(sub_state['joints'])


def make_node(args, with_joints):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    rclpy.init()
    node = Node("wrist_camera_calibration")
    state = {'tags': [], 'joints': []}

    def on_tag(m):
        p, o = m.pose.position, m.pose.orientation
        state['tags'].append(pose_to_T([p.x, p.y, p.z], [o.x, o.y, o.z, o.w]))

    def on_joint(m):
        names = list(m.name)
        if all(f'joint{k}' in names for k in range(1, 7)):
            state['joints'].append([m.position[names.index(f'joint{k}')] for k in range(1, 7)])
        else:
            state['joints'].append(list(m.position[:6]))
    node.create_subscription(PoseStamped, args.marker_topic, on_tag, 10)
    if with_joints:
        node.create_subscription(JointState, args.joint_topic, on_joint, 50)
    return rclpy, node, state


def load_top_calibration(path):
    """Top camera -> base (mm) from a calibration result; a clear message for the wrong file."""
    with open(path) as f:
        c = json.load(f)
    if 'position' not in c or 'orientation' not in c:
        hint = (" It is a hole snapshot: get the calibration with\n  python3 top_camera_calibration.py "
                f"--extrinsic-from {path}" if 'holes' in c and 'H_grid_to_px' in c else "")
        raise SystemExit(f"{path} is not a camera calibration (no position/orientation); pass the "
                         f"*_breadboard_calibration.json.{hint}")
    return pose_to_T(c['position'], c['orientation'])


def cmd_tag(args):
    T_base_cam = load_top_calibration(args.top_calibration)
    rclpy, node, state = make_node(args, with_joints=False)
    print(f"collecting {args.frames} tag poses on {args.marker_topic} (top camera)...")
    tags, _ = ros_collect(node, state, args.frames, args.timeout, rclpy)
    node.destroy_node()
    rclpy.shutdown()
    if len(tags) < max(5, args.frames // 2):
        raise SystemExit(f"only {len(tags)} tag poses; is the tag visible to the top camera (and not occluded by the arm)?")
    T, info = tag_in_base(tags, T_base_cam, args.board_z, args.tag_thickness)
    print(f"tag in base: position {np.round(T[:3, 3], 2).tolist()} mm, yaw "
          f"{np.degrees(np.arctan2(T[1, 0], T[0, 0])):.2f} deg (from {len(tags)} frames, spread "
          f"{info['spread_mm']:.2f} mm / {info['spread_deg']:.2f} deg)")
    print(f"tilt removed (tag laid flat): {info['tilt_removed_deg']:.2f} deg; ArUco's own position was "
          f"{np.round(info['raw_position_mm'], 1).tolist()} mm")
    if abs(info['size_ratio'] - 1) > 0.01:
        print(f"NOTE: ArUco distance / board-plane distance = {info['size_ratio']:.3f}: marker_size in the "
              f"launch file looks {abs(info['size_ratio'] - 1) * 100:.1f}% too {'big' if info['size_ratio'] > 1 else 'small'}"
              f" (or the top calibration's height is off). Wrist-camera distances scale with it too.")
    out = {'T_base_tag_mm': T.tolist(), **info, 'frames': len(tags), 'board_z_mm': args.board_z,
           'tag_thickness_mm': args.tag_thickness, 'top_calibration': args.top_calibration,
           'marker_topic': args.marker_topic}
    path = os.path.join(args.data_save_path, 'eye_in_hand')
    os.makedirs(path, exist_ok=True)
    fn = os.path.join(path, f"{timestamp()}_tag_in_base.json")
    with open(fn, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved {fn}")


def size_corrected(T_cam_tag, size_ratio):
    # ArUco position scales with the configured marker size; size_ratio = configured / true
    T = np.array(T_cam_tag, dtype=float)
    T[:3, 3] /= size_ratio
    return T


def cmd_record(args):
    with open(args.tag) as f:
        tag = json.load(f)
    T_base_tag = np.array(tag['T_base_tag_mm'])
    size_ratio = tag.get('size_ratio', 1.0) if args.size_correction else 1.0
    print(f"tag size ratio from the top camera: {tag.get('size_ratio', 1.0):.4f} "
          f"({'applied' if args.size_correction else 'not applied; --size-correction to apply'})")
    rclpy, node, state = make_node(args, with_joints=True)
    fn = os.path.join(args.data_save_path, 'eye_in_hand', f"{timestamp()}_tag_samples.json")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    views = []

    def save():
        with open(fn, 'w') as f:
            json.dump({'tag': args.tag, 'T_base_tag_mm': T_base_tag.tolist(), 'size_ratio': size_ratio,
                       'T_cam_tag_note': 'as published by aruco_ros (not size-corrected)', 'views': views}, f, indent=2)

    keys = []
    threading.Thread(target=lambda: [keys.append(input().strip().lower()) for _ in iter(int, 1)],
                     daemon=True).start()
    print(f"saving to {fn}\nEnter: capture a view   d: undo last   q: quit   (type + Enter)")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not keys:
                state['tags'].clear()                       # only poses during a capture matter
                state['joints'].clear()
                continue
            k = keys.pop(0)
            if k == 'q':
                break
            if k == 'd':
                if views:
                    views.pop()
                    save()
                print(f"removed last view, {len(views)} left")
                continue
            tags, joints = ros_collect(node, state, args.frames, args.timeout, rclpy)
            if len(tags) < max(5, args.frames // 2):
                print(f"rejected: only {len(tags)} tag poses (tag not visible?)")
                continue
            if len(joints) < 5:
                print(f"rejected: only {len(joints)} joint messages on {args.joint_topic} during capture "
                      f"(joint states not updating?)")
                continue
            js = np.array(joints)
            motion = np.degrees(np.ptp(js, axis=0)).max()
            if motion > args.max_joint_motion:
                print(f"rejected: joints moved {motion:.3f} deg during capture")
                continue
            T_ct, dt, dr = average_T(tags)
            if dr.max() > args.max_tag_jitter:
                # ArUco's planar pose can flip between two solutions; a mixed average is garbage
                print(f"rejected: tag orientation jumps {dr.max():.1f} deg between frames (pose ambiguity; "
                      f"tilt the camera more or move closer)")
                continue
            q = js.mean(0)
            Xk = mount(q, size_corrected(T_ct, size_ratio), T_base_tag)
            views.append({'joints': q.tolist(), 'T_cam_tag_mm': T_ct.tolist(), 'tag_frames': len(tags),
                          'tag_spread_mm': float(dt.max()), 'tag_spread_deg': float(dr.max()),
                          'joint_msgs': len(joints)})
            save()
            msg = (f"view {len(views)}: mount t {np.round(Xk[:3, 3], 1).tolist()} mm, "
                   f"tag jitter {dt.max():.2f} mm / {dr.max():.2f} deg")
            if len(views) >= 2:
                prev = [mount(np.array(v['joints']), size_corrected(v['T_cam_tag_mm'], size_ratio), T_base_tag)
                        for v in views[:-1]]
                Xm, _, _ = average_T(prev)
                d_t = np.linalg.norm(Xk[:3, 3] - Xm[:3, 3])
                d_r = np.degrees(np.linalg.norm(Rotation.from_matrix(Xm[:3, :3].T @ Xk[:3, :3]).as_rotvec()))
                msg += f", vs earlier views {d_t:.1f} mm / {d_r:.2f} deg"
                if d_t > args.check_mm or d_r > args.check_deg:
                    msg += f" -> BAD? (over {args.check_mm} mm / {args.check_deg} deg), 'd' to remove"
                allX = prev + [Xk]
                _, st, sr = average_T(allX)
                msg += f"\n   all {len(views)} views: mount spread {st.mean():.1f} mm / {sr.mean():.2f} deg mean"
            print(msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print(f"{len(views)} views in {fn}")


def cmd_fit(args):
    q, Ts, T_base_tag = [], [], None
    for p in args.samples:
        with open(p) as f:
            d = json.load(f)
        Tb = np.array(d['T_base_tag_mm'])
        if args.top_calibration:
            if 'T_top_tag_mm' not in d:
                raise SystemExit(f"{p} has no raw top-camera measurement (recorded before it was saved)")
            Tb, info = tag_in_base([np.array(d['T_top_tag_mm'])], load_top_calibration(args.top_calibration),
                                   d.get('board_z_mm', 0.7), d.get('tag_thickness_mm', 0.0))
            d['size_ratio'] = d['size_ratio_measured'] = info['size_ratio']
            print(f"{p}: tag in base with {args.top_calibration}: {np.round(Tb[:3, 3], 1).tolist()} mm "
                  f"(was {np.round(np.array(d['T_base_tag_mm'])[:3, 3], 1).tolist()})")
        if T_base_tag is not None and not np.allclose(Tb, T_base_tag):
            raise SystemExit(f"{p}: different tag pose than the first file (tag moved?)")
        T_base_tag = Tb
        ratio = d.get('size_ratio_measured', d.get('size_ratio', 1.0)) if args.size_correction else 1.0
        for v in d['views']:
            q.append(v['joints'])
            Ts.append(size_corrected(v['T_cam_tag_mm'], ratio))
    fit_report(np.array(q), Ts, T_base_tag, args.offsets, args.max_offset, args.samples, args.data_save_path)


def fit_report(q, Ts, T_base_tag, offsets='auto', max_offset=10.0, samples=(), data_save_path='./data'):
    """Fit, print, and save the calibration. Returns the saved result (+ 'file': where)."""
    # Offsets: in simulation, mount-only leaves ~20+ mm with +-2 deg offsets; fitting j1..j5 from
    # ~10 varied views gets ~0.5 mm. They need enough views to be pinned down, hence 'auto'.
    use = offsets == 'on' or (offsets == 'auto' and len(q) >= 8)
    joints = [0, 1, 2, 3, 4] if use else []
    if use and len(q) < 8:
        print(f"WARNING: {len(q)} views; offsets need ~8+ views with varied arm configurations")
    if not use:
        print(f"offsets not fitted ({len(q)} views; auto fits them from 8 views on): the mount absorbs the "
              f"joint offset errors, so expect several mm away from the recorded poses")
    X, delta = fit_views(q, Ts, T_base_tag, joints, max_offset)
    err = view_errors(X, delta, q, Ts, T_base_tag)
    loo = []
    for k in range(len(q)):
        m = np.arange(len(q)) != k
        Xk, dk = fit_views(q[m], [Ts[i] for i in np.flatnonzero(m)], T_base_tag, joints, max_offset)
        loo.append(view_errors(Xk, dk, q[k:k + 1], [Ts[k]], T_base_tag)[0])
    loo = np.array(loo)
    R = Rotation.from_matrix(X[:3, :3])
    off = np.degrees(delta)
    print(f"{len(q)} views, offsets {'j1..j5 fitted' if joints else 'not fitted (nominal FK)'}")
    stuck = [f"j{j + 1}" for j in joints if abs(off[j]) > 0.99 * max_offset]
    if joints:
        print("joint offsets (deg, true = reported + offset): " + " ".join(f"j{i + 1} {off[i]:+.3f}" for i in range(6)))
        if stuck:
            print(f"WARNING: {', '.join(stuck)} at the +-{max_offset} deg bound; don't use this result")
    print(f"camera in flange: t {np.round(X[:3, 3], 2).tolist()} mm, rpy "
          f"{np.round(R.as_euler('xyz', degrees=True), 3).tolist()} deg")
    print("per-view tag error (mm / deg): " + "  ".join(f"{k + 1}:{e[0]:.1f}/{e[1]:.2f}" for k, e in enumerate(err)))
    print(f"leave-one-out tag error: {np.median(loo[:, 0]):.1f} mm / {np.median(loo[:, 1]):.2f} deg median, "
          f"{loo[:, 0].max():.1f} mm / {loo[:, 1].max():.2f} deg worst")
    out = {'position': (X[:3, 3] / 1000).tolist(), 'orientation': R.as_quat().tolist(),
           'rpy': [R.as_euler('xyz').tolist()], 'frame': 'camera pose in the flange (end_pose) frame',
           'joint_offsets_deg': off.tolist(), 'joint_offsets_note': 'true joint angle = reported + offset; unfitted joints are 0',
           'joint_offsets_at_bound': stuck,
           'method': 'tag on breadboard, tag pose from the top camera',
           'quality': {'view_err_mm': err[:, 0].tolist(), 'view_err_deg': err[:, 1].tolist(),
                       'loo_mm': loo[:, 0].tolist(), 'loo_deg': loo[:, 1].tolist(), 'views': len(q)},
           'samples': list(samples)}
    path = os.path.join(data_save_path, 'eye_in_hand')
    os.makedirs(path, exist_ok=True)
    fn = os.path.join(path, f"{timestamp()}_tag_calibration.json")
    with open(fn, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved {fn}")
    return dict(out, file=fn)


# ---------------------------------------------------------------- run: both cameras at once

ARUCO_DICTS = ['DICT_ARUCO_ORIGINAL', 'DICT_4X4_1000', 'DICT_5X5_1000', 'DICT_6X6_1000', 'DICT_7X7_1000',
               'DICT_APRILTAG_36h11', 'DICT_ARUCO_MIP_36h12']


class TagDetector:
    """ArUco detection + pose (camera <- tag, mm) with OpenCV, the same code for both cameras so
    the tag frame convention is identical. Without --dictionary, the first dictionary in which
    the marker id shows up is used from then on."""

    def __init__(self, marker_id, size_mm, dict_name=None):
        import cv2
        if not hasattr(cv2, 'aruco'):
            raise SystemExit("this OpenCV has no aruco module (pip install opencv-contrib-python)")
        self.cv2, self.aruco = cv2, cv2.aruco
        self.marker_id, self.size = marker_id, size_mm
        self.dict_name = dict_name
        self.names = [dict_name] if dict_name else [d for d in ARUCO_DICTS if hasattr(cv2.aruco, d)]
        self.params = (self.aruco.DetectorParameters() if hasattr(self.aruco, 'DetectorParameters')
                       else self.aruco.DetectorParameters_create())
        self.params.cornerRefinementMethod = self.aruco.CORNER_REFINE_SUBPIX
        h = size_mm / 2
        # OpenCV ArUco corner order (TL, TR, BR, BL), tag frame x right, y up, z out of the tag
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

    def _detect(self, gray, name):
        d = self.aruco.getPredefinedDictionary(getattr(self.aruco, name))
        if hasattr(self.aruco, 'ArucoDetector'):
            corners, ids, _ = self.aruco.ArucoDetector(d, self.params).detectMarkers(gray)
        else:
            corners, ids, _ = self.aruco.detectMarkers(gray, d, parameters=self.params)
        return corners, ids

    def detect(self, bgr, K, D):
        cv2 = self.cv2
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        for name in ([self.dict_name] if self.dict_name else self.names):
            corners, ids = self._detect(gray, name)
            if ids is None or self.marker_id not in ids.ravel():
                continue
            if self.dict_name is None:
                self.dict_name = name
                print(f"tag {self.marker_id} found in {name}")
            c = corners[list(ids.ravel()).index(self.marker_id)].reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(self.obj, c, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                return None, None
            return to_T(Rotation.from_rotvec(rvec.ravel()).as_matrix(), tvec.ravel()), c
        return None, None


def cmd_run(args, T_base_top=None):
    """Both cameras live: the top camera measures the tag's pose in the base (and keeps checking
    it), the wrist camera collects the calibration views. Window keys: Enter capture,
    d undo, t re-measure the tag, q fit + quit. T_base_top: top camera -> base (mm), else read
    from --top-calibration. Returns the fit_report result, or None without views."""
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo, JointState
    from top_camera_calibration import to_bgr

    if T_base_top is None:
        T_base_top = load_top_calibration(args.top_calibration)
    det = TagDetector(args.marker_id, args.marker_size, args.dictionary)

    rclpy.init()
    node = Node("wrist_camera_calibration_run")
    st = {'top': None, 'wrist': None, 'K_top': None, 'K_wrist': None, 'cap_joints': None}
    cache = {}                                          # detection per image message, done once

    def info(key):
        return lambda m: st.__setitem__(key, (np.array(m.k).reshape(3, 3), np.array(m.d)))

    def on_joint(m):
        # Joint states only matter during a capture
        if st['cap_joints'] is None:
            return
        names = list(m.name)
        if all(f'joint{k}' in names for k in range(1, 7)):
            st['cap_joints'].append([m.position[names.index(f'joint{k}')] for k in range(1, 7)])
        else:
            st['cap_joints'].append(list(m.position[:6]))

    def detect(key, msg, K):
        # (image, T_cam_tag, corners) for this message, detected only the first time it's seen
        if msg is None or K is None:
            return None, None, None
        if cache.get(key, (None,))[0] is not msg:
            im = to_bgr(msg)
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            cache[key] = (msg, im, *det.detect(im, *K), True)
        else:
            cache[key] = cache[key][:4] + (False,)
        return cache[key][1:4]

    def is_new(key):
        return cache.get(key, (None,) * 5)[4]
    node.create_subscription(Image, args.top_topic, lambda m: st.__setitem__('top', m), qos_profile_sensor_data)
    node.create_subscription(Image, args.wrist_topic, lambda m: st.__setitem__('wrist', m), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, args.top_info, info('K_top'), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, args.wrist_info, info('K_wrist'), qos_profile_sensor_data)
    node.create_subscription(JointState, args.joint_topic, on_joint, 50)

    out_dir = os.path.join(args.data_save_path, 'eye_in_hand')
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{timestamp()}_tag_samples.json")
    views, T_base_tag, tag_info = [], None, {}
    top_poses, cap_tags = [], None                      # tag measurement / capture in progress
    lines = ["measuring the tag with the top camera..."]
    fit_thread = {'t': None, 'msg': None}

    def save():
        with open(fn, 'w') as f:
            json.dump({'T_base_tag_mm': T_base_tag.tolist(), **tag_info, 'size_ratio': tag_info.get('size_ratio', 1.0),
                       'marker_id': args.marker_id, 'marker_size_mm': args.marker_size, 'dictionary': det.dict_name,
                       'T_cam_tag_note': 'from the wrist camera, not size-corrected', 'views': views}, f, indent=2)

    def say(*msgs):
        for m in msgs:
            print(m)
        lines[:] = list(msgs)

    def background_fit():
        q = np.array([v['joints'] for v in views])
        Ts = [size_corrected(v['T_cam_tag_mm'], tag_info['size_ratio']) for v in views]

        def work():
            X, d = fit_views(q, Ts, T_base_tag, [0, 1, 2, 3, 4], args.max_offset)
            e = view_errors(X, d, q, Ts, T_base_tag)
            bad = [k + 1 for k, (em, _) in enumerate(e) if em > args.check_mm / 2]
            fit_thread['msg'] = (f"fit on {len(q)} views: tag error median {np.median(e[:, 0]):.1f} mm, worst "
                                 f"{e[:, 0].max():.1f} mm (view {int(np.argmax(e[:, 0])) + 1})"
                                 + (f"; views over {args.check_mm / 2:.0f} mm: {bad}" if bad else ""))
        fit_thread['t'] = threading.Thread(target=work, daemon=True)
        fit_thread['t'].start()

    print(f"top: {args.top_topic}, wrist: {args.wrist_topic}, joints: {args.joint_topic}; saving to {fn}")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if fit_thread['t'] is not None and not fit_thread['t'].is_alive():
                fit_thread['t'] = None
                say(fit_thread['msg'])
            # Top camera: measure the tag (start, or after 't'); later, check it hasn't moved
            top_img, T_top_tag, _ = detect('top', st['top'], st['K_top'])
            if T_base_tag is None and T_top_tag is not None and is_new('top'):
                top_poses.append(T_top_tag)
                if len(top_poses) >= args.frames:
                    T_base_tag, tag_info = tag_in_base(top_poses, T_base_top, args.board_z, args.tag_thickness)
                    # Raw top-camera measurement, so 'fit --top-calibration' can redo this with a newer
                    # top calibration (the top camera can shift: ~3 deg cost ~40 mm here once)
                    tag_info.update(T_top_tag_mm=average_T(top_poses)[0].tolist(), board_z_mm=args.board_z,
                                    tag_thickness_mm=args.tag_thickness, top_calibration=args.top_calibration)
                    if not args.size_correction:
                        tag_info['size_ratio_measured'], tag_info['size_ratio'] = tag_info['size_ratio'], 1.0
                    top_poses = []
                    say(f"tag in base: {np.round(T_base_tag[:3, 3], 1).tolist()} mm, yaw "
                        f"{np.degrees(np.arctan2(T_base_tag[1, 0], T_base_tag[0, 0])):.2f} deg, size ratio "
                        f"{tag_info['size_ratio']:.3f}" + ("  <- check --marker-size" if abs(tag_info['size_ratio'] - 1) > 0.01 else ""),
                        "pose the arm (teach mode) so the wrist camera sees the tag, then Enter")
            # Wrist camera
            img, T_ct, corners = detect('wrist', st['wrist'], st['K_wrist'])
            if img is None:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                continue
            if cap_tags is not None and T_ct is not None and is_new('wrist'):
                cap_tags.append(T_ct)
            if cap_tags is not None and len(cap_tags) >= args.frames:
                js = np.array(st['cap_joints'])
                st['cap_joints'] = None
                tags, cap_tags = cap_tags, None
                T_m, dt, dr = average_T(tags)
                motion = np.degrees(np.ptp(js, axis=0)).max() if len(js) else np.inf
                if len(js) < 5:
                    say(f"rejected: only {len(js)} joint messages during the capture (joint states not updating?)")
                elif motion > args.max_joint_motion:
                    say(f"rejected: joints moved {motion:.3f} deg during the capture")
                elif dr.max() > args.max_tag_jitter:
                    say(f"rejected: tag orientation jumps {dr.max():.1f} deg between frames (ArUco pose ambiguity; "
                        f"view the tag more obliquely or from closer)")
                else:
                    q = js.mean(0)
                    views.append({'joints': q.tolist(), 'T_cam_tag_mm': T_m.tolist(), 'tag_frames': len(tags),
                                  'tag_spread_mm': float(dt.max()), 'tag_spread_deg': float(dr.max()),
                                  'joint_msgs': int(len(js))})
                    save()
                    Xk = mount(q, size_corrected(T_m, tag_info['size_ratio']), T_base_tag)
                    msg = f"view {len(views)}: tag jitter {dt.max():.2f} mm / {dr.max():.2f} deg"
                    if len(views) >= 2:
                        prev = [mount(np.array(v['joints']), size_corrected(v['T_cam_tag_mm'], tag_info['size_ratio']),
                                      T_base_tag) for v in views[:-1]]
                        Xm, _, _ = average_T(prev)
                        d_t = np.linalg.norm(Xk[:3, 3] - Xm[:3, 3])
                        d_r = np.degrees(np.linalg.norm(Rotation.from_matrix(Xm[:3, :3].T @ Xk[:3, :3]).as_rotvec()))
                        msg += f", mount vs earlier views {d_t:.1f} mm / {d_r:.2f} deg"
                        if d_t > args.check_mm or d_r > args.check_deg:
                            msg += "  BAD? press d to remove"
                    # The tag must not have moved: compare with what the top camera sees now
                    if T_top_tag is not None:
                        now, _ = tag_in_base([T_top_tag], T_base_top, args.board_z, args.tag_thickness)
                        moved = np.linalg.norm(now[:3, 3] - T_base_tag[:3, 3])
                        if moved > args.moved_mm:
                            msg += f"  TAG MOVED {moved:.1f} mm? (t re-measures it; earlier views then don't match)"
                    say(msg)
                    if len(views) >= 8 and fit_thread['t'] is None:
                        background_fit()
            # Display: wrist view, tag outline, top-camera inset, status lines
            vis = img.copy()
            if corners is not None:
                cv2.polylines(vis, [np.round(corners).astype(np.int32)], True, (0, 255, 0), 2)
            if top_img is not None:
                inset = cv2.resize(top_img, (320, 180))
                cv2.rectangle(inset, (0, 0), (319, 179), (0, 255, 0) if T_top_tag is not None else (0, 0, 255), 3)
                vis[0:180, vis.shape[1] - 320:] = inset
            status = [f"views {len(views)}  tag {'seen' if corners is not None else 'NOT seen'}"
                      + ("  capturing..." if cap_tags is not None else "")
                      + "   [Enter capture, d undo, t re-measure tag, q fit+quit]"] + lines
            for i, t in enumerate(status):
                y = vis.shape[0] - 15 - 28 * (len(status) - 1 - i)
                cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("wrist camera calibration", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k in (13, 10, 32):
                if T_base_tag is None:
                    say("the top camera hasn't measured the tag yet (is it visible in the inset?)")
                elif cap_tags is None:
                    cap_tags, st['cap_joints'] = [], []
                    say("capturing... keep the arm still")
            elif k == ord('d') and views:
                views.pop()
                save()
                say(f"removed last view, {len(views)} left")
            elif k == ord('t'):
                T_base_tag, top_poses = None, []
                say("re-measuring the tag with the top camera...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    if not views:
        print("no views captured")
        return None
    print(f"{len(views)} views in {fn}")
    q = np.array([v['joints'] for v in views])
    Ts = [size_corrected(v['T_cam_tag_mm'], tag_info['size_ratio']) for v in views]
    return fit_report(q, Ts, T_base_tag, args.offsets, args.max_offset, [fn], args.data_save_path)


def add_run_args(ap):
    """`run` options (shared with multi_camera_calibration.py), except --top-calibration."""
    ap.add_argument('--marker-id', type=int, default=100)
    ap.add_argument('--marker-size', type=float, default=100.0, help="printed tag side, mm (measure it)")
    ap.add_argument('--dictionary', help="OpenCV ArUco dictionary, e.g. DICT_ARUCO_ORIGINAL (default: auto)")
    ap.add_argument('--top-topic', default='/top/camera/color/image_raw')
    ap.add_argument('--top-info', default='/top/camera/color/camera_info')
    ap.add_argument('--wrist-topic', default='/wrist/camera/color/image_raw')
    ap.add_argument('--wrist-info', default='/wrist/camera/color/camera_info')
    ap.add_argument('--joint-topic', default='/joint_states_single')
    ap.add_argument('--frames', type=int, default=30, help="tag detections averaged (top measurement and per view)")
    ap.add_argument('--board-z', type=float, default=0.7, help="board top surface z in base, mm")
    ap.add_argument('--tag-thickness', type=float, default=0.0, help="printed tag height above the board, mm")
    ap.add_argument('--max-joint-motion', type=float, default=0.05, help="deg during a capture")
    ap.add_argument('--max-tag-jitter', type=float, default=2.0, help="deg; reject a view whose tag orientation jumps")
    ap.add_argument('--check-mm', type=float, default=10.0, help="flag a view whose mount is this far from the others")
    ap.add_argument('--check-deg', type=float, default=2.0)
    ap.add_argument('--moved-mm', type=float, default=2.0, help="warn if the top camera sees the tag this far off")
    ap.add_argument('--offsets', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--max-offset', type=float, default=10.0)
    ap.add_argument('--size-correction', action='store_true',
                    help="rescale wrist distances by the top camera's tag size estimate (unmeasured tag only)")
    ap.add_argument('--data-save-path', default='./data')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('tag', 'record'):
        s = sub.add_parser(name)
        s.add_argument('--marker-topic', default='/aruco_single/pose')
        s.add_argument('--frames', type=int, default=30, help="tag poses averaged")
        s.add_argument('--timeout', type=float, default=10.0, help="s to collect them")
        s.add_argument('--data-save-path', default='./data')
    sub.choices['tag'].add_argument('--top-calibration', required=True,
                                    help="top camera -> base result (top_camera_calibration.py extrinsic)")
    sub.choices['tag'].add_argument('--board-z', type=float, default=0.7, help="board top surface z in base, mm")
    sub.choices['tag'].add_argument('--tag-thickness', type=float, default=0.0, help="printed tag height above the board, mm")
    r = sub.choices['record']
    r.add_argument('--tag', required=True, help="*_tag_in_base.json from the tag step")
    r.add_argument('--joint-topic', default='/joint_states_single')
    r.add_argument('--max-joint-motion', type=float, default=0.05, help="deg during a capture")
    r.add_argument('--max-tag-jitter', type=float, default=2.0, help="deg; reject a view whose tag orientation jumps")
    r.add_argument('--check-mm', type=float, default=10.0, help="flag a view whose mount is this far from the others")
    r.add_argument('--check-deg', type=float, default=2.0)
    r.add_argument('--size-correction', action='store_true',
                   help="rescale wrist distances by the top camera's tag size estimate (unmeasured tag only)")
    r = sub.add_parser('run', help="both cameras at once (multi_camera_calibration.py wraps this)")
    r.add_argument('--top-calibration', required=True, help="top camera -> base result (top_camera_calibration.py extrinsic)")
    add_run_args(r)
    f = sub.add_parser('fit')
    f.add_argument('samples', nargs='+')
    f.add_argument('--offsets', choices=['auto', 'on', 'off'], default='auto',
                   help="fit the j1..j5 zero offsets too (auto: from 8 views on)")
    f.add_argument('--max-offset', type=float, default=10.0, help="deg bound on each offset")
    f.add_argument('--top-calibration', help="recompute the tag's base pose with this (newer) top calibration")
    f.add_argument('--size-correction', action='store_true',
                   help="rescale wrist distances by the top camera's tag size estimate (unmeasured tag only)")
    f.add_argument('--data-save-path', default='./data')
    args = ap.parse_args()
    {'tag': cmd_tag, 'record': cmd_record, 'fit': cmd_fit, 'run': cmd_run}[args.cmd](args)


if __name__ == '__main__':
    main()
