#!/usr/bin/env python3
# -*-coding:utf8-*-
# Automated wrist camera calibration: the top camera measures the tag lying on the board, then
# the arm drives the wrist camera through planned views of it and captures each one, until the
# fit is good enough. Same result as wrist_camera_calibration.py run, without teaching the views
# by hand; multi_camera_calibration.py runs it as the wrist step (--wrist-mode auto, the default).
#
# Views: camera positions on a cap above the tag (--view-tilt deg from straight down at
# --view-dist mm), aimed near the tag, each solved by IK on the same FK as the fit from a few
# wrist seeds, so the rotation about the view axis varies too. A view is kept if the whole tag
# lands inside the image, the joints stay inside the limits, and every link, the gripper tip and
# the camera stay --clearance mm above the board. Kept views are visited most different first
# (farthest point in joint space), so the first ones already pin the fit down.
#
# Mount guess: planning needs the camera mount (flange -> camera) roughly, from --mount-guess,
# else the existing wrist_calibration.json, else nominal_wrist_mount.json (station 11's measured
# mount). Every --refit-after views it is refitted from the captures and the remaining views are
# re-solved with it.
#
# Stop: from --min-views on, a full fit (mount + j1..j5 offsets) after every view; done when the
# leave-one-out tag error median is under --target-loo-mm and the mount moved less than
# --converge-mm / --converge-deg since the previous fit, else after --max-views (not converged).
#
# Motion: JointState on --command-topic (default /joint_states: start_station_calibration.launch.py
# remaps the Piper's command input there, outside the station namespace). All six joints by name,
# since the Piper node takes a missing joint as 0; position[6] holds the gripper where it is
# (it is commanded on every message); velocity[6] is the speed in percent (the node runs at 100
# without it). Moves are cut into waypoints of at most --step-deg per joint, so the arm follows
# the straight joint-space line that was checked for clearance instead of each joint finishing on
# its own; a line that isn't clear goes through the pose the arm started in. q in the window or
# Ctrl-C stops sending commands (the arm stays at its last command) and fits what was captured.

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from breadboard_model import fk, inv_T
from joint_offset_fit import A, ALPHA, THETA, D
from wrist_camera_calibration import (TagDetector, average_T, fit_report, fit_views, mount, pose_to_T,
                                      size_corrected, tag_in_base, timestamp, to_T, view_errors)

# Piper joint limits, rad (piper_kinematics.py); planning stays --limit-margin inside them
JOINT_LIMITS = np.array([(-2.6179, 2.6179), (0.0, 3.14), (-2.967, 0.0),
                         (-1.745, 1.745), (-1.22, 1.22), (-2.09439, 2.09439)])
JOINT_NAMES = [f'joint{k}' for k in range(1, 7)]
TOOL_MM = 120.0      # flange -> gripper tip along the flange z (piper_kinematics.py's d6 extension)
ROT_MM = 200.0       # 1 rad of view-axis error weighs like 200 mm in the IK


class Abort(Exception):
    pass


# ---------------------------------------------------------------- kinematics and planning

def frames(q):
    """Base, then every link frame of the flange DH chain (same chain as breadboard_model.fk), mm."""
    out, T = [np.eye(4)], np.eye(4)
    for i in range(6):
        ca, sa = np.cos(ALPHA[i]), np.sin(ALPHA[i])
        ct, st = np.cos(q[i] + THETA[i]), np.sin(q[i] + THETA[i])
        T = T @ np.array([[ct, -st, 0, A[i]], [st * ca, ct * ca, -sa, -sa * D[i]],
                          [st * sa, ct * sa, ca, ca * D[i]], [0, 0, 0, 1]])
        out.append(T)
    return out


def lowest_point(q, X):
    """Lowest z (mm) along the links from the shoulder on, the gripper tip and the camera."""
    F = frames(q)
    o = [T[:3, 3] for T in F[2:]]
    z = [a[2] + s * (b[2] - a[2]) for a, b in zip(o[:-1], o[1:]) for s in np.linspace(0, 1, 5)]
    z.append((F[-1] @ [0, 0, TOOL_MM, 1])[2])
    z.append((F[-1] @ X)[2, 3])
    return min(z)


def is_clear(q, X, args):
    return lowest_point(q, X) >= args.board_z + args.clearance


def path_clear(qa, qb, X, args):
    # The straight joint-space line, checked every 2 deg of the largest joint motion
    n = max(2, int(np.ceil(np.degrees(np.abs(qb - qa).max()) / 2.0)) + 1)
    return all(is_clear(qa + s * (qb - qa), X, args) for s in np.linspace(0, 1, n))


def limits(args):
    m = np.radians(args.limit_margin)
    lo, hi = JOINT_LIMITS[:, 0] + m, JOINT_LIMITS[:, 1] - m
    # j4 / j6 kept to +-wrist_range so the camera cable doesn't wind around the wrist
    w = np.radians(args.wrist_range)
    lo[[3, 5]], hi[[3, 5]] = np.maximum(lo[[3, 5]], -w), np.minimum(hi[[3, 5]], w)
    return lo, hi


def ik_view(cam_p, z_t, seeds, X, lo, hi):
    """Joints putting the camera at cam_p (mm) looking along z_t, any rotation about the view
    axis; one solution per seed that converges (within 0.5 mm / 0.2 deg)."""
    def res(q):
        T = fk(q[None])[0] @ X
        return np.r_[T[:3, 3] - cam_p, (T[:3, 2] - z_t) * ROT_MM]
    out = []
    for s in seeds:
        sol = least_squares(res, np.clip(s, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=60)
        if np.linalg.norm(sol.fun[:3]) < 0.5 and np.linalg.norm(sol.fun[3:]) / ROT_MM < np.radians(0.2):
            out.append(sol.x)
    return out


def tag_in_image(T_base_cam, T_base_tag, K, size_mm, wh, margin):
    """The whole tag inside the image (margin px from the edges) and at least 40 px across."""
    h = size_mm / 2
    c = np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1], [-h, -h, 0, 1]]).T
    Pc = (inv_T(T_base_cam) @ T_base_tag @ c)[:3].T
    if (Pc[:, 2] <= 50).any():
        return False
    uv = Pc[:, :2] / Pc[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]
    inside = (uv > margin).all() and (uv[:, 0] < wh[0] - margin).all() and (uv[:, 1] < wh[1] - margin).all()
    return bool(inside and np.linalg.norm(uv[0] - uv[2]) > 40 * np.sqrt(2))


def plan_views(T_base_tag, X, K, wh, q0, args, log=print):
    """Candidate views: [{'cam_p', 'z', 'q'}], most different first (farthest point in joint space)."""
    lo, hi = limits(args)
    p = T_base_tag[:3, 3]
    rng = np.random.default_rng(0)
    # Seeds: where the arm is, plus wrist variations of it, so solutions differ in camera roll
    seeds0 = [q0] + [np.r_[q0[:3], j4, q0[4], j6] for j4 in (-0.8, 0.0, 0.8) for j6 in (-1.2, 0.0, 1.2)]
    cands = []
    t0 = time.time()
    for tilt in np.radians(args.view_tilt):
        for az in np.radians(np.arange(0.0, 360.0, 360.0 / args.view_azimuths)):
            for dist in args.view_dist:
                cam_p = p + dist * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
                aim = p + np.r_[rng.uniform(-args.aim_jitter, args.aim_jitter, 2), 0.0]
                z_t = (aim - cam_p) / np.linalg.norm(aim - cam_p)
                seeds = [c['q'] for c in cands[-3:]] + seeds0
                kept = []
                for q in ik_view(cam_p, z_t, seeds, X, lo, hi):
                    if any(np.abs(q - k).max() < np.radians(15) for k in kept):
                        continue                                    # same solution again
                    T = fk(q[None])[0] @ X
                    if tag_in_image(T, T_base_tag, K, args.marker_size, wh, args.image_margin) and is_clear(q, X, args):
                        kept.append(q)
                cands += [{'cam_p': cam_p, 'z': z_t, 'q': q} for q in kept]
    log(f"planned {len(cands)} reachable views in {time.time() - t0:.1f} s")
    if not cands:
        return []
    # Farthest point order, starting from the view closest to where the arm is
    Q = np.array([c['q'] for c in cands])
    order = [int(np.argmin(np.abs(Q - q0).max(1)))]
    dmin = np.linalg.norm(Q - Q[order[0]], axis=1)
    while len(order) < min(len(cands), args.max_views * 2):
        k = int(np.argmax(dmin))
        order.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(Q - Q[k], axis=1))
    return [cands[k] for k in order]


def resolve(view, X, args):
    """Re-solve a planned view with a new mount estimate, near its old joints; None if lost."""
    lo, hi = limits(args)
    sols = ik_view(view['cam_p'], view['z'], [view['q']], X, lo, hi)
    if not sols or not is_clear(sols[0], X, args):
        return None
    return dict(view, q=sols[0])


# ---------------------------------------------------------------- fit bookkeeping

def view_data(views, ratio):
    q = np.array([v['joints'] for v in views])
    return q, [size_corrected(v['T_cam_tag_mm'], ratio) for v in views]


def full_fit(views, T_base_tag, ratio, args):
    """Mount + j1..j5 offsets, with the leave-one-out tag error. -> (X, delta, loo (n, 2), stuck)"""
    q, Ts = view_data(views, ratio)
    joints = [0, 1, 2, 3, 4]
    X, d = fit_views(q, Ts, T_base_tag, joints, args.max_offset)
    loo = []
    for k in range(len(q)):
        m = np.arange(len(q)) != k
        Xk, dk = fit_views(q[m], [Ts[i] for i in np.flatnonzero(m)], T_base_tag, joints, args.max_offset)
        loo.append(view_errors(Xk, dk, q[k:k + 1], [Ts[k]], T_base_tag)[0])
    stuck = [j for j in joints if abs(np.degrees(d[j])) > 0.99 * args.max_offset]
    return X, d, np.array(loo), stuck


def T_diff(Ta, Tb):
    return (np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]),
            np.degrees(np.linalg.norm(Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).as_rotvec())))


# ---------------------------------------------------------------- the run

def run_auto(args, rig, X0, log=print):
    """Measure the tag, plan, visit and capture views until converged. Returns the fit_report
    result (+ 'auto': how it ended), or None with fewer than --min-views views."""
    T_base_tag, tag_info = rig.measure_tag()
    ratio = tag_info.get('size_ratio', 1.0)
    log(f"tag in base: {np.round(T_base_tag[:3, 3], 1).tolist()} mm, yaw "
        f"{np.degrees(np.arctan2(T_base_tag[1, 0], T_base_tag[0, 0])):.1f} deg")
    q0 = rig.joints()
    home = q0 if is_clear(q0, X0, args) else None
    if home is None:
        log(f"NOTE: the start pose is within {args.clearance} mm of the board by the model; moves that "
            f"aren't clear on their own are skipped instead of routed through it")
    K, wh = rig.wrist_camera()
    plan = plan_views(T_base_tag, X0, K, wh, q0, args, log)
    if len(plan) < args.min_views:
        raise SystemExit(f"only {len(plan)} reachable views (need {args.min_views}); move the tag nearer the "
                         f"arm's reach, or widen --view-tilt / --view-dist")
    if not args.yes:
        ans = input(f"\n{len(plan)} views planned, up to {args.max_views} visited at {args.speed}% speed. The arm "
                    f"must be enabled, OUT of teaching mode, workspace clear.\nEnter to start, anything else aborts: ")
        if ans.strip() != '':
            raise SystemExit("aborted, the arm didn't move")

    out_dir = os.path.join(args.data_save_path, 'eye_in_hand')
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{timestamp()}_auto_tag_samples.json")
    views, X, prev_X, end = [], X0, None, 'max views'
    fit_X = None

    def save():
        with open(fn, 'w') as f:
            json.dump({'T_base_tag_mm': T_base_tag.tolist(), **tag_info, 'size_ratio': ratio,
                       'marker_id': args.marker_id, 'marker_size_mm': args.marker_size, 'source': 'auto',
                       'T_cam_tag_note': 'from the wrist camera, not size-corrected', 'views': views}, f, indent=2)

    q_now = q0
    try:
        while plan and len(views) < args.max_views:
            v = plan.pop(0)
            if not path_clear(q_now, v['q'], X, args):
                if home is None or not (path_clear(q_now, home, X, args) and path_clear(home, v['q'], X, args)):
                    log("skipped a view: no clear path to it")
                    continue
                rig.move(home)
            log(f"view {len(views) + 1}: moving to joints {np.round(np.degrees(v['q']), 1).tolist()} deg")
            if not rig.move(v['q']):
                log("  did not reach it in time, skipped")
                q_now = rig.joints()
                continue
            q_now = rig.joints()
            tags, js = rig.capture(args.frames)
            js = np.array(js)
            if len(tags) < max(5, args.frames // 2):
                log(f"  tag not seen ({len(tags)} detections), skipped")
                continue
            if len(js) < 5 or np.degrees(np.ptp(js, axis=0)).max() > args.max_joint_motion:
                log("  joints not steady during the capture, skipped")
                continue
            T_m, dt, dr = average_T(tags)
            if dr.max() > args.max_tag_jitter:
                log(f"  tag orientation jumps {dr.max():.1f} deg between frames (pose ambiguity), skipped")
                continue
            views.append({'joints': js.mean(0).tolist(), 'T_cam_tag_mm': T_m.tolist(), 'tag_frames': len(tags),
                          'tag_spread_mm': float(dt.max()), 'tag_spread_deg': float(dr.max()),
                          'joint_msgs': int(len(js)), 'planned_joints': v['q'].tolist()})
            save()
            n = len(views)
            if n % args.refit_after == 0:
                # Better mount for planning (nominal FK, like the plan), then re-solve what's left
                q, Ts = view_data(views, ratio)
                X = fit_views(q, Ts, T_base_tag)[0]
                plan = [r for r in (resolve(w, X, args) for w in plan) if r is not None]
                log(f"  mount refit from {n} views: t {np.round(X[:3, 3], 1).tolist()} mm; {len(plan)} views left")
            if n >= args.min_views:
                fit_X, _, loo, stuck = full_fit(views, T_base_tag, ratio, args)
                med = float(np.median(loo[:, 0]))
                moved = T_diff(fit_X, prev_X) if prev_X is not None else (np.inf, np.inf)
                prev_X = fit_X
                log(f"  fit on {n} views: leave-one-out median {med:.2f} mm (target {args.target_loo_mm})"
                    + (f", mount moved {moved[0]:.2f} mm / {moved[1]:.3f} deg" if np.isfinite(moved[0]) else "")
                    + (f", offsets at the bound: {['j%d' % (j + 1) for j in stuck]}" if stuck else ""))
                if med < args.target_loo_mm and moved[0] < args.converge_mm and moved[1] < args.converge_deg and not stuck:
                    end = 'converged'
                    break
        else:
            if not plan and len(views) < args.max_views:
                end = 'out of views'
    except (Abort, KeyboardInterrupt):
        end = 'stopped'
        log("\nstopped: no more commands sent")
    else:
        if home is not None and path_clear(rig.joints(), home, X, args):
            log("back to the start pose")
            rig.move(home)
    finally:
        rig.close()
    log(f"{len(views)} views ({end}) in {fn}")
    if len(views) < args.min_views:
        log(f"fewer than {args.min_views} views, not fitted")
        return None
    q, Ts = view_data(views, ratio)
    out = fit_report(q, Ts, T_base_tag, args.offsets, args.max_offset, [fn], args.data_save_path)
    out['auto'] = {'end': end, 'converged': end == 'converged', 'views': len(views)}
    return out


NOMINAL_MOUNT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nominal_wrist_mount.json')


def load_mount(path):
    """Flange -> camera (mm) from a wrist calibration result (or nominal_wrist_mount.json)."""
    with open(path) as f:
        c = json.load(f)
    if 'rpy_deg' in c and 'orientation' not in c:
        c['orientation'] = Rotation.from_euler('xyz', c['rpy_deg'], degrees=True).as_quat().tolist()
    if 'position' not in c or 'orientation' not in c:
        raise SystemExit(f"{path} has no position/orientation; pass a wrist calibration (*_tag_calibration.json "
                         f"or wrist_calibration.json) as --mount-guess")
    return pose_to_T(c['position'], c['orientation'])


# ---------------------------------------------------------------- ROS side

class RosRig:
    """Top + wrist cameras, joint feedback and joint commands; a window shows the wrist view."""

    def __init__(self, args, T_base_top):
        import cv2
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo, JointState
        from top_camera_calibration import to_bgr
        self.cv2, self.rclpy, self.JointState, self.to_bgr = cv2, rclpy, JointState, to_bgr
        self.args, self.T_base_top = args, T_base_top
        self.det = TagDetector(args.marker_id, args.marker_size, args.dictionary)
        rclpy.init()
        self.node = Node("wrist_camera_auto")
        self.st = {'top': None, 'wrist': None, 'K_top': None, 'K_wrist': None, 'q': None, 'grip': 0.0, 'cap': None}
        self.seen = {}                                      # last image message detected, per camera
        self.status = []

        def info(key):
            return lambda m: self.st.__setitem__(key, (np.array(m.k).reshape(3, 3), np.array(m.d), (m.width, m.height)))

        def on_joint(m):
            names, pos = list(m.name), list(m.position)
            if all(n in names for n in JOINT_NAMES):
                q = [pos[names.index(n)] for n in JOINT_NAMES]
            else:
                q = pos[:6]
            self.st['q'] = np.array(q)
            if len(pos) >= 7:
                self.st['grip'] = float(pos[6])
            if self.st['cap'] is not None:
                self.st['cap'].append(q)
        sd = qos_profile_sensor_data
        self.node.create_subscription(Image, args.top_topic, lambda m: self.st.__setitem__('top', m), sd)
        self.node.create_subscription(Image, args.wrist_topic, lambda m: self.st.__setitem__('wrist', m), sd)
        self.node.create_subscription(CameraInfo, args.top_info, info('K_top'), sd)
        self.node.create_subscription(CameraInfo, args.wrist_info, info('K_wrist'), sd)
        self.node.create_subscription(JointState, args.joint_topic, on_joint, 50)
        self.pub = self.node.create_publisher(JointState, args.command_topic, 10)
        print(f"top {args.top_topic}, wrist {args.wrist_topic}, joints {args.joint_topic}, commands {args.command_topic}")
        self.wait(lambda: self.st['q'] is not None and self.st['K_wrist'] is not None and self.st['wrist'] is not None,
                  10.0, "joint feedback, wrist image and camera_info")

    def spin(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.01)
        self.show()

    def wait(self, cond, timeout, what):
        t0 = time.time()
        while not cond():
            if time.time() - t0 > timeout:
                raise SystemExit(f"no {what} within {timeout:.0f} s; is start_station_calibration.launch.py running "
                                 f"(same station / ROS_DOMAIN_ID)?")
            self.spin()

    def detect(self, key):
        # Tag pose in this camera for the newest image, once per message: (T, corners, is_new)
        msg, K = self.st[key], self.st['K_' + key]
        if msg is None or K is None:
            return None, None, False
        if self.seen.get(key, (None,))[0] is msg:
            return self.seen[key][1], self.seen[key][2], False
        im = self.to_bgr(msg)
        if im.ndim == 2:
            im = self.cv2.cvtColor(im, self.cv2.COLOR_GRAY2BGR)
        T, c = self.det.detect(im, K[0], K[1])
        self.seen[key] = (msg, T, c, im)
        return T, c, True

    def show(self):
        cv2 = self.cv2
        if 'wrist' not in self.seen:
            self.detect('wrist')
            if 'wrist' not in self.seen:
                return
        _, _, c, im = self.seen['wrist']
        vis = im.copy()
        if c is not None:
            cv2.polylines(vis, [np.round(c).astype(np.int32)], True, (0, 255, 0), 2)
        lines = self.status[-3:] + ["[q stop]"]
        for i, t in enumerate(lines):
            y = vis.shape[0] - 15 - 28 * (len(lines) - 1 - i)
            cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("wrist camera auto calibration", vis)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            raise Abort()

    def log(self, msg):
        print(msg)
        self.status.append(msg.strip().split('\n')[-1][:110])

    def joints(self):
        self.spin()
        return self.st['q'].copy()

    def wrist_camera(self):
        K, _, wh = self.st['K_wrist']
        return K, wh

    def measure_tag(self):
        a = self.args
        self.wait(lambda: self.st['K_top'] is not None and self.st['top'] is not None, 10.0, "top camera image")
        poses, t0 = [], time.time()
        while len(poses) < a.frames and time.time() - t0 < 15.0:
            self.spin()
            T, _, new = self.detect('top')
            if T is not None and new:
                poses.append(T)
        if len(poses) < max(5, a.frames // 2):
            raise SystemExit(f"the top camera saw the tag only {len(poses)} times; is it on the board, in view, "
                             f"and not under the arm?")
        T_base_tag, info = tag_in_base(poses, self.T_base_top, a.board_z, a.tag_thickness)
        info.update(T_top_tag_mm=average_T(poses)[0].tolist(), board_z_mm=a.board_z,
                    tag_thickness_mm=a.tag_thickness, top_calibration=a.top_calibration)
        if not a.size_correction:
            info['size_ratio_measured'], info['size_ratio'] = info['size_ratio'], 1.0
        if abs(info['size_ratio_measured' if not a.size_correction else 'size_ratio'] - 1) > 0.01:
            self.log("NOTE: the top camera's tag size estimate is off by over 1%; check --marker-size")
        return T_base_tag, info

    def send(self, q):
        m = self.JointState()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.name = JOINT_NAMES + ['gripper']
        m.position = [float(x) for x in q] + [self.st['grip']]
        m.velocity = [0.0] * 6 + [float(self.args.speed)]
        self.pub.publish(m)

    def move(self, q_target):
        """Through waypoints of at most --step-deg per joint; True once settled at q_target."""
        a = self.args
        q_start = self.joints()
        n = max(1, int(np.ceil(np.degrees(np.abs(q_target - q_start).max()) / a.step_deg)))
        t0 = time.time()
        for k in range(1, n + 1):
            wp = q_start + (q_target - q_start) * k / n
            last = k == n
            tol = np.radians(a.joint_tol if last else a.step_deg)
            reached_at, sent = None, 0.0
            while True:
                now = time.time()
                if now - t0 > a.move_timeout:
                    return False
                if now - sent >= 0.1:
                    self.send(wp)
                    sent = now
                self.spin()
                err = np.abs(self.st['q'] - wp).max()
                if err < tol:
                    if not last:
                        break
                    reached_at = reached_at or now
                    if now - reached_at >= a.settle:
                        return True
                else:
                    reached_at = None

    def capture(self, n):
        self.st['cap'] = []
        tags, t0 = [], time.time()
        while len(tags) < n and time.time() - t0 < 10.0:
            self.spin()
            T, _, new = self.detect('wrist')
            if T is not None and new:
                tags.append(T)
        js, self.st['cap'] = self.st['cap'], None
        return tags, js

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()
        self.cv2.destroyAllWindows()
        self.cv2.waitKey(1)


def add_auto_args(ap):
    """Options of the automated wrist step (on top of wrist_camera_calibration.add_run_args)."""
    ap.add_argument('--mount-guess', help="wrist calibration json for planning (default: the existing "
                    "wrist_calibration.json, else nominal_wrist_mount.json)")
    ap.add_argument('--command-topic', default='/joint_states', help="Piper joint command topic")
    ap.add_argument('--speed', type=int, default=20, help="Piper speed, percent")
    ap.add_argument('--step-deg', type=float, default=5.0, help="max joint change per waypoint")
    ap.add_argument('--joint-tol', type=float, default=0.3, help="deg: arrived when every joint is this close")
    ap.add_argument('--settle', type=float, default=1.0, help="s at the target before capturing")
    ap.add_argument('--move-timeout', type=float, default=60.0, help="s per move")
    ap.add_argument('--clearance', type=float, default=40.0, help="mm above the board for links, gripper tip and camera")
    ap.add_argument('--limit-margin', type=float, default=5.0, help="deg inside the joint limits")
    ap.add_argument('--wrist-range', type=float, default=90.0, help="deg; |j4| and |j6| limit (camera cable)")
    ap.add_argument('--view-tilt', type=float, nargs='+', default=[20.0, 35.0, 50.0], help="deg from straight down")
    ap.add_argument('--view-dist', type=float, nargs='+', default=[250.0, 330.0, 410.0], help="camera to tag, mm")
    ap.add_argument('--view-azimuths', type=int, default=8, help="directions around the tag")
    ap.add_argument('--aim-jitter', type=float, default=30.0, help="mm; views aim this far around the tag center")
    ap.add_argument('--image-margin', type=float, default=60.0, help="px; the tag must be this far inside the image")
    ap.add_argument('--min-views', type=int, default=8)
    ap.add_argument('--max-views', type=int, default=20)
    ap.add_argument('--refit-after', type=int, default=4, help="refit the planning mount every this many views")
    ap.add_argument('--target-loo-mm', type=float, default=3.0,
                    help="leave-one-out tag error median to stop at (hand-taught real runs reached 2.6-2.8)")
    ap.add_argument('--converge-mm', type=float, default=0.5)
    ap.add_argument('--converge-deg', type=float, default=0.1)
    ap.add_argument('--yes', action='store_true', help="don't ask before the first move")
