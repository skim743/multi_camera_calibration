#!/usr/bin/env python3
# -*-coding:utf8-*-
# Offline joint-offset fit for an eye_to_hand calibration.
#
# Model per sample i:  T_cam_marker_i = X^-1 * FK(q_i + delta) * Y
#   X: camera -> base, Y: flange -> marker, delta: zero offsets of j2..j5.
# j1 and j6 offsets are not identifiable (absorbed by X and Y). FK uses the DH set that
# reproduces the firmware's end_pose, ending at the flange (tool length is absorbed by Y).
#
# python3 joint_offset_fit.py <samples.json> [<samples.json> ...]
#   Every file must share the same marker mount (one Y); X is initialized from Park.

import sys
import json
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

A     = [0, 0, 285.03, -21.98, 0, 0]
ALPHA = [0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, np.pi / 2]
THETA = [0, -np.pi * 172.22 / 180, -102.78 / 180 * np.pi, 0, 0, 0]
D     = [123, 0, 0, 250.75, 0, 91]
FIT_JOINTS = [1, 2, 3, 4]         # j2..j5 (0-based)
SIGMA_POS = 1.0                   # mm, marker position noise used for weighting
SIGMA_ROT = np.radians(0.2)       # rad, marker orientation noise used for weighting

def fk_flange(q):
    # Modified DH, mm
    T = np.eye(4)
    for i in range(6):
        ca, sa = np.cos(ALPHA[i]), np.sin(ALPHA[i])
        ct, st = np.cos(q[i] + THETA[i]), np.sin(q[i] + THETA[i])
        T = T @ np.array([[ct, -st, 0, A[i]],
                          [st * ca, ct * ca, -sa, -sa * D[i]],
                          [st * sa, ct * sa, ca, ca * D[i]],
                          [0, 0, 0, 1]])
    return T

def to_T(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T

def pose_T(d, scale=1000.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(d['orientation']).as_matrix()
    T[:3, 3] = np.array(d['position']) * scale
    return T

def load(files):
    q, M = [], []
    for f in files:
        with open(f) as json_file:
            for s in json.load(json_file)['samples']:
                q.append(np.array(s['joint_positions'][:6]))
                M.append(pose_T(s['marker_pose']))
    return np.array(q), M

def unpack(x, fit_offsets):
    X = to_T(x[0:3], x[3:6])
    Y = to_T(x[6:9], x[9:12])
    delta = np.zeros(6)
    if fit_offsets:
        delta[FIT_JOINTS] = x[12:16]
    return X, Y, delta

def residuals(x, q, M, fit_offsets):
    X, Y, delta = unpack(x, fit_offsets)
    X_inv = np.linalg.inv(X)
    r = []
    for qi, Mi in zip(q, M):
        P = X_inv @ fk_flange(qi + delta) @ Y
        r.append((P[:3, 3] - Mi[:3, 3]) / SIGMA_POS)
        r.append(Rotation.from_matrix(Mi[:3, :3].T @ P[:3, :3]).as_rotvec() / SIGMA_ROT)
    return np.concatenate(r)

def initial_guess(q, M):
    # Park on the FK poses, then Y as the rotation/translation average over samples
    B = [fk_flange(qi) for qi in q]
    B_inv = [np.linalg.inv(b) for b in B]  # eye_to_hand: pass base->gripper
    R_x, t_x = cv2.calibrateHandEye([b[:3, :3] for b in B_inv], [b[:3, 3] for b in B_inv],
                                    [m[:3, :3] for m in M], [m[:3, 3] for m in M],
                                    method=cv2.CALIB_HAND_EYE_PARK)
    X = np.eye(4)
    X[:3, :3] = R_x
    X[:3, 3] = t_x.ravel()
    Ys = [bi @ X @ m for bi, m in zip(B_inv, M)]
    Y_R = Rotation.from_matrix(np.array([y[:3, :3] for y in Ys])).mean()
    Y_t = np.mean([y[:3, 3] for y in Ys], axis=0)
    return np.concatenate([Rotation.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3], Y_R.as_rotvec(), Y_t])

def fit(q, M, fit_offsets, x0):
    x0 = np.concatenate([x0, np.zeros(4)]) if fit_offsets else x0
    sol = least_squares(residuals, x0, args=(q, M, fit_offsets), method='lm')
    r = sol.fun.reshape(-1, 2, 3)
    pos = np.linalg.norm(r[:, 0], axis=1) * SIGMA_POS
    rot = np.degrees(np.linalg.norm(r[:, 1], axis=1) * SIGMA_ROT)
    return sol, pos, rot

def report(name, sol, pos, rot, fit_offsets):
    X, Y, delta = unpack(sol.x, fit_offsets)
    print(f"{name}: marker residual RMS {np.sqrt(np.mean(pos ** 2)):.2f} mm / {np.sqrt(np.mean(rot ** 2)):.3f} deg, "
          f"X t = {np.round(X[:3, 3], 1).tolist()} mm")
    if fit_offsets:
        # 1-sigma from the Gauss-Newton covariance, scaled by the residual variance
        J = sol.jac
        dof = max(len(sol.fun) - len(sol.x), 1)
        cov = np.linalg.pinv(J.T @ J) * (sol.fun @ sol.fun) / dof
        std = np.degrees(np.sqrt(np.diag(cov))[12:16])
        for j, d, s in zip(FIT_JOINTS, np.degrees(delta[FIT_JOINTS]), std):
            print(f"    j{j + 1} offset: {d:+.3f} deg (1-sigma {s:.3f})")
    return X, delta

def main(files):
    q, M = load(files)
    print(f"{len(q)} samples from {len(files)} file(s)\n")
    x0 = initial_guess(q, M)

    sol0, pos0, rot0 = fit(q, M, False, x0)
    report("X, Y only (no offsets)", sol0, pos0, rot0, False)
    sol1, pos1, rot1 = fit(q, M, True, sol0.x)
    X1, delta1 = report("X, Y + j2..j5 offsets", sol1, pos1, rot1, True)

    # Hold out each distinct joint configuration (all its repeats) and refit, with and without offsets
    labels = -np.ones(len(q), dtype=int)
    for i in range(len(q)):
        if labels[i] < 0:
            labels[(labels < 0) & (np.abs(q - q[i]).max(axis=1) < 0.01)] = labels.max() + 1
    held = {False: [], True: []}
    offsets = []
    for c in range(labels.max() + 1):
        mask = labels == c
        M_train = [m for m, k in zip(M, mask) if not k]
        M_test = [m for m, k in zip(M, mask) if k]
        for fit_offsets in (False, True):
            s, _, _ = fit(q[~mask], M_train, fit_offsets, sol0.x)
            r = residuals(s.x, q[mask], M_test, fit_offsets).reshape(-1, 2, 3)
            held[fit_offsets].append((np.linalg.norm(r[:, 0], axis=1) * SIGMA_POS,
                                      np.degrees(np.linalg.norm(r[:, 1], axis=1) * SIGMA_ROT)))
            if fit_offsets:
                offsets.append(np.degrees(unpack(s.x, True)[2][FIT_JOINTS]))
    offsets = np.array(offsets)
    print(f"\nleave-one-configuration-out ({labels.max() + 1} configs):")
    print(f"    offset range j2..j5 (deg): min {np.round(offsets.min(axis=0), 2).tolist()}, max {np.round(offsets.max(axis=0), 2).tolist()}")
    for fit_offsets, name in ((False, "no offsets"), (True, "with offsets")):
        pos = np.concatenate([h[0] for h in held[fit_offsets]])
        rot = np.concatenate([h[1] for h in held[fit_offsets]])
        print(f"    held-out marker RMS, {name}: {np.sqrt(np.mean(pos ** 2)):.2f} mm / {np.sqrt(np.mean(rot ** 2)):.3f} deg")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__ or "usage: python3 joint_offset_fit.py <samples.json> [...]")
        sys.exit(1)
    main(sys.argv[1:])
