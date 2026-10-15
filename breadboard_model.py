#!/usr/bin/env python3
# -*-coding:utf8-*-
# Shared model for breadboard-only calibration (sim_breadboard_calibration.py and the
# breadboard_eye_in_hand_* tools): Piper FK to the flange, pinhole camera, the MB2424 hole
# grid, and the joint least-squares fit.
#
# A hole p_ij (board frame) is seen by the wrist camera at
#   u = project(K, dist, X^-1 * FK(q_meas + delta)^-1 * B * p_ij)
#   delta: joint zero offsets (true angle = reported + delta), X: flange -> camera,
#   B: base -> board (board frame: origin at the board center, z up, holes at z = 0).
# Units: mm and rad.

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from joint_offset_fit import A, ALPHA, THETA, D   # flange DH set (firmware end_pose)

PITCH = 25.4
BOARD_N = 24
FIT_JOINTS = [1, 2, 3, 4]                          # j2..j5; j1 needs a fixed board yaw

g = (np.arange(BOARD_N) - (BOARD_N - 1) / 2) * PITCH
HOLES = np.array([[x, y, 0.0] for x in g for y in g])  # board frame, origin at the center


def to_T(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:6]
    return T


def inv_T(T):
    Ti = np.zeros_like(T)
    Ti[..., 3, 3] = 1
    R = T[..., :3, :3]
    Rt = np.swapaxes(R, -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -(Rt @ T[..., :3, 3:4])[..., 0]
    return Ti


def fk(q):
    """Batched modified-DH FK to the flange. q: (P, 6) rad -> (P, 4, 4) mm."""
    q = np.atleast_2d(q)
    T = np.eye(4)[None].repeat(len(q), 0)
    for i in range(6):
        ca, sa = np.cos(ALPHA[i]), np.sin(ALPHA[i])
        th = q[:, i] + THETA[i]
        ct, st = np.cos(th), np.sin(th)
        L = np.zeros((len(q), 4, 4))
        L[:, 0, 0], L[:, 0, 1], L[:, 0, 3] = ct, -st, A[i]
        L[:, 1, 0], L[:, 1, 1], L[:, 1, 2], L[:, 1, 3] = st * ca, ct * ca, -sa, -sa * D[i]
        L[:, 2, 0], L[:, 2, 1], L[:, 2, 2], L[:, 2, 3] = st * sa, ct * sa, ca, ca * D[i]
        L[:, 3, 3] = 1
        T = T @ L
    return T


def project(Pc, intr):
    """Camera-frame points (..., 3) -> pixels, radial k1/k2 distortion."""
    fx, fy, cx, cy, k1, k2 = intr
    x = Pc[..., 0] / Pc[..., 2]
    y = Pc[..., 1] / Pc[..., 2]
    r2 = x * x + y * y
    s = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([fx * x * s + cx, fy * y * s + cy], -1)


class Model:
    def __init__(self, delta, X, B, intr):
        self.delta, self.X, self.B, self.intr = np.asarray(delta), np.asarray(X), np.asarray(B), np.asarray(intr)

    def cam_in_board(self, q):
        # board -> camera transform for each pose; j1/j6 gauge freedoms cancel in this frame
        return inv_T(to_T(self.B)) @ fk(q + self.delta) @ to_T(self.X)

    def holes_in_cam(self, q):
        T = inv_T(self.cam_in_board(q))                        # board -> camera
        return np.einsum('pij,nj->pni', T[:, :3, :3], HOLES) + T[:, None, :3, 3]


def b_index(fit_B):
    # fit_B: True = whole board pose, False = none, or indices into (rotvec, t)
    return np.arange(6) if fit_B is True else np.arange(0) if fit_B is False else np.asarray(fit_B)


def pack(m, fit_intr, fit_B=True, joints=FIT_JOINTS):
    p = np.r_[m.delta[joints], m.X, m.B[b_index(fit_B)]]
    return np.r_[p, m.intr] if fit_intr else p


def unpack(p, init, fit_intr, fit_B=True, joints=FIT_JOINTS):
    nj, bi = len(joints), b_index(fit_B)
    delta = np.zeros(6)
    delta[joints] = p[:nj]
    B = init.B.copy()
    B[bi] = p[nj + 6:nj + 6 + len(bi)]
    rest = p[nj + 6 + len(bi):]
    return Model(delta, p[nj:nj + 6], B, rest[:6] if fit_intr else init.intr)


def fit(q_meas, uv, vis, init, fit_intr, loss='linear', f_scale=1.0, fit_B=True, max_nfev=100,
        joints=FIT_JOINTS, max_offset_deg=10.0):
    """fit_B: fit the whole board pose (True), keep it at init.B (False, e.g. a measured board
    position), or fit only some of its components (indices into rotvec + t).
    joints: which offsets to fit; j1 only makes sense once the board yaw is fixed.
    max_offset_deg bounds the joint offsets: real zero offsets are a few degrees at most, and
    without a bound a weakly constrained offset (e.g. j5 with few views) can trade tens of
    degrees against the camera mount rotation and land in a wrong solution."""
    idx = np.nonzero(vis)
    p0 = pack(init, fit_intr, fit_B, joints)
    lim = np.radians(max_offset_deg)
    lo, hi = np.full(len(p0), -np.inf), np.full(len(p0), np.inf)
    lo[:len(joints)], hi[:len(joints)] = -lim, lim
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    def res(p):
        m = unpack(p, init, fit_intr, fit_B, joints)
        Pc = m.holes_in_cam(q_meas)
        return (project(Pc, m.intr)[idx] - uv[idx]).ravel()

    sol = least_squares(res, p0, bounds=(lo, hi), x_scale='jac', method='trf', loss=loss,
                        f_scale=f_scale, max_nfev=max_nfev)
    J = sol.jac
    rms = np.sqrt(np.mean(sol.fun ** 2))
    try:
        cov = np.linalg.inv(J.T @ J)
        std = np.sqrt(np.abs(np.diag(cov)))          # abs: near-singular fits (too few views)
        corr = cov / np.outer(std, std)
    except np.linalg.LinAlgError:
        corr = None
    return unpack(sol.x, init, fit_intr, fit_B, joints), rms, corr, sol


