#!/usr/bin/env python3
# -*-coding:utf8-*-
# Top camera calibration from the optical breadboard holes (Thorlabs MB2424: 24 x 24 grid of
# 1/4"-20 taps on 1" centers) seen by the top-down camera, plus a detection stability test.
# multi_camera_calibration.py runs this as its first step. Holes are found as dark blobs, a planar grid is fitted through them
# (homography grid -> pixels) so every hole gets an integer (i, j) index, and per-hole pixel
# jitter / detection rate are accumulated over frames.
#
# Live (ROS 2, next to handeye_bringup.launch.py):
#   python3 top_camera_calibration.py
#   python3 top_camera_calibration.py --topic /stereo/left/image_rect_color --min-area 30 --max-area 800
# Offline (no ROS needed):
#   python3 top_camera_calibration.py --image board.png
#
# Keys: q quit, r reset stats (and grid labels), s save snapshot + stats,
#       j toggle jitter coloring, l toggle (i, j) labels, b toggle preprocessed (CLAHE) view,
#       e camera extrinsic (camera pose in the robot base frame) from the averaged hole centers
#
# Extrinsic: with the whole board in view, the 24 x 24 board window is the physical board, so
# every hole has a known base-frame position given where the board center is (--board-center,
# --board-z) and assuming the base x/y axes run along the hole rows. The arm sits on the image's
# left edge, so base +x = image right and base +y = image up. solvePnP on the averaged
# (undistorted) centers gives the camera pose. Also offline from a saved snapshot:
#   python3 top_camera_calibration.py --extrinsic-from data/breadboard/<ts>_breadboard_holes.json
#
# Counterbores: the MB2424 also has five 11.5 mm counterbored mounting holes, at the center and
# 11 pitches diagonally out from it, each in the middle of four taps (a half-pitch grid point).
# Lit from above, mostly their through hole shows dark, so they are told apart by that position.
# Their pattern fixes which tracked index is the board corner, so the board window stays right
# when rows of taps are missed (e.g. dark near the arm base); two or more must agree, else the
# window is the block holding the most taps. Their centers also go into the extrinsic.
#
# Hole indices are relative to the first fitted frame (seed hole = (0, 0)) and are carried
# frame to frame by the previous homography. They are not tied to the physical board corner;
# if tracking is lost the labels re-seed and the stats reset.

import argparse
import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import json
import os
import time

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PITCH_MM = 25.4  # MB2424 hole pitch (1")
CB_TOL = 0.2     # max distance of a counterbore from its half-pitch point, in pitches (taps sit 0.5 off)


def counterbore_offsets(b):
    # Lower-left tap of each counterbore, from the board corner: center, then the 4 corners
    # (MB2424: (11, 11), (0, 0), (0, 22), (22, 0), (22, 22))
    c, e = (b - 2) // 2, b - 2
    return ((c, c), (0, 0), (0, e), (e, 0), (e, e))


def timestamp():
    # Same convention as the rest of the package: US Eastern wall-clock time
    try:
        now = datetime.datetime.now(ZoneInfo("America/New_York"))
    except ZoneInfoNotFoundError:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-5)))
    return now.strftime("%Y-%m-%d_%H-%M-%S")


class HoleDetector:
    def __init__(self, args):
        self.args = args
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        p = cv2.SimpleBlobDetector_Params()
        p.minThreshold, p.maxThreshold, p.thresholdStep = 10, 200, 10
        p.filterByColor = True
        p.blobColor = 255 if args.invert else 0   # holes are darker than the anodized top
        p.filterByArea = True
        p.minArea, p.maxArea = args.min_area, args.max_area
        p.filterByCircularity = True
        p.minCircularity = args.min_circularity
        p.filterByConvexity = True
        p.minConvexity = 0.8
        p.filterByInertia = True
        p.minInertiaRatio = 0.4   # allows the ellipse a slightly tilted view makes
        self.blob = cv2.SimpleBlobDetector_create(p)

    def preprocess(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        # Black anodized surface + dark holes = low contrast; local equalization helps a lot
        gray = self.clahe.apply(gray)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def detect(self, gray):
        kps = self.blob.detect(gray)
        if not kps:
            return np.zeros((0, 2)), np.zeros(0)
        pts = np.array([k.pt for k in kps], dtype=np.float64)
        sizes = np.array([k.size for k in kps])
        # Refine centers to sub-pixel on the intensity centroid of a small patch, then (default)
        # on the hole's rim
        pts = self.refine(gray, pts, sizes)
        self.centroid_pts = pts                         # kept for comparison (stats_c)
        if not getattr(self.args, 'no_rim', False):
            pts = self.rim_refine(gray, pts, sizes)
        return pts, sizes

    def rim_refine(self, gray, pts, sizes, n_rays=32):
        """Move each center to the center of the hole's top rim. In a tilted view part of the
        hole's inner wall is visible, so the dark region (and its centroid) is lopsided; the rim,
        where the board surface meets the hole, projects to a clean ellipse whose center is the
        hole center (perspective shift of a 6 mm hole is << 0.1 px). On rays from the center,
        the rim is the outermost strong intensity edge (inner edges come from the wall/threads).
        Blobs whose rim doesn't fit an ellipse cleanly keep their centroid; self.rim_ok counts."""
        self.rim_ok = 0
        if len(pts) == 0:
            return pts
        g = gray.astype(np.float32)
        ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        dx, dy = np.cos(ang), np.sin(ang)
        r0 = sizes / 2
        t = np.linspace(0.3, 2.0, 35)                   # along the ray, in units of r0
        rr = r0[:, None, None] * t[None, None, :]
        xs = pts[:, 0, None, None] + rr * dx[None, :, None]
        ys = pts[:, 1, None, None] + rr * dy[None, :, None]
        prof = map_coordinates(g, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(xs.shape)
        ag = np.abs(np.gradient(prof, axis=2))
        locmax = np.zeros(ag.shape, bool)
        locmax[..., 1:-1] = (ag[..., 1:-1] >= ag[..., :-2]) & (ag[..., 1:-1] >= ag[..., 2:])
        cand = locmax & (ag >= 0.4 * ag.max(axis=2, keepdims=True)) & (ag > 2.0)
        T = len(t)
        idx = np.clip(T - 1 - np.argmax(cand[..., ::-1], axis=2), 1, T - 2)   # outermost edge
        has = cand.any(axis=2)
        a0, a1, a2 = (np.take_along_axis(ag, (idx + d)[..., None], 2)[..., 0] for d in (-1, 0, 1))
        den = a0 - 2 * a1 + a2
        off = np.where(np.abs(den) > 1e-6, 0.5 * (a0 - a2) / np.where(den == 0, 1, den), 0.0)
        tt = t[idx] + np.clip(off, -1, 1) * (t[1] - t[0])
        ex = pts[:, 0, None] + r0[:, None] * tt * dx
        ey = pts[:, 1, None] + r0[:, None] * tt * dy
        out = pts.copy()
        for n in range(len(pts)):
            P = np.c_[ex[n], ey[n]][has[n]]
            c = self.fit_rim(P, pts[n], r0[n])
            if c is not None:
                out[n] = c
                self.rim_ok += 1
        return out

    @staticmethod
    def fit_rim(P, c0, r0):
        # Ellipse through the rim points, dropping rays that hit something else; None if unclean
        for _ in range(3):
            if len(P) < 10:
                return None
            (cx, cy), (A, B), th = cv2.fitEllipse(P.astype(np.float32))
            a, b = A / 2, B / 2
            if not (0.4 * r0 < min(a, b) and max(a, b) < 2.2 * r0):
                return None
            ct, st = np.cos(np.radians(th)), np.sin(np.radians(th))
            u = (P[:, 0] - cx) * ct + (P[:, 1] - cy) * st
            v = -(P[:, 0] - cx) * st + (P[:, 1] - cy) * ct
            res = np.abs(np.sqrt((u / a) ** 2 + (v / b) ** 2) - 1) * 0.5 * (a + b)
            keep = res < max(0.75, 2.5 * np.median(res))
            if keep.all():
                break
            P = P[keep]
        if np.sqrt(np.mean(res ** 2)) > max(0.6, 0.12 * r0) or np.hypot(cx - c0[0], cy - c0[1]) > 0.6 * r0:
            return None
        return np.array([cx, cy])

    def refine(self, gray, pts, sizes):
        out = pts.copy()
        h, w = gray.shape
        for n, ((x, y), s) in enumerate(zip(pts, sizes)):
            r = int(max(3, s * 0.75))
            x0, y0, x1, y1 = int(x) - r, int(y) - r, int(x) + r + 1, int(y) + r + 1
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue
            patch = gray[y0:y1, x0:x1].astype(np.float64)
            # Darkness weights (or brightness with --invert), background subtracted
            wgt = (patch - patch.min()) if self.args.invert else (patch.max() - patch)
            wgt = np.clip(wgt - 0.5 * wgt.max(), 0, None)
            if wgt.sum() <= 0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            out[n] = [(wgt * xx).sum() / wgt.sum(), (wgt * yy).sum() / wgt.sum()]
        return out


class GridFitter:
    """Assigns integer grid indices to blob centers via a homography grid(i, j) -> pixel."""

    def __init__(self, tol=0.3, min_holes=8, board_size=24):
        self.tol = tol            # max |grid residual| (in pitches) to accept a blob as a hole
        self.min_holes = min_holes
        self.board_size = board_size  # holes per side; 0 = don't restrict to a board window
        self.H = None             # carried between frames so labels stay consistent
        self.window = None        # (i0, j0) lower corner of the board window, in tracked indices
        self.reseeded = False     # last fit() replaced the tracked labels
        self.cb_pts = np.zeros((0, 2))  # counterbore candidates (undistorted px), set before fit()
        self.cb = {}              # counterbore blob per lower-left tap index, for the last H
        self.window_cbs = 0       # counterbores that placed the window (0: placed by tap count)
        self.forced_window = None # board_window() uses this instead of counting, when set

    def reset(self):
        self.H = None

    def seed(self, pts):
        # Lattice spacing + orientation from nearest-neighbour vectors
        tree = cKDTree(pts)
        d, nn = tree.query(pts, k=5)
        spacing = np.median(d[:, 1])
        vecs = (pts[nn[:, 1:]] - pts[:, None, :]).reshape(-1, 2)
        lens = np.linalg.norm(vecs, axis=1)
        vecs = vecs[np.abs(lens - spacing) < 0.25 * spacing]
        if len(vecs) < 4:
            return None
        # Square lattice: directions repeat every 90 deg, so average angle*4 on the circle
        ang = np.arctan2(vecs[:, 1], vecs[:, 0])
        theta = np.angle(np.exp(4j * ang).mean()) / 4
        u = spacing * np.array([np.cos(theta), np.sin(theta)])
        v = spacing * np.array([-np.sin(theta), np.cos(theta)])
        A = np.column_stack([u, v])
        # Seed at a blob near the centroid; affine labels are fine only close to it. The blob
        # nearest it can be off the tap grid (the center counterbore), so try the next ones too.
        best = None                             # (fraction on the grid, H)
        for n in np.argsort(np.linalg.norm(pts - pts.mean(0), axis=1))[:5]:
            c = pts[n]
            near = np.linalg.norm(pts - c, axis=1) < 3.5 * spacing
            g = np.linalg.solve(A, (pts[near] - c).T).T
            gi = np.round(g)
            ok = np.abs(g - gi).max(1) < self.tol
            if ok.sum() < 4:
                continue
            H, _ = cv2.findHomography(gi[ok], pts[near][ok], 0)
            if ok.mean() >= 0.6:
                return H
            if best is None or ok.mean() > best[0]:
                best = (ok.mean(), H)
        return None if best is None else best[1]

    def assign(self, pts, H):
        # pixel -> grid through H^-1, round to the nearest hole, keep the close ones
        g = cv2.perspectiveTransform(pts[None], np.linalg.inv(H))[0]
        # A degenerate homography can send blobs to absurd grid coordinates; those are no holes
        # (and would blow up the board window's index table)
        sane = np.all(np.isfinite(g) & (np.abs(g) < 1000), axis=1)
        g = np.where(sane[:, None], g, 0.5)                    # 0.5: never within tol of a node
        gi = np.round(g).astype(int)
        res = np.abs(g - gi).max(1)
        ok = sane & (res < self.tol)
        # One blob per hole: keep the best residual if two blobs round to the same index
        best = {}
        for n in np.flatnonzero(ok):
            key = (gi[n, 0], gi[n, 1])
            if key not in best or res[n] < res[best[key]]:
                best[key] = n
        self.counterbores(H)
        return self.board_window(best)

    def counterbores(self, H):
        """Counterbore candidates on a half-pitch grid point -> self.cb {lower-left tap index: blob}."""
        self.cb = {}
        if len(self.cb_pts) == 0:
            return
        g = cv2.perspectiveTransform(self.cb_pts[None], np.linalg.inv(H))[0]
        sane = np.all(np.isfinite(g) & (np.abs(g) < 1000), axis=1)
        g = np.where(sane[:, None], g, 0.0)
        lo = np.floor(g).astype(int)
        res = np.abs(g - lo - 0.5).max(1)
        res = np.where(sane, res, np.inf)
        for n in np.flatnonzero(res < CB_TOL):
            key = (lo[n, 0], lo[n, 1])
            if key not in self.cb or res[n] < res[self.cb[key]]:
                self.cb[key] = n

    def board_window(self, best):
        # The grid extrapolates past the board, so blobs on the floor/arm can land on a grid
        # node by chance. The real board is one board_size x board_size block of indices:
        # keep only the block holding the most holes.
        if not self.board_size or not best:
            return best
        b = self.board_size
        if self.forced_window is not None:
            i0, j0 = self.window = self.forced_window
            return {k: n for k, n in best.items() if i0 <= k[0] < i0 + b and j0 <= k[1] < j0 + b}
        gi = np.array(list(best.keys()))
        if np.ptp(gi, axis=0).max() > 10 * self.board_size:
            # Shouldn't happen after assign()'s guard; keep the table small regardless
            near = np.all(np.abs(gi - np.median(gi, axis=0)) <= 2 * self.board_size, axis=1)
            best = {k: n for (k, n), ok in zip(best.items(), near) if ok}
            gi = gi[near]
        lo, win = self.window_counts(gi)
        i0, j0 = np.unravel_index(np.argmax(win), win.shape) + lo
        self.window = (int(i0), int(j0))
        return {k: n for k, n in best.items()
                if i0 <= k[0] < i0 + b and j0 <= k[1] < j0 + b}

    def window_counts(self, gi):
        """Hole count of every board_size window -> (lower corner of win[0, 0], win)."""
        b = self.board_size
        lo = gi.min(0) - b + 1
        occ = np.zeros(gi.max(0) - lo + 1, dtype=np.int32)
        occ[gi[:, 0] - lo[0], gi[:, 1] - lo[1]] = 1
        # 2D prefix sum -> hole count of every board_size window (lower corner at each cell)
        c = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return lo, c[b:, b:] - c[:-b, b:] - c[b:, :-b] + c[:-b, :-b]

    def counterbore_window(self, pts, H):
        """Board window from the counterbores on the converged grid: ((i0, j0), votes), or None
        unless 2+ agree (one could be debris at a half-pitch point). Each counterbore votes once
        for each window it fits; the tap count breaks ties (center + one corner fits two)."""
        b = self.board_size
        if not b or len(self.cb_pts) == 0:
            return None
        self.board_size = 0                     # every tap on the grid, not just the current window
        try:
            best = self.assign(pts, H)
        finally:
            self.board_size = b
        if not best or len(self.cb) < 2:
            return None
        gi = np.array(list(best.keys()))
        lo, win = self.window_counts(gi)
        hit = np.zeros(win.shape, dtype=np.int64)
        for key in self.cb:
            for off in counterbore_offsets(b):
                w = (key[0] - off[0] - lo[0], key[1] - off[1] - lo[1])
                if 0 <= w[0] < hit.shape[0] and 0 <= w[1] < hit.shape[1]:
                    hit[w] += 1
        if hit.max() < 2:
            return None
        w = np.unravel_index(np.argmax(hit * (b * b + 1) + win), win.shape)
        return (int(w[0] + lo[0]), int(w[1] + lo[1])), int(hit[w])

    def grow(self, pts, H):
        """Assign holes from a starting homography, refitting outward. -> (H, best, rms px)."""
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            return None, {}, np.inf
        prev = -1
        for _ in range(10):
            best = self.assign(pts, H)
            if len(best) < 4 or len(best) == prev:
                break
            prev = len(best)
            idx = np.array(list(best.keys()), dtype=np.float64)
            H_new, inl = cv2.findHomography(idx, pts[list(best.values())], cv2.RANSAC, 3.0)
            if not self.sane(H_new, idx):
                break
            H = H_new
        best = self.assign(pts, H)
        if len(best) < self.min_holes or not self.sane(H, np.array(list(best.keys()), dtype=np.float64)):
            return None, {}, np.inf
        # The assignment tolerance is loose (0.3 pitch) so the grid can grow; blobs off the board
        # (paper, screws, cables) can still land on a grid node by chance. Keep only blobs that
        # sit on the refitted grid to within ~5% of a pitch.
        idx = np.array(list(best.keys()), dtype=np.float64)
        img = pts[list(best.values())]
        tol = max(2.0, 0.05 * self.pitch_px(H, idx))
        H_in, inl = cv2.findHomography(idx, img, cv2.RANSAC, tol)
        if self.sane(H_in, idx) and inl.sum() >= self.min_holes:
            H = H_in
            best = {k: n for (k, n), ok in zip(best.items(), inl.ravel()) if ok}
            idx, img = idx[inl.ravel() == 1], img[inl.ravel() == 1]
        err = img - cv2.perspectiveTransform(idx[None], H)[0]
        return H, best, float(np.sqrt((err ** 2).sum(1).mean()))

    @classmethod
    def sane(cls, H, idx):
        # Usable grid -> pixel homography: finite, invertible, plausible hole spacing
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            return False
        p = cls.pitch_px(H, idx)
        return bool(np.isfinite(p) and 3 < p < 1000)

    @staticmethod
    def pitch_px(H, idx):
        # Local hole spacing in pixels around the middle of the assigned holes
        c = np.median(idx, axis=0)
        q = cv2.perspectiveTransform(np.array([[c, c + [1, 0], c + [0, 1]]]), H)[0]
        return 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[0]))

    @staticmethod
    def relabel(H, best, ref):
        """Re-express a fresh fit in the labels of a reference fit (ref: blob -> key), using the
        blobs both assigned: the grids differ by a 90-deg rotation + integer shift, found by
        majority. Keeps hole labels stable across a re-seed. None if they don't agree."""
        inv = {n: k for k, n in ref.items()}
        common = [(np.array(k), np.array(inv[n])) for k, n in best.items() if n in inv]
        if len(common) < 4:
            return None
        a = np.array([c[0] for c in common])
        b = np.array([c[1] for c in common])
        for k in range(4):
            R = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), k)
            d, cnt = np.unique(b - a @ R.T, axis=0, return_counts=True)
            if cnt.max() >= 0.8 * len(common):
                M = np.eye(3)
                M[:2, :2], M[:2, 2] = R, d[np.argmax(cnt)]       # fresh grid -> reference grid
                new = {tuple((M[:2, :2] @ np.array(kk) + M[:2, 2]).astype(int)): n for kk, n in best.items()}
                return H @ np.linalg.inv(M), new
        return None

    def fit(self, pts):
        self.reseeded = False
        if len(pts) < self.min_holes:
            self.H, self.cb = None, {}
            return None, {}
        if self.H is not None and not np.all(np.isfinite(self.H)):
            self.H = None
        tracked = self.grow(pts, self.H) if self.H is not None else (None, {}, np.inf)
        H, best, rms = tracked
        # A moving camera can leave the previous frame's grid far off, and tracking then locks
        # onto a wrong lattice. Unless tracking already explains most blobs, also fit from scratch
        # and keep that if it explains clearly more holes (or fits much better).
        if H is None or len(best) < 0.8 * len(pts):
            H0 = self.seed(pts)
            fresh = self.grow(pts, H0) if H0 is not None else (None, {}, np.inf)
            if fresh[0] is not None and (H is None or len(fresh[1]) > len(best) + max(3, 0.1 * len(best))
                                         or fresh[2] < 0.5 * rms):
                same = self.relabel(fresh[0], fresh[1], best) if best else None
                if same is not None:
                    H, best = same
                else:
                    H, best = fresh[0], fresh[1]
                    self.reseeded = self.H is not None          # labels changed
        if H is None:
            self.H, self.cb = None, {}
            return None, {}
        # The tap-count window can be off when rows are missed (dark near the arm base) or blobs
        # beyond the board sit on grid points; the counterbores, once seen on the converged grid,
        # fix it, and the grid is refitted inside their window
        cbw = self.counterbore_window(pts, H)
        self.window_cbs = 0
        if cbw is not None:
            self.forced_window = cbw[0]
            try:
                H2, best2, _ = self.grow(pts, H)
            finally:
                self.forced_window = None
            if H2 is not None:
                H, best, self.window_cbs = H2, best2, cbw[1]
        self.forced_window = cbw[0] if self.window_cbs else None
        try:
            self.assign(pts, H)                                 # leaves self.window, self.cb for this H
        finally:
            self.forced_window = None
        self.H = H
        return H, best


class Stats:
    """Per-hole running mean / variance of the pixel center (Welford) + detection count."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = 0
        self.n, self.mean, self.m2 = {}, {}, {}
        self.rms_hist, self.count_hist = [], []

    def update(self, best, pts, rms):
        self.frames += 1
        self.rms_hist.append(rms)
        self.count_hist.append(len(best))
        for key, i in best.items():
            p = pts[i]
            if key not in self.n:
                self.n[key], self.mean[key], self.m2[key] = 0, np.zeros(2), np.zeros(2)
            self.n[key] += 1
            d = p - self.mean[key]
            self.mean[key] += d / self.n[key]
            self.m2[key] += d * (p - self.mean[key])

    def std(self, key):
        # Combined x/y std in px; needs a few samples to mean anything
        if self.n.get(key, 0) < 3:
            return None
        return float(np.sqrt(self.m2[key].sum() / (self.n[key] - 1)))

    def summary(self):
        stds = [s for s in (self.std(k) for k in self.n) if s is not None]
        rates = [self.n[k] / self.frames for k in self.n] if self.frames else []
        return {
            'frames': self.frames,
            'holes_seen': len(self.n),
            'holes_per_frame_mean': float(np.mean(self.count_hist)) if self.count_hist else 0.0,
            'holes_per_frame_min': int(np.min(self.count_hist)) if self.count_hist else 0,
            'detection_rate_median': float(np.median(rates)) if rates else 0.0,
            'jitter_px_median': float(np.median(stds)) if stds else None,
            'jitter_px_p95': float(np.percentile(stds, 95)) if stds else None,
            'fit_rms_px_mean': float(np.mean(self.rms_hist)) if self.rms_hist else None,
        }


def jitter_color(s):
    # green < 0.1 px, yellow ~0.3 px, red >= 0.5 px
    if s is None:
        return (200, 200, 200)
    t = float(np.clip(s / 0.5, 0, 1))
    return (0, int(255 * min(1, 2 * (1 - t))), int(255 * min(1, 2 * t)))


class BreadboardHoleTest:
    def __init__(self, args):
        self.args = args
        self.detector = HoleDetector(args)
        self.fitter = GridFitter(tol=args.grid_tol, min_holes=args.min_holes, board_size=args.board_size)
        self.stats = Stats()
        self.stats_c = Stats()          # same holes, centroid centers (rim refinement A/B)
        self.stats_cb = Stats()         # counterbores on the board, keyed by grid position (i + 0.5, j + 0.5)
        self.K = self.D = None
        self.show_jitter, self.show_labels, self.show_thresh = False, False, False
        self.last = None
        self.last_reseed_msg = -np.inf
        os.makedirs(args.save_path, exist_ok=True)

    def set_camera_info(self, K, D):
        if self.K is None:
            print(f"camera_info: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}, D={np.round(D, 4).tolist()}")
        self.K, self.D = K, D

    def undistort(self, pts):
        # Lens distortion bends the grid and inflates the homography residual; remove it
        # when camera_info is available (P=K keeps the result in pixel units)
        if self.K is None or not np.any(self.D) or len(pts) == 0:
            return pts
        return cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.D, P=self.K).reshape(-1, 2)

    def process(self, bgr):
        gray = self.detector.preprocess(bgr)
        raw, sizes = self.detector.detect(gray)
        cen = self.detector.centroid_pts if len(raw) else raw
        if len(sizes) >= self.args.min_holes:
            # Holes are all the same size; partly occluded ones (arm edge) come out smaller
            # with a shifted center, so drop blobs far from the size of their neighbors (local,
            # because perspective in an oblique view changes the hole size across the image)
            _, nn = cKDTree(raw).query(raw, k=min(7, len(raw)))
            med = np.median(sizes[nn[:, 1:]], axis=1)
            keep = (sizes > 0.75 * med) & (sizes < 1.33 * med)
            # Counterbore candidates: under the top light usually only the through hole in the
            # middle is dark, so most look like taps (some up to ~1.6x bigger). Only their place,
            # a half-pitch grid point, sets them apart; the grid fit checks that.
            cand = (sizes > 0.75 * med) & (sizes < 2.2 * med)
            cb_raw, cb_sizes = raw[cand], sizes[cand]
            raw, sizes, cen = raw[keep], sizes[keep], cen[keep]
        else:
            cb_raw, cb_sizes = np.zeros((0, 2)), np.zeros(0)
        self.fitter.cb_pts = self.undistort(cb_raw)
        pts = self.undistort(raw)
        was_tracking = self.fitter.H is not None
        H, best = self.fitter.fit(pts)
        if (H is None and was_tracking) or self.fitter.reseeded:
            # Labels are only meaningful while tracked; start over (message rate-limited: a moving
            # wrist camera re-seeds often)
            now = time.monotonic()
            if now - self.last_reseed_msg > 5.0:
                print("grid tracking lost -> labels re-seeded, stats reset")
                self.last_reseed_msg = now
            self.stats.reset()
            self.stats_c.reset()
            self.stats_cb.reset()
        rms = None
        if H is not None:
            idx = np.array(list(best.keys()), dtype=np.float64)
            proj = cv2.perspectiveTransform(idx[None], H)[0]
            err = pts[list(best.values())] - proj
            rms = float(np.sqrt((err ** 2).sum(1).mean()))
            self.stats.update(best, pts, rms)
            self.stats_c.update(best, self.undistort(cen), rms)
            self.stats_cb.update(self.board_counterbores(), self.fitter.cb_pts, rms)
        self.last = dict(raw=raw, sizes=sizes, H=H, best=best, rms=rms)
        self.cb_draw = [(cb_raw[n], cb_sizes[n]) for n in self.board_counterbores().values()]
        return self.draw(bgr, gray, raw, sizes, H, best, rms)

    def board_counterbores(self):
        # This frame's counterbores inside the board window: {(i + 0.5, j + 0.5): blob}
        w, b = self.fitter.window, self.args.board_size
        if self.fitter.H is None or w is None or not b:
            return {}
        return {(k[0] + 0.5, k[1] + 0.5): n for k, n in self.fitter.cb.items()
                if w[0] <= k[0] < w[0] + b - 1 and w[1] <= k[1] < w[1] + b - 1}

    def draw(self, bgr, gray, raw, sizes, H, best, rms):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if self.show_thresh else bgr.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        assigned = set(best.values())
        # Unassigned blobs (outliers: screws, dirt, the arm) in magenta
        for n, (p, s) in enumerate(zip(raw, sizes)):
            if n not in assigned:
                cv2.circle(vis, tuple(np.round(p).astype(int)), int(s / 2) + 2, (255, 0, 255), 1)
        # Counterbores found on the board's half-pitch points, in orange
        for p, s in getattr(self, 'cb_draw', []):
            cv2.circle(vis, tuple(np.round(p).astype(int)), int(s / 2) + 4, (0, 165, 255), 3)
        if H is not None:
            keys = list(best.keys())
            gi = np.array(keys)
            # Predicted holes inside the seen index range that had no blob this frame
            lo, hi = gi.min(0), gi.max(0)
            grid = np.array([(i, j) for i in range(lo[0], hi[0] + 1) for j in range(lo[1], hi[1] + 1)
                             if (i, j) not in best], dtype=np.float64)
            if len(grid):
                proj = self.distort(cv2.perspectiveTransform(grid[None], H)[0])
                for p in proj:
                    c = tuple(np.round(p).astype(int))
                    cv2.drawMarker(vis, c, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
            for key, n in best.items():
                c = tuple(np.round(raw[n]).astype(int))
                color = jitter_color(self.stats.std(key)) if self.show_jitter else (0, 255, 0)
                cv2.circle(vis, c, int(sizes[n] / 2) + 2, color, 2)
                if self.show_labels:
                    cv2.putText(vis, f"{key[0]},{key[1]}", (c[0] + 5, c[1] - 5),
                                cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            # Grid axes from the seed hole so orientation/labels are visible
            o, x, y = self.distort(cv2.perspectiveTransform(
                np.array([[[0, 0], [2, 0], [0, 2]]], dtype=np.float64), H)[0])
            cv2.arrowedLine(vis, tuple(np.round(o).astype(int)), tuple(np.round(x).astype(int)), (0, 0, 255), 2)
            cv2.arrowedLine(vis, tuple(np.round(o).astype(int)), tuple(np.round(y).astype(int)), (0, 255, 0), 2)

        s = self.stats.summary()
        jit = f"{s['jitter_px_median']:.3f}/{s['jitter_px_p95']:.3f}" if s['jitter_px_median'] is not None else "-"
        lines = [
            f"blobs {len(raw)}  holes {len(best)}  fit rms {rms:.2f}px" if rms is not None
            else f"blobs {len(raw)}  holes 0  (no grid fit)",
            f"frames {s['frames']}  holes seen {s['holes_seen']}  per-frame mean {s['holes_per_frame_mean']:.1f} min {s['holes_per_frame_min']}",
            f"det rate med {s['detection_rate_median']:.2f}  jitter med/p95 {jit} px",
            f"counterbores {len(self.board_counterbores())}/5  board window from "
            + (f"{self.fitter.window_cbs} counterbores" if self.fitter.window_cbs else "tap count (need 2+ counterbores)"),
        ]
        if len(sizes):
            pitch = f"  pitch {self.pitch_px(H):.1f}px" if H is not None else ""
            lines.append(f"blob dia med {np.median(sizes):.1f}px{pitch}  [q r s e j l b]")
        for k, t in enumerate(lines):
            cv2.putText(vis, t, (10, 25 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(vis, t, (10, 25 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return vis

    def distort(self, pts):
        # Map undistorted pixel coordinates back onto the raw image for drawing
        if self.K is None or not np.any(self.D) or len(pts) == 0:
            return pts
        n = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, None)  # normalized, no distortion
        n3 = cv2.convertPointsToHomogeneous(n)
        out, _ = cv2.projectPoints(n3, np.zeros(3), np.zeros(3), self.K, self.D)
        return out.reshape(-1, 2)

    def pitch_px(self, H):
        c = cv2.perspectiveTransform(np.array([[[0, 0], [1, 0], [0, 1]]], dtype=np.float64), H)[0]
        return 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[0]))

    def key(self, k, vis):
        if k == ord('r'):
            self.stats.reset()
            self.stats_c.reset()
            self.stats_cb.reset()
            self.fitter.reset()
            print("stats reset, grid re-seeded")
        elif k == ord('j'):
            self.show_jitter = not self.show_jitter
        elif k == ord('l'):
            self.show_labels = not self.show_labels
        elif k == ord('b'):
            self.show_thresh = not self.show_thresh
        elif k == ord('s'):
            self.save(vis)
        elif k == ord('e'):
            self.extrinsic()
        return k != ord('q')

    def extrinsic(self):
        """Camera pose in base from the averaged holes; the saved result, or None."""
        if self.K is None or self.fitter.H is None or self.fitter.window is None:
            print("extrinsic needs camera_info, a grid fit and --board-size > 0")
            return None
        # Averaged centers only, from holes seen in most frames (drops flickering edge holes)
        holes = [(k, self.stats.mean[k]) for k in self.stats.n
                 if self.stats.n[k] >= 0.5 * self.stats.frames]
        cbs = [(k, self.stats_cb.mean[k]) for k in self.stats_cb.n
               if self.stats_cb.n[k] >= 0.5 * self.stats_cb.frames]
        return board_extrinsic(holes + cbs, self.K, self.fitter.H, self.fitter.window, self.args,
                               window_cbs=self.fitter.window_cbs)

    def save(self, vis):
        base = os.path.join(self.args.save_path, f"{timestamp()}_breadboard_holes")
        cv2.imwrite(base + ".png", vis)
        holes = []
        for key in sorted(self.stats.n):
            holes.append({'index': list(map(int, key)), 'mean_px': self.stats.mean[key].tolist(),
                          'std_px': self.stats.std(key), 'detected': self.stats.n[key]})
        # Counterbores have half-integer indices; --extrinsic-from takes them like taps
        for key in sorted(self.stats_cb.n):
            holes.append({'index': list(map(float, key)), 'mean_px': self.stats_cb.mean[key].tolist(),
                          'std_px': self.stats_cb.std(key), 'detected': self.stats_cb.n[key]})
        out = {'summary': self.stats.summary(), 'pitch_mm': PITCH_MM,
               'H_grid_to_px': self.fitter.H.tolist() if self.fitter.H is not None else None,
               'board_window': self.fitter.window, 'window_counterbores': self.fitter.window_cbs,
               'K': self.K.tolist() if self.K is not None else None,
               'D': self.D.tolist() if self.D is not None else None,
               'params': vars(self.args), 'holes': holes}
        with open(base + ".json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {base}.png/.json")
        print(json.dumps(out['summary'], indent=2))


def board_extrinsic(holes, K, H, window, args, window_cbs=0):
    """holes: [(tracked (i, j), undistorted mean px)], counterbores at (i + 0.5, j + 0.5).
    window_cbs: counterbores that placed the window. Prints + saves the camera pose in base;
    returns the saved result (+ 'file': where), or None if it wasn't good enough to save."""
    b = args.board_size
    i0, j0 = window
    holes = [(k, p) for k, p in holes if i0 <= k[0] < i0 + b and j0 <= k[1] < j0 + b]
    gi = np.array([k for k, _ in holes], dtype=np.float64) - [i0, j0]   # 0..b-1 on the board
    img = np.array([p for _, p in holes], dtype=np.float64)
    if len(holes) < 20:
        print(f"only {len(holes)} board holes, not computing extrinsic")
        return None
    n_cb = int(np.sum(np.any(gi % 1 != 0, axis=1)))
    print(f"counterbores used: {n_cb}; board window placed by "
          + (f"{window_cbs} counterbores" if window_cbs else "tap count"))
    # Placed by tap count, the window only equals the physical board if every edge row/column
    # was seen; a fully hidden edge shifts every hole position by one pitch
    for name, sel in ([] if window_cbs else (("i=0", gi[:, 0] == 0), (f"i={b-1}", gi[:, 0] == b - 1),
                      ("j=0", gi[:, 1] == 0), (f"j={b-1}", gi[:, 1] == b - 1))):
        if sel.sum() < 3:
            print(f"WARNING: board edge {name} has {sel.sum()} holes; board window may be off by one pitch")

    # Which grid axis is base x (image right) / base y (image up), measured at the board center
    c = np.array([[[i0 + (b - 1) / 2, j0 + (b - 1) / 2], [i0 + (b + 1) / 2, j0 + (b - 1) / 2],
                   [i0 + (b - 1) / 2, j0 + (b + 1) / 2]]])
    o, di, dj = cv2.perspectiveTransform(c, H)[0]
    dirs = np.array([di - o, dj - o])
    ax = int(np.argmax(np.abs(dirs[:, 0])))          # grid axis most along image u
    sx = np.sign(dirs[ax, 0])
    ay = 1 - ax
    sy = -np.sign(dirs[ay, 1])                       # image up = -v
    g = gi - (b - 1) / 2                             # board center sits between holes 11 and 12
    cx, cy = args.board_center
    obj = np.column_stack([cx + PITCH_MM * sx * g[:, ax], cy + PITCH_MM * sy * g[:, ay],
                           np.full(len(g), args.board_z)])

    # Centers are already undistorted into K pixels, so no distortion here. IPPE is the planar
    # solver; refine with LM on all points.
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = proj.reshape(-1, 2) - img
    rms = float(np.sqrt((err ** 2).sum(1).mean()))
    R_b2c = Rotation.from_rotvec(rvec.ravel()).as_matrix()
    R = R_b2c.T                                     # camera -> base, as in the hand-eye results
    t = -R @ tvec.ravel()                           # camera position in base, mm
    tilt = np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))   # optical axis vs straight down
    if R[2, 2] > 0:
        print("WARNING: camera z points up in base frame -> axis mapping is wrong")
    if rms > args.max_extrinsic_rms:
        # Holes matched to the wrong board positions (window off, mixed labels, debris on the grid):
        # the pose is meaningless however good it looks, so don't write it
        print(f"holes used: {len(holes)}, reprojection rms: {rms:.3f} px -> NOT saved (limit "
              f"{args.max_extrinsic_rms} px). Press r, let it collect again, and save a snapshot (s) for "
              f"--extrinsic-from if it keeps failing.")
        return None

    print(f"holes used: {len(holes)}, reprojection rms: {rms:.3f} px")
    print(f"camera position in base: {np.round(t, 1).tolist()} mm (board_z = {args.board_z} mm)")
    print(f"camera rpy (xyz, deg): {np.round(Rotation.from_matrix(R).as_euler('xyz', degrees=True), 2).tolist()}, "
          f"optical axis {tilt:.2f} deg off vertical")
    # Same layout as handeye_calibration_record output, so handeye_touch_test can read it
    out = {'position': (t / 1000).tolist(), 'orientation': Rotation.from_matrix(R).as_quat().tolist(),
           'rpy': [Rotation.from_matrix(R).as_euler('xyz').tolist()],
           'breadboard': {'reproj_rms_px': rms, 'holes': len(holes), 'counterbores': n_cb,
                          'window_counterbores': window_cbs, 'tilt_deg': float(tilt),
                          'board_center_mm': list(args.board_center), 'board_z_mm': args.board_z,
                          'pitch_mm': PITCH_MM, 'K': K.tolist()}}
    path = os.path.join(args.data_save_path, 'eye_to_hand')
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f"{timestamp()}_breadboard_calibration.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {fname}")
    return dict(out, file=fname)


def run_image(app, path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    vis = app.process(bgr)
    print(json.dumps(app.stats.summary(), indent=2))
    while True:
        cv2.imshow("breadboard holes", vis)
        k = cv2.waitKey(0) & 0xFF
        if not app.key(k, vis):
            break
        # Re-run on the same image so toggles redraw (stats count it as another frame)
        vis = app.process(bgr)


def to_bgr(msg):
    # sensor_msgs/Image -> numpy, minimal conversion so cv_bridge is not required
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('rgb8', 'bgr8'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img.copy()
    if msg.encoding in ('mono8', '8UC1'):
        return buf.reshape(msg.height, msg.step)[:, :msg.width].copy()
    raise ValueError(f"unsupported encoding {msg.encoding}")


def run_ros(app, args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo

    rclpy.init()
    node = Node("top_camera_calibration")
    latest = {}
    node.create_subscription(Image, args.topic, lambda m: latest.__setitem__('img', m), qos_profile_sensor_data)
    if args.camera_info:
        node.create_subscription(CameraInfo, args.camera_info, lambda m: app.set_camera_info(
            np.array(m.k).reshape(3, 3), np.array(m.d)), qos_profile_sensor_data)
    print(f"topic: {args.topic}, camera_info: {args.camera_info or '(none, no undistortion)'}")
    vis = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            msg = latest.pop('img', None)   # always process the newest frame, drop backlog
            if msg is not None:
                vis = app.process(to_bgr(msg))
            if vis is not None:
                cv2.imshow("breadboard holes", vis)
            k = cv2.waitKey(1) & 0xFF
            if k != 255 and not app.key(k, vis):
                break
    finally:
        print(json.dumps(app.stats.summary(), indent=2))
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


def add_board_args(ap):
    """Hole detector + board extrinsic options (shared with multi_camera_calibration.py)."""
    ap.add_argument('--min-area', type=float, default=15, help="blob area px^2; tune from 'blob dia' in the HUD")
    ap.add_argument('--max-area', type=float, default=2500)
    ap.add_argument('--min-circularity', type=float, default=0.6)
    ap.add_argument('--invert', action='store_true', help="holes appear brighter than the surface")
    ap.add_argument('--no-rim', action='store_true', help="centroid centers only (skip rim ellipse refinement)")
    ap.add_argument('--grid-tol', type=float, default=0.3, help="max grid residual in pitches")
    ap.add_argument('--min-holes', type=int, default=8)
    ap.add_argument('--board-size', type=int, default=24, help="holes per side (MB2424: 24); 0 = no board window")
    ap.add_argument('--data-save-path', default='./data', help="calibration goes to <this>/eye_to_hand/")
    ap.add_argument('--board-center', type=float, nargs=2, default=[349.8, 0.0],
                    help="board center (x, y) in the robot base frame, mm")
    ap.add_argument('--board-z', type=float, default=0.7, help="board top surface z in the base frame, mm (12.7 mm board - 12 mm base mount)")
    ap.add_argument('--max-extrinsic-rms', type=float, default=2.0, help="px; refuse to save a worse extrinsic")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', help="run on a single image file instead of ROS")
    ap.add_argument('--topic', default='/stereo/left/image_rect_color')
    ap.add_argument('--camera-info', default='/stereo/left/camera_info', help="'' to skip undistortion")
    add_board_args(ap)
    ap.add_argument('--save-path', default='./data/breadboard')
    ap.add_argument('--extrinsic-from', help="compute the extrinsic from a saved snapshot json and exit")
    args = ap.parse_args()

    if args.extrinsic_from:
        with open(args.extrinsic_from) as f:
            snap = json.load(f)
        if snap.get('board_window') is None or snap.get('K') is None:
            raise SystemExit("snapshot has no board_window/K (saved by an older version or without camera_info)")
        frames = snap['summary']['frames']
        holes = [(tuple(h['index']), np.array(h['mean_px'])) for h in snap['holes']
                 if h['detected'] >= 0.5 * frames]
        board_extrinsic(holes, np.array(snap['K']), np.array(snap['H_grid_to_px']),
                        tuple(snap['board_window']), args, snap.get('window_counterbores', 0))
        return

    app = BreadboardHoleTest(args)
    if args.image:
        run_image(app, args.image)
    else:
        run_ros(app, args)


if __name__ == '__main__':
    main()
