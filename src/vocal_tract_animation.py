from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import bezier


def cubic_bezier(p0, p1, p2, p3, n: int = 32) -> np.ndarray:
    nodes = np.asfortranarray([
        [p0[0], p1[0], p2[0], p3[0]],
        [p0[1], p1[1], p2[1], p3[1]],
    ])
    curve = bezier.Curve(nodes, degree=3)
    s_vals = np.linspace(0.0, 1.0, n)
    points = curve.evaluate_multi(s_vals).T  # shape (n, 2)
    return points

ArtDict = Dict[str, np.ndarray]

# Helper to compute average lip distance
def _mean_lip_distance(traj: Dict[str, np.ndarray]) -> float:
    """Return the average UL–LL Euclidean distance (cm) across all frames."""
    ul, ll = traj["UL"], traj["LL"]
    return float(np.mean(np.linalg.norm(ll - ul, axis=1)))

###############################################################################
# Coordinate normalisation
###############################################################################

def _affine_from_frame(art: ArtDict, target_lip_dist: float = 1.6):
    """
    Compute (R, t, s) so that:

      * the lower incisor (LI) maps to (0, 0)
      * the UL->LL vector becomes vertical (positive y down)
      * the UL-LL distance is `target_lip_dist`

    Returns rotation matrix R (2x2), translation vector t (2,), scale s.
    """
    ul, ll, li = art["UL"], art["LL"], art["LI"]

    # 1) translate so LI is the origin  (bite‑plane reference point)
    t = li

    # 2) rotate so the lip‑line (UL→LL) becomes vertical (positive‑y down)
    v   = ll - ul
    phi = np.arctan2(v[1], v[0])        # current angle of UL→LL
    # Rotate so UL→LL ends up pointing **downwards** (−y) in the new frame.
    theta = -np.pi/2 - phi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # 3) scale so the UL–LL distance equals target_lip_dist (≈16 mm default)
    d = np.linalg.norm(v)
    s = target_lip_dist / (d + 1e-9)

    return R, t, s

def _apply_affine(xy: np.ndarray, R: np.ndarray, t: np.ndarray, s: float):
    """Return (xy – t) rotated by R then scaled by s."""
    return (R @ (xy - t).T).T * s


def _rig_points(art: ArtDict) -> Dict[str, np.ndarray]:
    """Compute hidden rig anchors for one frame (heuristic rules)."""
    rig: Dict[str, np.ndarray] = {}

    # Average TB & TT for a smooth mid‑tongue anchor
    rig["mid_tongue"] = 0.5 * (art["TB"] + art["TT"])

    # Tongue root: TD minus 4 mm in y
    rig["root"] = art["TD"] + np.array([0.0, -0.4])

    # Palate mid‑point (static reference after normalisation)
    rig["palate_mid"] = np.array([0.0, 2.0])

    # Pharynx wall behind the root
    rig["pharynx_wall"] = rig["root"] + np.array([-1.2, 0.0])

    # Lip corners: ±6 mm from UL in x
    lip_half = 0.6
    rig["lip_left"]  = art["UL"] + np.array([-lip_half, 0.0])
    rig["lip_right"] = art["UL"] + np.array([ lip_half, 0.0])
    return rig


def _catmull_rom(P: np.ndarray, n: int = 60) -> np.ndarray:
    """
    Return n points on a Catmull‑Rom spline passing through the
    control points P (k×2).  Endpoints are repeated for natural ends.
    """
    P = np.asarray(P)
    if P.shape[0] < 2:
        return P
    # repeat first and last
    pts = np.vstack([P[0], P, P[-1]])
    curve = []
    for i in range(pts.shape[0] - 3):
        p0, p1, p2, p3 = pts[i : i + 4]
        for t in np.linspace(0, 1, n // (pts.shape[0] - 3)):
            t2, t3 = t * t, t * t * t
            c = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2*p0 - 5*p1 + 4*p2 - p3) * t2
                + (-p0 + 3*p1 - 3*p2 + p3) * t3
            )
            curve.append(c)
    return np.vstack(curve)


def _translate_outline(base_outline: np.ndarray,
                       rig: Dict[str, np.ndarray],
                       gap_y: float = 0.3,
                       align_x: bool = True) -> np.ndarray:
    """
    Affine‑translate the static outline so that:

    • the hard‑palate node (index 3) sits `gap_y` cm above the tongue root, and
    • IF `align_x` is True, the front lip point (index 6) aligns with UL.

    This keeps the outline vertically and horizontally centred on the
    current speaker position after normalisation.
    """
    # vertical shift
    dy = (rig["root"][1] + gap_y) - base_outline[3, 1]

    # horizontal shift (bring mouth opening to UL)
    dx = 0.0
    if align_x:
        dx = (0.5 * (rig["lip_left"][0] + rig["lip_right"][0])) - base_outline[6, 0]

    return base_outline + np.array([dx, dy])


def _tongue_surface(art: ArtDict, rig: Dict[str, np.ndarray]) -> np.ndarray:
    # Measured tongue points from back to front
    pts = np.vstack([art["TD"], art["TB"], art["TT"]])
    return _catmull_rom(pts, n=60)


def _palate_surface(rig: Dict[str, np.ndarray]) -> np.ndarray:
    pm = rig["palate_mid"]
    p0 = pm + np.array([-0.8, 0.0])
    p1 = pm + np.array([-0.4, 0.4])
    p2 = pm + np.array([0.4, 0.4])
    p3 = pm + np.array([0.8, 0.0])
    return cubic_bezier(p0, p1, p2, p3, n=60)


def _lips_outer(art: ArtDict) -> np.ndarray:
    ul, ll = art["UL"], art["LL"]
    centre = 0.5 * (ul + ll)
    v = ll - ul
    d = np.linalg.norm(v)

    # If UL == LL the curve collapses – bail out
    if d < 1e-4:
        return np.vstack([ul, ll])

    # outward (perpendicular) unit vector – pick the side facing away from tongue tip
    perp = np.array([-v[1], v[0]])
    perp /= np.linalg.norm(perp)
    # choose direction so the normal points away from the tongue centre
    if np.dot(perp, art["TT"] - centre) > 0:
        perp = -perp

    # bulge is 25 % of lip gap, minimum 4 mm so a closed mouth is still visible
    bulge = max(0.25 * d, 0.04)

    p0, p3 = ul, ll
    p1 = centre + perp * bulge
    p2 = centre - perp * bulge
    return cubic_bezier(p0, p1, p2, p3, n=50)


def calibrate_outline(traj: Dict[str, np.ndarray],
                      gap_y: float = 0.3,
                      target_lip_dist: float = 1.6) -> np.ndarray:
    """
    Build a mid‑sagittal vocal‑tract outline that fits *this* speaker.

    * Averages UL, LL, LI, TD across all frames.
    * Uses the same affine (LI origin, UL‑LL vertical) for consistency.
    * Palate arch spans from UL to a point ~1.4 cm above TD.
    * Pharynx + larynx trace follows behind TD.
    * Returns an outline in the **raw coordinate space** so animation can
      still apply the per‑speaker affine each frame.

    """
    # ── anchor means ───────────────────────────────────────────────
    mean = {k: traj[k].mean(axis=0) for k in ("UL", "LL", "LI", "TD")}
    R, t, s = _affine_from_frame(mean, target_lip_dist)
    a = {k: _apply_affine(v, R, t, s) for k, v in mean.items()}

    # ── palate arch (quadratic) ───────────────────────────────────
    pal_front = a["UL"]
    pal_back  = a["TD"] + np.array([-1.0, 1.4])        # 1 cm behind & 1.4 cm up
    xs = np.linspace(pal_back[0], pal_front[0], 40)
    ys = pal_back[1] + (pal_front[1] - pal_back[1]) * \
         (1 - ((xs - pal_back[0]) / (pal_front[0]-pal_back[0]))**2)
    palate = np.c_[xs, ys]

    # ── lip corners ───────────────────────────────────────────────
    lip_left  = a["UL"] + np.array([-0.6, 0.0])
    lip_right = a["UL"] + np.array([ 0.6, 0.0])

    # ── pharynx / larynx polyline ─────────────────────────────────
    pharynx = np.array([
        a["TD"] + [-1.0,  1.0],
        a["TD"] + [-1.0, -2.0],
    ])

    outline_affine = np.vstack([
        pharynx,
        palate[::-1],
        lip_left,
        lip_right,
        pharynx[0],
    ])

    # ── map back to raw space so animation’s affine still applies ─
    outline_raw = (R.T @ (outline_affine / s).T).T + t
    return outline_raw


def _to_path(points: np.ndarray) -> Path:
    codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1)
    return Path(points, codes)


###############################################################################
# Public animation function
###############################################################################


def animate_vocal_tract(
    traj: Dict[str, np.ndarray],
    *,
    fps: int = 25,
    save_gif: Optional[str] = None,
    save_mp4: Optional[str] = None,
    custom_outline: Optional[np.ndarray] = None,
) -> None:
    required = {"UL", "LL", "LI", "TT", "TB", "TD"}
    if not required.issubset(traj):
        raise ValueError(f"traj missing keys: {required - traj.keys()}")

    # Per‑speaker lip distance for better scaling
    lip_dist = _mean_lip_distance(traj)
    if lip_dist < 0.5:      # improbable => fallback to 1.6 cm
        lip_dist = 1.6

    T = next(iter(traj.values())).shape[0]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Speaker‑specific outline
    outline = (custom_outline
               if custom_outline is not None
               else calibrate_outline(traj, target_lip_dist=lip_dist))

    outline_line, = ax.plot(outline[:,0], outline[:,1], color='k', lw=1.5, alpha=0.4)

    sensor_scatter = ax.scatter([], [], s=20, c="k", marker="x", alpha=0.6)

    # Empty PathPatches for dynamic surfaces.
    tongue_patch = PathPatch(Path([[0, 0]]), fill=False, lw=4, color="tab:red")
    lip_patch = PathPatch(Path([[0, 0]]), fill=False, lw=4, color="tab:blue")
    palate_patch = PathPatch(
        Path([[0, 0]]), fill=False, lw=2, color="k", alpha=0.5
    )
    ax.add_patch(palate_patch)
    ax.add_patch(tongue_patch)
    ax.add_patch(lip_patch)

    # Compute static palate surface once.
    raw0 = {k: v[0] for k, v in traj.items()}
    R0, t0, s0 = _affine_from_frame(raw0, target_lip_dist=lip_dist)
    rig0 = _rig_points({k: _apply_affine(raw0[k], R0, t0, s0) for k in raw0})
    palate_patch.set_path(_to_path(_palate_surface(rig0)))

    # --- animation driver -----------------------------------------------------

    def _update(frame: int):
        raw = {k: v[frame] for k, v in traj.items()}
        art = {k: _apply_affine(raw[k], R0, t0, s0) for k in raw}

        sensor_xy = np.vstack([art[k] for k in ("UL","LL","LI","TT","TB","TD")])
        sensor_scatter.set_offsets(sensor_xy)

        rig = _rig_points(art)

        outline_frame = _translate_outline(outline, rig)
        outline_path  = _apply_affine(outline_frame, R0, t0, s0)
        outline_line.set_data(outline_path[:, 0], outline_path[:, 1])

        tongue_patch.set_path(_to_path(_tongue_surface(art, rig)))
        lip_patch.set_path(_to_path(_lips_outer(art)))
        return tongue_patch, lip_patch, palate_patch, outline_line, sensor_scatter

    def _init():
        # static palette and outline for first frame
        raw = {k: v[0] for k, v in traj.items()}
        art = {k: _apply_affine(raw[k], R0, t0, s0) for k in raw}
        sensor_xy = np.vstack([art[k] for k in ("UL","LL","LI","TT","TB","TD")])
        sensor_scatter.set_offsets(sensor_xy)

        rig_init = _rig_points(art)
        outline_frame = _translate_outline(outline, rig_init)
        outline_path  = _apply_affine(outline_frame, R0, t0, s0)
        outline_line.set_data(outline_path[:, 0], outline_path[:, 1])
        palate_patch.set_path(_to_path(_palate_surface(rig0)))
        return tongue_patch, lip_patch, palate_patch, outline_line, sensor_scatter

    ani = animation.FuncAnimation(
        fig, _update, frames=T, init_func=_init,
        interval=1000 / fps, blit=True
    )

    if save_gif:
        ani.save(save_gif, writer=animation.PillowWriter(fps=fps))
    if save_mp4:
        ani.save(save_mp4, writer=animation.FFMpegWriter(fps=fps))

    plt.show()


if __name__ == "__main__":
    T = 150
    th = np.linspace(0, 2 * np.pi, T)
    demo_traj = {
        "UL": np.c_[np.zeros(T),  0.35*np.ones(T)],
        "LL": np.c_[np.zeros(T), -0.05+0.1*np.sin(th)],
        "LI": np.c_[np.zeros(T), -1.3+0.1*np.sin(th)],
        "TT": np.c_[ 0.5*np.cos(th), -0.1+0.2*np.sin(th)],
        "TB": np.c_[ 0.1*np.cos(th), -0.25+0.15*np.sin(th)],
        "TD": np.c_[ -0.8*np.ones(T), -0.3+0.12*np.sin(th)],
    }

    animate_vocal_tract(demo_traj, fps=30)