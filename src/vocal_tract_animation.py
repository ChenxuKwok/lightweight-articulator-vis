from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import bezier
import logging

logger = logging.getLogger(__name__)

# Colour cycle for rig points
import itertools
_cmap_cycle = itertools.cycle(plt.cm.tab20.colors)
SENSOR_KEYS = {"UL", "LL", "LI", "TT", "TB", "TD"}

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

RigDict = Dict[str, np.ndarray]

def load_rig_csv(csv_path: str) -> RigDict:
    """
    Read a simple rig CSV with columns: name,x,y
    Returns dict[name] = np.ndarray([x,y]).
    """
    import csv
    rig = {}
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().lower() in ("name", "#", "//"):
                # skip header or comment lines
                continue
            name, x, y = row[:3]
            rig[name.strip()] = np.array([float(x), float(y)])
    return rig

###############################################################################
# Coordinate normalisation
###############################################################################

def _affine_from_frame(*args, **kwargs):
    """ (disabled) returns identity transform """
    R = np.eye(2)
    t = np.zeros(2)
    s = 1.0
    return R, t, s

def _apply_affine(xy: np.ndarray, R: np.ndarray, t: np.ndarray, s: float):
    """ (disabled) passthrough – no transform """
    return xy


def _rig_points(art: ArtDict) -> Dict[str, np.ndarray]:
    """
    No automatic rigging.  We rely entirely on the hand‑labelled
    points 1‑13 loaded from CSV.
    """
    return {}

# Allow numeric rig ids (“1”..“13”) for hand‑labelled anchors.
def _get_point(name: str,
               art: ArtDict,
               rig: RigDict) -> np.ndarray:
    """Return the xy for *name* from either sensors or rig dict."""
    if name in art:
        return art[name]
    if name in rig:
        return rig[name]
    raise KeyError(f"Point '{name}' not found in rig or sensors.")


def _catmull_rom(P: np.ndarray, n: int = 60) -> np.ndarray:
    """
    Return n points on a Catmull-Rom spline passing through the
    control points P (kx2).  Endpoints are repeated for natural ends.
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


def _tongue_surface(art: ArtDict, rig: RigDict) -> np.ndarray:
    """
    Draw Catmull‑Rom through:
      1 - 2 - 3 - 4 - 5 - 6 - TD - TB - TT - 7 - 8 - 9 - 10 - 11 - LL - 12 - 13 - 14 - 15
    Numeric anchors (1‑15) must exist in `rig`.
    """
    order = [
        "1", "2", "3", "4", "5", "6",
        "TD", "TB", "TT",
        "7", "8", "9", "10", "11",
        "LL", "12", "13", "14", "15"
    ]
    pts = np.vstack([_get_point(n, art, rig) for n in order])
    return _catmull_rom(pts, n=80)


def _upper_jaw_surface(art: ArtDict, rig: RigDict) -> np.ndarray:
    """
    Smooth Catmull‑Rom through:
      16 – UL – 17 – 18 – 19 – 20 – 21 – 22
    """
    order = [
        "16", "UL", "17", "18", "19", "20", "21", "22"
    ]
    pts = np.vstack([_get_point(n, art, rig) for n in order])
    return _catmull_rom(pts, n=80)


# def calibrate_outline(traj: Dict[str, np.ndarray],
#                       gap_y: float = 0.3,
#                       target_lip_dist: float = 1.6) -> np.ndarray:
#     """
#     Build a mid‑sagittal vocal‑tract outline that fits *this* speaker.

#     * Averages UL, LL, LI, TD across all frames.
#     * Uses the same affine (LI origin, UL‑LL vertical) for consistency.
#     * Palate arch spans from UL to a point ~1.4 cm above TD.
#     * Pharynx + larynx trace follows behind TD.
#     * Returns an outline in the **raw coordinate space** so animation can
#       still apply the per‑speaker affine each frame.

#     """
#     # ── anchor means ───────────────────────────────────────────────
#     mean = {k: traj[k].mean(axis=0) for k in ("UL", "LL", "LI", "TD")}
#     R, t, s = _affine_from_frame(mean, target_lip_dist)
#     a = {k: _apply_affine(v, R, t, s) for k, v in mean.items()}

#     # ── palate arch (quadratic) ───────────────────────────────────
#     pal_front = a["UL"]
#     pal_back  = a["TD"] + np.array([-1.0, 1.4])        # 1 cm behind & 1.4 cm up
#     xs = np.linspace(pal_back[0], pal_front[0], 40)
#     ys = pal_back[1] + (pal_front[1] - pal_back[1]) * \
#          (1 - ((xs - pal_back[0]) / (pal_front[0]-pal_back[0]))**2)
#     palate = np.c_[xs, ys]

#     # ── lip corners ───────────────────────────────────────────────
#     lip_left  = a["UL"] + np.array([-0.6, 0.0])
#     lip_right = a["UL"] + np.array([ 0.6, 0.0])

#     # ── pharynx / larynx polyline ─────────────────────────────────
#     pharynx = np.array([
#         a["TD"] + [-1.0,  1.0],
#         a["TD"] + [-1.0, -2.0],
#     ])

#     outline_affine = np.vstack([
#         pharynx,
#         palate[::-1],
#         lip_left,
#         lip_right,
#         pharynx[0],
#     ])

#     # ── map back to raw space so animation’s affine still applies ─
#     outline_raw = (R.T @ (outline_affine / s).T).T + t
#     return outline_raw


def _to_path(points: np.ndarray) -> Path:
    codes = [Path.MOVETO] + [Path.LINETO] * (len(points) - 1)
    return Path(points, codes)


###############################################################################
# Public animation function
###############################################################################


# Helper to merge rigs
def _merge_rigs(auto_rig: RigDict, custom: Optional[RigDict]) -> RigDict:
    """
    With auto rig disabled, just return the custom rig (if any).
    """
    return custom or {}

def animate_vocal_tract(
    traj: Dict[str, np.ndarray],
    *,
    fps: int = 25,
    save_gif: Optional[str] = None,
    save_mp4: Optional[str] = None,
    custom_outline: Optional[np.ndarray] = None,
    custom_rig: Optional[RigDict] = None,
    show_labels: bool = False,
    show_axes: bool = False,
    label_sensors: bool = False,        # display names for UL LL LI TT TB TD
    xlim: tuple[float, float] = (-3.0, 3.0),
    ylim: tuple[float, float] = (-3.0, 3.0),
    rig_scale: float = 100.0   # divide raw rig coords by this value
) -> None:
    """
    Animate a vocal tract trajectory.

    Parameters
    ----------
    traj : Dict[str, np.ndarray]
        Dictionary of trajectories for sensors and rig points.
    fps : int
        Frames per second for animation.
    save_gif : Optional[str]
        Path to save animation as GIF.
    save_mp4 : Optional[str]
        Path to save animation as MP4.
    custom_outline : Optional[np.ndarray]
        Custom outline points.
    custom_rig : Optional[RigDict]
        Custom rig dictionary.
    show_labels : bool
        Show rig-point labels.
    show_axes : bool
        Show axes grid.
    label_sensors : bool
        If True, always show text labels for UL, LL, LI, TT, TB, TD.
        Independent of `show_labels`, which controls rig‑point labels.
    xlim : tuple[float, float]
        X-axis limits.
    ylim : tuple[float, float]
        Y-axis limits.
    rig_scale : float
        Scale factor for rig coordinates.
    """
    required = {"UL", "LL", "LI", "TT", "TB", "TD"}
    if not required.issubset(traj):
        raise ValueError(f"traj missing keys: {required - traj.keys()}")


    T = next(iter(traj.values())).shape[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    if show_axes:
        ax.grid(ls="--", alpha=0.3)
    else:
        ax.axis("off")

    def _convert_rig(raw_rig: RigDict) -> RigDict:
        if "root" not in raw_rig:
            return raw_rig
        origin = raw_rig["root"]
        conv = {}
        for k, v in raw_rig.items():
            if k == "root":
                continue
            # conv[k] = (v - origin) / rig_scale
            conv[k] = np.zeros(2)
            conv[k][0] = (v[0] - origin[0]) / rig_scale
            conv[k][1] = -(v[1] - origin[1]) / rig_scale
        return conv

    custom_rig = _convert_rig(custom_rig or {})

    sensor_scatter = ax.scatter([], [], s=20, c="k", marker="x", alpha=0.6)

    # Label texts (created lazily when show_labels is True)
    text_artists: Dict[str, plt.Text] = {}
    rig_scatters: Dict[str, plt.Line2D] = {}

    def _label(name: str, xy: np.ndarray):
        """
        Place a small text label and coloured dot.

        * LI is always labelled.
        * Other points follow the `show_labels` flag.
        """
        always = name == "LI"

        # ── ensure scatter exists & move it ────────────────────────
        if name not in rig_scatters:
            rig_scatters[name], = ax.plot(
                [], [], marker="o", markersize=5,
                color='yellow', linestyle="None", zorder=4
            )
        rig_scatters[name].set_data([xy[0]], [xy[1]])

        # ── decide whether to draw / update the text ───────────────
        if not show_labels and always:
            return                          # LI scatter only, no text
        if not show_labels and not always:
            return                          # other points not shown

        # create or move text
        if name not in text_artists:
            text_artists[name] = ax.text(
                xy[0] + 0.1, xy[1] + 0.1, name,
                fontsize=7, ha="left", va="bottom", color="black"
            )
        else:
            text_artists[name].set_position((xy[0] + 0.1, xy[1] + 0.1))

    # Empty PathPatches for dynamic surfaces.
    tongue_patch = PathPatch(Path([[0, 0]]), fill=False, lw=3, color="tab:red")
    upper_patch = PathPatch(Path([[0, 0]]), fill=False, lw=3, color="tab:blue")
    ax.add_patch(tongue_patch)
    ax.add_patch(upper_patch)

    # Compute static palate surface once.
    # raw0 = {k: v[0] for k, v in traj.items()}
    # R0, t0, s0 = np.eye(2), np.zeros(2), 1.0
    # rig0 = custom_rig
    # --- animation driver -----------------------------------------------------

    def _update(frame: int):
        raw = {k: v[frame] for k, v in traj.items()}
        art = raw

        sensor_xy = np.vstack([art[k] for k in ("UL","LL","LI","TT","TB","TD")])
        sensor_scatter.set_offsets(sensor_xy)

        # move sensor labels each frame
        if label_sensors or show_labels:
            for n, xy in zip(("UL","LL","LI","TT","TB","TD"), sensor_xy):
                _label(n, xy)
        else:
            _label("LI", art["LI"])   # LI scatter only

        rig = custom_rig

        # outline disabled

        tongue_patch.set_path(_to_path(_tongue_surface(art, rig)))
        upper_patch.set_path(_to_path(_upper_jaw_surface(art, rig)))

        return tuple([tongue_patch, upper_patch, sensor_scatter] +
                     list(text_artists.values()) +
                     list(rig_scatters.values()))

    def _init():
        # static palette and outline for first frame
        raw = {k: v[0] for k, v in traj.items()}
        art = raw
        sensor_xy = np.vstack([art[k] for k in ("UL","LL","LI","TT","TB","TD")])
        sensor_scatter.set_offsets(sensor_xy)
        if label_sensors or show_labels:
            for n, xy in zip(("UL","LL","LI","TT","TB","TD"), sensor_xy):
                _label(n, xy)
        else:
            _label("LI", art["LI"])   # LI scatter only

        rig_init = custom_rig
        if show_labels:
            for n, xy in rig_init.items():
                _label(str(n), xy)
        tongue_patch.set_path(_to_path(_tongue_surface(art, rig_init)))
        upper_patch.set_path(_to_path(_upper_jaw_surface(art, rig_init)))
        return tuple([tongue_patch, upper_patch, sensor_scatter] +
                     list(text_artists.values()) +
                     list(rig_scatters.values()))

    ani = animation.FuncAnimation(
        fig, _update, frames=T, init_func=_init,
        interval=1000 / fps, blit=True
    )

    if save_gif:
        ani.save(save_gif, writer=animation.PillowWriter(fps=fps))
    if save_mp4:
        ani.save(save_mp4, writer=animation.FFMpegWriter(fps=fps))

    logging.info(f"Animation created: {save_gif or save_mp4}")

    # plt.show()