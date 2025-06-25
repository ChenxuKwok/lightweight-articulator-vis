import numpy as np

def process(ema):
    channels = ['TD', 'TB', 'TT', 'LI', 'UL', 'LL']
    idx = {
        'TD': (0, 1),
        'TB': (2, 3),
        'TT': (4, 5),
        'LI': (6, 7),
        'UL': (8, 9),
        'LL': (10, 11),
    }
    traj = {name: ema[:, idx[name]] for name in channels} # (T, 2) each
    return traj

def denorm_traj(traj_z, zmap):
    """
    Convert a z‑scored trajectory dict to real‑value coordinates.

    Parameters
    ----------
    traj_z : dict[str -> (T,2)]
        z‑score trajectory for each articulator.
    zmap : dict[str -> tuple[np.ndarray(2,), np.ndarray(2,)]]
        Per‑channel (center_cm, scale_cm_per_z) pairs.

    Returns
    -------
    dict[str -> (T,2)]
        De‑normalised trajectory in centimetres.
    """
    out = {}
    for key, z in traj_z.items():
        if key not in zmap:
            raise KeyError(f"{key} missing in zmap")
        center, scale = zmap[key]
        out[key] = center + z * scale
    return out


# Helper to build zmap
def build_zmap(
    traj_z: dict[str, np.ndarray],
    *,
    target_ul_ll_cm: float = 1.6,
    uniform_scale: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    required = {"UL", "LL"}
    if not required.issubset(traj_z):
        missing = ", ".join(required - traj_z.keys())
        raise ValueError(f"traj_z missing channels: {missing}")

    center = {k: traj_z[k].mean(axis=0) for k in traj_z}

    ul_z, ll_z = traj_z["UL"], traj_z["LL"]
    if uniform_scale:
        d_z = np.linalg.norm(ul_z - ll_z, axis=1).mean()
        scale_vec = np.array([target_ul_ll_cm / d_z] * 2)
        scale = {k: scale_vec for k in traj_z}
    else:
        diff = np.abs(ul_z - ll_z).mean(axis=0)  # (dx̄, dȳ)
        scale = {k: target_ul_ll_cm / diff for k in traj_z}

    zmap = {k: (center[k], scale[k]) for k in traj_z}
    return zmap

# ----------------------------------------------------------------------
# Build a fully-manual zmap from dicts
# ----------------------------------------------------------------------
def manual_zmap(
    centres: dict[str, tuple[float, float]],
    scales:  dict[str, float | tuple[float, float]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    centres : sensor → (x_cm , y_cm)
    scales  : sensor → scale (cm / z).  Give one number or (sx , sy)

    Returns a zmap you can pass to denorm_traj().
    """
    zmap = {}
    for k in centres:
        if k not in scales:
            raise KeyError(f"scale for {k} missing")
        c = np.asarray(centres[k], dtype=float)
        s = scales[k]
        s = np.asarray((s, s) if np.isscalar(s) else s, dtype=float)
        zmap[k] = (c, s)
    return zmap