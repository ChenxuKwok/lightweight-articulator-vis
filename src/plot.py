import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from matplotlib.animation import FuncAnimation
import logging


def show_ema_frame(traj: dict, frame: int):
    keys = ["UL", "LL", "LI", "TT", "TB", "TD"]
    colours = dict(UL="crimson", LL="gold", LI="grey",
                   TT="limegreen", TB="orchid", TD="steelblue")

    plt.figure(figsize=(4,4))
    for k in keys:
        x, y = traj[k][frame]
        plt.scatter(x, y, color=colours[k], s=40, zorder=3)
        plt.text(x+0.05, y+0.05, f"{k} ({x:.2f},{y:.2f})",
                 ha="left", va="bottom", fontsize=8)

    plt.title(f"EMA sensors – frame {frame}")
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()      # common EMA convention
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def live_view(traj, fps=25, save_gif=None):
    colours = dict(UL="crimson", LL="gold", LI="grey",
                   TT="limegreen", TB="orchid", TD="steelblue")

    T = next(iter(traj.values())).shape[0]
    keys = ["UL", "LL", "LI", "TT", "TB", "TD"]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.invert_yaxis()           # EMA y grows upward, invert for mouth-view
    ax.grid(True, ls="--", alpha=0.2)

    scatters = {}
    labels   = {}
    for k in keys:
        scatters[k] = ax.scatter([], [], c=colours[k], marker="x", s=40, zorder=3)
        labels[k]   = ax.text(0, 0, "", fontsize=7, ha="left", va="bottom")

    def update(frame):
        for k in keys:
            x, y = traj[k][frame]
            scatters[k].set_offsets([x, y])
            labels[k].set_position((x + 0.2, y + 0.2))
            labels[k].set_text(f"{k}")
        return list(scatters.values()) + list(labels.values())

    ani = FuncAnimation(fig, update, frames=T,
                        interval=1000 / fps, blit=True)
    plt.title("EMA live view")
    plt.show()
    if save_gif:
        ani.save(save_gif, writer='pillow', fps=fps)
        logging.info(f"Animation saved as {save_gif}")

if __name__ == "__main__":
    ema = np.load('sample1_ema.npy')  # Load EMA data (T, 12)
    T = ema.shape[0]
    # EMA channel_label = ['TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY']
    channels = ['TD', 'TB', 'TT', 'LI', 'UL', 'LL']   # order below matches doc
    idx = {                     # column indices for each articulator
        'TD': (0, 1),           # Tongue Dorsum
        'TB': (2, 3),           # Tongue Blade
        'TT': (4, 5),           # Tongue Tip
        'LI': (6, 7),           # Lower Incisor
        'UL': (8, 9),           # Upper Lip
        'LL': (10, 11),         # Lower Lip
    }
    traj = {name: ema[:, idx[name]] for name in channels}   # (T, 2) each
    # Show a specific frame
    frame_to_show = 0  # Change this to any frame index you want to visualize
    show_ema_frame(traj, frame_to_show)