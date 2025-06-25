from sparc import load_model
import soundfile as sf
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import logging

# ---- Custom imports ----
from src.vocal_tract_animation import animate_vocal_tract, load_rig_csv
from src.plot import show_ema_frame, live_view, show_avg_frame
from src.init import process, build_zmap, denorm_traj, manual_zmap
from src.logging.logging_setup import setup_logging
import time

setup_logging()
logging.config.fileConfig('./src/logging/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# logging.info("Loading model...")
# coder = load_model(
#     config="sparc_models/model_english_1500k.yaml",
#     ckpt="sparc_models/model_english_1500k.ckpt",
#     device="cpu",
#     linear_model_path="sparc_models/wavlm_large-9_cut-10_mngu_linear.pkl"
# )
# logging.info("Model loaded successfully.")


# audio = 'sample_audio/sample2.wav'
# # check if audio file exists
# try:
#     wav, sr = sf.read(audio)
# except FileNotFoundError:
#     logging.error(f"Audio file {audio} not found.")
#     quit()

# logging.info("Encoding audio...")
# code = coder.encode(audio)
# np.save('code_sample2.npy', code, allow_pickle=True)
# if code is None:
#     logging.error("Failed to encode audio.")
#     quit()
# logging.info("Audio encoded successfully.")

code = np.load('code_sample2.npy', allow_pickle=True).item()

ema = code['ema']

T = ema.shape[0]
logging.info(f"Loaded EMA data with {T} frames.")
traj_z = process(ema)        # z‑scored trajectory

centres = {
    "LI": (8.0,  -10.0),
    "UL": (0.0,  0.8),
    "LL": (1.5, -7.0),
    "TT": (8.0, -2.0),
    "TB": (16.0, -2.5),
    "TD": (24.0, -2.0),
}

scales = {
    "LI": 0.4,
    "UL": 0.8,
    "LL": 0.4,
    "TT": 1.0,
    "TB": 1.0,
    "TD": 1.0,
}

zmap = manual_zmap(centres, scales)

# convert z-scores → cm
traj_cm = denorm_traj(traj_z, zmap)



logging.info(f"Z-map built with {len(zmap)} points.")
logging.info(f"Z-map: {zmap}")
traj_cm = denorm_traj(traj_z, zmap)

# # show certain frame
# show_ema_frame(traj_cm, 0)

# show average frame
# show_avg_frame(traj_cm)

# # live ema view
# live_view(traj_cm, fps=10, save_gif="sample1.gif")

# animate vocal tract
rig = load_rig_csv('rigs/rig_points.csv')
animate_vocal_tract(
    traj_cm,
    fps=25,
    save_gif="demos/sample-2-manu.gif",
    custom_rig=rig,
    show_labels=False,
    show_axes=True,
    label_sensors=True,
    rig_scale=15,
    xlim=(-10, 40),
    ylim=(-30, 10),
)
