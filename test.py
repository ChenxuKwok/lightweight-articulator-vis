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
from src.init import process
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


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
# if code is None:
#     logging.error("Failed to encode audio.")
#     quit()
# logging.info("Audio encoded successfully.")

code = np.load('code_sample1.npy', allow_pickle=True).item()
ema = code['ema']
T = ema.shape[0]
logging.info(f"Loaded EMA data with {T} frames.")
traj = process(ema)

# # show certain frame
# show_ema_frame(traj, 0)

# # show average frame
# show_avg_frame(traj)

# # live ema view
# live_view(traj, fps=10, save_gif="sample1.gif")

# animate vocal tract
rig = load_rig_csv('rigs/rig_points.csv')
animate_vocal_tract(traj, 
                    fps=25, 
                    save_gif="sample-test.gif", 
                    custom_rig=rig, 
                    show_labels=True, 
                    show_axes=True,
                    rig_scale=100*0.5)

