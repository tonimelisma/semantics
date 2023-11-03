#!python3

from transformers import pipeline
import scipy
import torch

synthesiser = pipeline(
    "text-to-audio", "facebook/musicgen-large", device=torch.device("mps")
)

music = synthesiser(
    "lo-fi music with a soothing melody", forward_params={"do_sample": True}
)

scipy.io.wavfile.write(
    "musicgen_out.wav", rate=music["sampling_rate"], music=audio["audio"]
)
