#!python3

from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
import numpy as np

# Load the model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-fin")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fin")

# Input text
text = "Hei! Miten sulla menee? Oot tosi ihana."
inputs = tokenizer(text, return_tensors="pt")

# Generate the waveform
with torch.no_grad():
    outputs = model(**inputs).waveform

# Check the sampling rate
sampling_rate = model.config.sampling_rate
print(f"Sampling rate: {sampling_rate}")

# Verify if the sampling rate is a 16-bit unsigned short
if not (0 <= sampling_rate <= 65535):
    raise ValueError(f"The sampling rate ({sampling_rate}) is not within the range of 0 to 65535")

# Convert the output waveform to float numpy array
waveform = outputs.numpy()
waveform = waveform.squeeze()  # Remove the batch dimension if present

# Print min and max of the waveform for debugging
print(f"Waveform min: {waveform.min()}, max: {waveform.max()}")

# Ensure the waveform data is in the range -1 to 1
normalized_waveform = np.clip(waveform, -1.0, 1.0)

# Print min and max of the normalized waveform for debugging
print(f"Normalized waveform min: {normalized_waveform.min()}, max: {normalized_waveform.max()}")

# Write the waveform to a WAV file
try:
    scipy.io.wavfile.write("suomi.wav", rate=sampling_rate, data=normalized_waveform)
    print("WAV file written successfully.")
except Exception as e:
    print(f"Failed to write WAV file: {e}")
