import sounddevice as sd
import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 5

print("Recording audio for 5 seconds...")
print("Speak normally during recording.")

# Record audio
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()
audio = audio.flatten()

# Activity 3.1: Audio Signal Extraction
duration = len(audio) / SAMPLE_RATE
volume = np.sqrt(np.mean(audio ** 2))
maximum_volume = np.max(np.abs(audio))

# Divide audio into small frames
rms_frames = librosa.feature.rms(
    y=audio,
    frame_length=512,
    hop_length=256
)[0]

# Activity 3.2: Audio Activity Analysis
threshold = max(0.003, np.percentile(rms_frames, 25) * 2)

active_frames = rms_frames > threshold
activity_percentage = np.mean(active_frames) * 100

# Zero-crossing rate: another audio feature
zero_crossing = librosa.feature.zero_crossing_rate(
    y=audio,
    frame_length=512,
    hop_length=256
)[0]

average_zcr = np.mean(zero_crossing)

# Spectral centroid: average frequency-related feature
spectral_centroid = librosa.feature.spectral_centroid(
    y=audio,
    sr=SAMPLE_RATE,
    n_fft=512,
    hop_length=256
)[0]

average_centroid = np.mean(spectral_centroid)

print("\n--- Audio Features ---")
print(f"Duration: {duration:.2f} seconds")
print(f"RMS Volume: {volume:.4f}")
print(f"Maximum Volume: {maximum_volume:.4f}")
print(f"Zero Crossing Rate: {average_zcr:.4f}")
print(f"Spectral Centroid: {average_centroid:.2f} Hz")

print("\n--- Audio Activity Analysis ---")
print(f"Audio Activity: {activity_percentage:.2f}%")

if volume < 0.003:
    print("Result: Very quiet audio or silence.")
elif activity_percentage > 20:
    print("Result: Significant audio activity detected.")
else:
    print("Result: Low or moderate audio activity detected.")

print("\nAudio analysis completed!")