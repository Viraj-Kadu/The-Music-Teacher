import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

def record_audio(seconds, fs, message):
    print(f"\n>>> {message}")
    print(f"Recording for {seconds} seconds...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Done.")
    return recording.flatten()

def run_calibration():
    fs = 44100
    duration = 5
    
    # 1. Capture Noise Floor (Silence)
    noise_sample = record_audio(duration, fs, "STAY SILENT: Capturing your room's noise floor...")
    
    # 2. Capture Vocal Range (Singing)
    vocal_sample = record_audio(duration, fs, "SING: Start quiet and get progressively LOUDER...")

    # --- Analysis Logic ---
    noise_rms = np.sqrt(np.mean(noise_sample**2))
    vocal_rms = np.sqrt(np.mean(vocal_sample**2))
    
    # Avoid log(0)
    snr = 20 * np.log10(vocal_rms / (noise_rms + 1e-9))
    
    # Check for clipping (values hitting 1.0 or -1.0)
    clipping_ratio = np.sum(np.abs(vocal_sample) >= 0.98) / len(vocal_sample)
    
    print(f"\n--- HARDWARE HEALTH REPORT ---")
    print(f"Noise Floor (RMS): {noise_rms:.6f}")
    print(f"Vocal Signal (RMS): {vocal_rms:.6f}")
    print(f"SNR: {snr:.2f} dB (Target: > 30dB)")
    print(f"Clipping detected in {clipping_ratio:.2%} of samples.")

    if snr < 10:
        print("\n[!] WARNING: Very low signal. Check your System Settings > Sound > Input Volume.")

    # --- Visualization ---
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    D_noise = librosa.amplitude_to_db(np.abs(librosa.stft(noise_sample)), ref=np.max)
    librosa.display.specshow(D_noise, y_axis='log', sr=fs)
    plt.title(f"Noise Floor Spectrum (SNR: {snr:.1f}dB)")
    
    plt.subplot(3, 1, 2)
    D_vocal = librosa.amplitude_to_db(np.abs(librosa.stft(vocal_sample, hop_length=256)), ref=np.max)
    librosa.display.specshow(D_vocal, y_axis='mel', sr=fs, hop_length=256)
    plt.title("Vocal Harmonic Clarity (HD 256-hop)")

    plt.subplot(3, 1, 3)
    plt.plot(vocal_sample, color='gray', alpha=0.5)
    plt.axhline(y=0.95, color='r', linestyle='--', label='Clipping Zone')
    plt.axhline(y=-0.95, color='r', linestyle='--')
    plt.title("Dynamic Range / Clipping Check")
    plt.tight_layout()

    plt.savefig("research/analysis_images/mic_calibration_report.png")
    print(f"\nReport saved to research/analysis_images/mic_calibration_report.png")

if __name__ == "__main__":
    run_calibration()