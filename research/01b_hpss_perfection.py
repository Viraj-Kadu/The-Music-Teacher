import librosa
import numpy as np
import matplotlib.pyplot as plt

def perfection_hpss(file_path):
    print(f"--- Refining Signal: {file_path} ---")
    y, sr = librosa.load(file_path)

    # 1. Perform HPSS
    # This mathematically splits the audio into two arrays
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # 2. Re-run Tempo Detection on PERCUSSIVE only
    # We ignore the vocals/melody to find the true beat
    tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_hpss, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # Handle array outputs
    t_raw = float(tempo_raw[0]) if isinstance(tempo_raw, np.ndarray) else float(tempo_raw)
    t_hpss = float(tempo_hpss[0]) if isinstance(tempo_hpss, np.ndarray) else float(tempo_hpss)

    print(f"Original Result: {t_raw:.2f} BPM")
    print(f"Refined (Percussive) Result: {t_hpss:.2f} BPM")

    # 3. Re-run Key Detection on HARMONIC only
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    key_idx = np.argmax(np.mean(chroma, axis=1))
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    print(f"Refined (Harmonic) Key: {notes[key_idx]}")

    # 4. Save the "Clean" versions for visual inspection
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_harmonic)), ref=np.max), y_axis='log')
    plt.title("Harmonic Component (Melody Focus)")
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_percussive)), ref=np.max), y_axis='log')
    plt.title("Percussive Component (Rhythm Focus)")
    
    plt.tight_layout()
    plt.savefig("research/analysis_images/hpss_separation.png")
    print("Separation visual saved to research/analysis_images/hpss_separation.png")

if __name__ == "__main__":
    perfection_hpss("data/raw/Ben E. King - Stand By Me (Audio).mp3")