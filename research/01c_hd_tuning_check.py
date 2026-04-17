import librosa
import numpy as np
import matplotlib.pyplot as plt

def perfection_hd_analysis(file_path):
    print(f"--- HD Analysis & Tuning Check: {file_path} ---")
    y, sr = librosa.load(file_path)

    # 1. Tuning Check
    # Measures deviation from A440 in fractions of a semitone (cents)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    print(f"Detected Tuning Offset: {tuning:.2f} cents")
    
    if abs(tuning) > 0.05:
        print(f"Note: This song is slightly {'sharp' if tuning > 0 else 'flat'}.")
    else:
        print("Note: This song is perfectly tuned to A440.")

    # 2. HD Spectrogram Settings
    # hop_length: Smaller means more 'vertical slices' per second (Higher Time Resolution)
    # n_fft: Larger means better 'pitch separation' (Higher Frequency Resolution)
    n_fft = 2048
    hop_length = 256 

    # 3. Generate the HD Mel-Spectrogram
    # We'll use the Percussive component we found yesterday for cleaner beats
    _, y_percussive = librosa.effects.hpss(y)
    
    S = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=128
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 4. Visualization
    plt.figure(figsize=(15, 8))
    librosa.display.specshow(
        S_dB, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='mel',
        fmin=50, 
        fmax=8000
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"HD Analysis | Tuning: {tuning:.2f} cents | Res: {hop_length} hop")
    
    output_path = "research/analysis_images/hd_spectrogram.png"
    plt.savefig(output_path, dpi=300) # dpi=300 for high-print quality
    print(f"HD Visualization saved to: {output_path}")

if __name__ == "__main__":
    perfection_hd_analysis("data/raw/Ben E. King - Stand By Me (Audio).mp3")