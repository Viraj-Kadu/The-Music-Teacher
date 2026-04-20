import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def isolate_vocals(file_path):
    print(f"--- Isolating Vocals: {file_path} ---")
    y, sr = librosa.load(file_path)

    # 1. Get the Magnitude Spectrogram (S) and Phase (P)
    S_full, phase = librosa.magphase(librosa.stft(y))

    # 2. Filter the Background (The 'Repeating' parts)
    # We use a median filter to find the common elements across time
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    # Ensure the filter doesn't exceed the original signal
    S_filter = np.minimum(S_full, S_filter)

    # 3. Create Masks
    # We use a 'soft' mask to separate components based on the filter
    margin_i, margin_v = 2, 10
    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=2)
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=2)

    # 4. Apply the masks to the original magnitude
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    # 5. Reconstruct the audio
    # Using the original phase to make it sound like a song again
    y_vocal = librosa.istft(S_foreground * phase)
    y_inst = librosa.istft(S_background * phase)

    # 6. Save the results
    vocal_out = "data/raw/stand_by_me_vocals.wav"
    inst_out = "data/raw/stand_by_me_instruments.wav"
    sf.write(vocal_out, y_vocal, sr)
    sf.write(inst_out, y_inst, sr)
    print(f"Isolated Vocals saved to: {vocal_out}")

    # 7. Visualization
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_foreground, ref=np.max), y_axis='log', x_axis='time')
    plt.title("Isolated Vocals (Foreground)")
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_background, ref=np.max), y_axis='log', x_axis='time')
    plt.title("Isolated Instruments (Background)")
    
    plt.tight_layout()
    plt.savefig("research/analysis_images/vocal_isolation_check.png")
    print("Visualization saved to research/analysis_images/vocal_isolation_check.png")

if __name__ == "__main__":
    isolate_vocals("data/raw/Ben E. King - Stand By Me (Audio).mp3")