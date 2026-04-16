import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def run_research_extraction(file_path):
    print(f"--- Processing: {file_path} ---")
    
    # 1. Load audio
    y, sr = librosa.load(file_path)

    # 2. Extract Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # FIX: Handle tempo being returned as an array
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo[0])
    else:
        tempo_val = float(tempo)
        
    print(f"Result -> Tempo: {tempo_val:.2f} BPM")

    # 3. Extract Scale (Chromagram)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # We'll print the top 3 likely notes for better research insights
    top_3_indices = np.argsort(mean_chroma)[-3:][::-1]
    likely_keys = [notes[i] for i in top_3_indices]
    
    print(f"Result -> Detected Key: {likely_keys[0]} (Alternatives: {', '.join(likely_keys[1:])})")

    # 4. Generate Visual Representation
    plt.figure(figsize=(12, 6))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Analysis: {likely_keys[0]} @ {tempo_val:.2f} BPM')
    
    output_viz = "research/analysis_images/analysis_standbyme.png"
    plt.savefig(output_viz)
    plt.close() # Close to free up memory
    print(f"Visualization saved to: {output_viz}")

if __name__ == "__main__":
    target_file = "data/raw/Ben E. King - Stand By Me (Audio).mp3"
    try:
        run_research_extraction(target_file)
    except Exception as e:
        print(f"An error occurred: {e}")