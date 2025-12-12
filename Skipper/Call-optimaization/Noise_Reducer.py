import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy import signal
import os
import glob
from pathlib import Path

def enhance_audio(input_file, output_folder):
    """
    Advanced audio enhancement for call recordings:
    - Multi-stage noise reduction
    - Voice frequency isolation (highpass/lowpass filtering)
    - Amplitude normalization and boosting
    """
    
    print(f"\nProcessing: {os.path.basename(input_file)}")
    
    # 1. Load audio
    y, sr = librosa.load(input_file, sr=None)
    original_duration = len(y) / sr
    print(f"  Duration: {original_duration:.1f}s | Sample rate: {sr}Hz")
    
    # 2. STAGE 1: Remove stationary noise
    print("  [1/5] Removing stationary noise...")
    y_clean1 = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=True,
        prop_decrease=0.90,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50
    )
    
    # 3. STAGE 2: Remove non-stationary noise
    print("  [2/5] Removing vehicle/traffic noise...")
    y_clean2 = nr.reduce_noise(
        y=y_clean1,
        sr=sr,
        stationary=False,
        prop_decrease=0.70,
        n_std_thresh_stationary=1.2,
        time_constant_s=2.0
    )
    
    # 4. STAGE 3: Apply highpass filter (AUTO-ADJUSTS for sample rate)
    print("  [3/5] Filtering low-frequency rumble...")
    highpass_freq = min(80, sr * 0.01)
    sos_high = signal.butter(4, highpass_freq, 'hp', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos_high, y_clean2)
    
    # 5. STAGE 4: Apply lowpass filter (AUTO-ADJUSTS for Nyquist limit)
    print("  [4/5] Filtering high-frequency hiss...")
    nyquist = sr / 2
    lowpass_freq = min(nyquist * 0.95, 8000)
    sos_low = signal.butter(4, lowpass_freq, 'lp', fs=sr, output='sos')
    y_filtered = signal.sosfilt(sos_low, y_filtered)
    
    # 6. STAGE 5: Normalize and boost voice
    print("  [5/5] Normalizing and boosting voice...")
    peak = np.abs(y_filtered).max()
    if peak > 0:
        y_normalized = y_filtered / peak * 0.707
    else:
        y_normalized = y_filtered
    
    threshold = 0.3
    ratio = 3.0
    compressed = np.where(
        np.abs(y_normalized) > threshold,
        np.sign(y_normalized) * (threshold + (np.abs(y_normalized) - threshold) / ratio),
        y_normalized
    )
    
    y_final = np.clip(compressed * 1.5, -1.0, 1.0)
    
    # 7. Save
    output_filename = os.path.basename(input_file).replace('.mp3', '_enhanced.wav')
    output_path = os.path.join(output_folder, output_filename)
    sf.write(output_path, y_final, sr, subtype='PCM_16')
    
    print(f"  ✓ Saved: {output_filename}")
    return output_path


def batch_process(input_folder, output_folder="enhanced_audio"):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.MP3', '*.WAV']:
        audio_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not audio_files:
        print(f"❌ No audio files found in: {input_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"BATCH AUDIO ENHANCEMENT")
    print(f"{'='*60}")
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Files found:   {len(audio_files)}")
    print(f"{'='*60}")
    
    success_count = 0
    for i, file_path in enumerate(audio_files, 1):
        try:
            print(f"\n[{i}/{len(audio_files)}]", end=" ")
            enhance_audio(file_path, output_folder)
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully processed: {success_count}/{len(audio_files)} files")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    batch_process(
        input_folder=r"D:\Skipper\Call-optimaization\input",
        output_folder=r"D:\Skipper\Call-optimaization\output"
    )
