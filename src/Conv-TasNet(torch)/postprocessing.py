
import os
import numpy as np
import librosa
from pesq import pesq
from scipy.io.wavfile import write
import argparse

def spectral_subtraction(noisy_mag, alpha=1.0, noise_estimate=None):
    # Estimate noise using the first 10 frames if not provided
    if noise_estimate is None:
        noise_estimate = np.mean(noisy_mag[:, :9], axis=1, keepdims=True)
    enhanced_mag = noisy_mag - alpha * noise_estimate
    # Prevent negative magnitudes
    return np.maximum(enhanced_mag, 0.0)


# Argument parsing
parser = argparse.ArgumentParser(description="Postprocessing for denoised audio evaluation")
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing both 03_denoised and 01_clean wav files')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save enhanced audio')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)


pesq_scores = []
for filename in os.listdir(input_dir):
    if not (filename.endswith(".wav") and "03_denoised" in filename):
        continue

    # 파일명에서 {num} 추출
    # 예: test_sample_12_03_denoised.wav -> num = 12
    parts = filename.split('_')
    if len(parts) < 4:
        print(f"Filename format error: {filename}")
        continue
    num = parts[2]
    clean_filename = f"test_sample_{num}_01_clean.wav"

    input_path = os.path.join(input_dir, filename)
    clean_path = os.path.join(input_dir, clean_filename)
    output_path = os.path.join(output_dir, filename)

    try:
        # Load noisy audio
        y, sr = librosa.load(input_path, sr=16000, mono=True)
        D = librosa.stft(y)
        magnitude, phase = librosa.magphase(D)

        # Apply spectral subtraction
        enhanced_magnitude = spectral_subtraction(magnitude, alpha=1.1)
        enhanced_D = enhanced_magnitude * phase
        enhanced_y = librosa.istft(enhanced_D)

        # Save enhanced audio
        write(output_path, sr, (enhanced_y * 32767).astype(np.int16))

        # Load reference (clean) audio
        ref_y, _ = librosa.load(clean_path, sr=16000, mono=True)
        min_len = min(len(ref_y), len(enhanced_y))
        ref_y = ref_y[:min_len]
        enhanced_y = enhanced_y[:min_len]

        # Compute PESQ
        score = pesq(16000, ref_y, enhanced_y, 'wb')
        pesq_scores.append((filename, score))
        print(f"{filename}: PESQ = {score:.4f}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Print PESQ summary
print("\nPESQ Score Summary:")
for fname, score in pesq_scores:
    print(f"{fname}: {score:.4f}")

# Average PESQ
if pesq_scores:
    avg_pesq = np.mean([s for _, s in pesq_scores])
    print(f"\nAverage PESQ: {avg_pesq:.4f}")
else:
    print("No PESQ scores could be calculated.")
