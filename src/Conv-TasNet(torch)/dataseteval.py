import os
import glob
import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def get_file_pairs(clean_dir, noisy_dir):
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    pairs = []
    for clean_path in clean_files:
        fname = os.path.basename(clean_path)
        noisy_path = os.path.join(noisy_dir, fname)
        if os.path.exists(noisy_path):
            pairs.append((clean_path, noisy_path))
    return pairs

def calc_metrics(pair, sample_rate=16000):
    clean_path, noisy_path = pair
    sr_c, clean = wavfile.read(clean_path)
    sr_n, noisy = wavfile.read(noisy_path)
    if sr_c != sample_rate or sr_n != sample_rate:
        raise ValueError(f"Sample rate mismatch: {clean_path}, {noisy_path}")
    if len(clean.shape) > 1:
        clean = clean[:, 0]
    if len(noisy.shape) > 1:
        noisy = noisy[:, 0]
    pesq_score = pesq(sample_rate, clean, noisy, 'wb')
    stoi_score = stoi(clean, noisy, sample_rate, extended=True)
    return pesq_score, stoi_score

def eval_folder(folder):
    clean_dir = os.path.join(folder, "clean")
    noisy_dir = os.path.join(folder, "noisy")
    pairs = get_file_pairs(clean_dir, noisy_dir)
    pesq_scores = []
    stoi_scores = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calc_metrics, pair) for pair in pairs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{folder}"):
            try:
                pesq_score, stoi_score = future.result()
                pesq_scores.append(pesq_score)
                stoi_scores.append(stoi_score)
            except Exception as e:
                print(f"Error: {e}")
    print(f"{folder}:")
    print(f"  PESQ 평균: {np.mean(pesq_scores):.3f}")
    print(f"  STOI 평균: {np.mean(stoi_scores):.3f}")
    print(f"  파일 수: {len(pesq_scores)}")

if __name__ == "__main__":
    base = "dataset"
    for split in ["train", "test", "validation"]:
        folder = os.path.join(base, split)
        eval_folder(folder)