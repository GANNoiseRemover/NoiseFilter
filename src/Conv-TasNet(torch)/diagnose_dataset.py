"""
Dataset diagnostics & fixer for speech denoising pairs.

Features
- Validate CSV pairs (existence, basename match, duplicates)
- Basic audio stats: duration, RMS/peak, clipped/zero ratio, DC offset
- SR/Channels check
- Alignment: GCC-PHAT based delay estimate (clean vs noisy)
- SNR estimate with optional delay compensation
- Optional volume normalization (RMS based) and writing adjusted copies
- Optional writing time-aligned copies (shift noisy to clean)
- Saves a CSV report + summary plots

Usage (examples)
  python diagnose_dataset.py --csv dataset/validation.csv --sr 16000 --max-pairs 200
  python diagnose_dataset.py --csv dataset/train.csv --write-report diagnostics_out --volume-normalize --write-adjusted --normalized-outdir normalized --target-rms-db -23
  python diagnose_dataset.py --csv dataset/train.csv --write-aligned --aligned-outdir aligned_pairs

Notes
- Volume normalization is RMS-based (not LUFS) to avoid extra dependencies.
- Alignment uses a single-shot GCC-PHAT over (optionally truncated) signals; for highly non-stationary audio,
  consider chunk-wise estimation.
"""

from __future__ import annotations

import os
import argparse
import math
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torchaudio
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


# -----------------------------
# Small audio utilities
# -----------------------------

EPS = 1e-12


def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 in [-1, 1]. Returns (samples[T], sr)."""
    wav, sr = torchaudio.load(path)  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return wav, int(sr)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + EPS))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)) if x.size else 0.0)


def clipped_ratio(x: np.ndarray, thr: float = 0.999) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= thr))


def zero_ratio(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x == 0.0))


def dc_offset(x: np.ndarray) -> float:
    return float(np.mean(x))


def dbfs_from_lin(a: float) -> float:
    return 20.0 * math.log10(max(a, EPS))


def lin_from_db(db: float) -> float:
    return 10.0 ** (db / 20.0)


def compute_gcc_phat(sig: np.ndarray, refsig: np.ndarray, sr: int, max_delay_s: Optional[float] = None) -> Tuple[int, float]:
    """Estimate delay of refsig->sig using GCC-PHAT.

    Returns (delay_samples, max_corr_value).
    Positive delay means: sig is delayed vs refsig (sig starts later),
    so shifting sig by -delay will align.
    """
    # Zero-pad to same length
    n = int(2 ** math.ceil(math.log2(max(len(sig), len(refsig)))))
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REF)
    denom = np.abs(R) + EPS
    R /= denom
    cc = np.fft.irfft(R, n=n)
    # Shift zero lag to center
    cc = np.concatenate((cc[-(n // 2):], cc[: (n // 2)]))

    if max_delay_s is not None and max_delay_s > 0:
        max_lag = min(int(max_delay_s * sr), len(cc) // 2)
        mid = len(cc) // 2
        search = cc[mid - max_lag : mid + max_lag + 1]
        offset = np.argmax(search) - max_lag
        value = float(search[offset + max_lag])
        return int(offset), value
    else:
        offset = int(np.argmax(cc) - (len(cc) // 2))
        value = float(np.max(cc))
        return offset, value


def apply_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Shift signal by delay_samples (positive: shift right)."""
    if delay_samples == 0:
        return signal
    if delay_samples > 0:
        return np.pad(signal, (delay_samples, 0), mode="constant")
    else:
        return signal[-delay_samples:]


def estimate_snr_db(clean: np.ndarray, noisy: np.ndarray, offset: int = 0) -> float:
    """Estimate SNR after compensating the given offset (noisy aligned to clean).
    SNR = ||clean||^2 / ||noisy - clean||^2, computed on overlapping region.
    """
    if offset != 0:
        noisy = apply_delay(noisy, -offset)  # align to clean
    # Overlap region
    n = min(len(clean), len(noisy))
    if n <= 1:
        return float("nan")
    c = clean[:n]
    y = noisy[:n]
    noise = y - c
    p_clean = float(np.mean(np.square(c), dtype=np.float64) + EPS)
    p_noise = float(np.mean(np.square(noise), dtype=np.float64) + EPS)
    return 10.0 * math.log10(p_clean / p_noise)


def volume_normalize(x: np.ndarray, target_rms_db: float, peak_headroom_db: float = 1.0) -> Tuple[np.ndarray, float, float]:
    """Normalize to target RMS (dBFS) and limit peaks to 1 - headroom.
    Returns (y, gain_db, post_peak).
    """
    cur_rms = rms(x)
    cur_db = dbfs_from_lin(cur_rms)
    gain_db = target_rms_db - cur_db
    y = x * lin_from_db(gain_db)
    peak_lim = lin_from_db(-peak_headroom_db)  # e.g., -1 dBFS
    cur_peak = peak(y)
    if cur_peak > peak_lim + 1e-6:
        y = y * (peak_lim / (cur_peak + EPS))
    return y.astype(np.float32, copy=False), gain_db, float(peak(y))


def safe_save_wav(path: str, data: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = torch.from_numpy(data).unsqueeze(0)  # (1, T)
    torchaudio.save(path, tensor, sr)


# -----------------------------
# Diagnostics core
# -----------------------------

@dataclass
class PairReport:
    index: int
    clean_path: str
    noisy_path: str
    sr_clean: int
    sr_noisy: int
    dur_clean_s: float
    dur_noisy_s: float
    rms_clean_db: float
    rms_noisy_db: float
    peak_clean: float
    peak_noisy: float
    clipped_pct_clean: float
    clipped_pct_noisy: float
    zero_pct_clean: float
    zero_pct_noisy: float
    dc_clean: float
    dc_noisy: float
    basename_match: bool
    delay_samples: int
    delay_ms: float
    gcc_max: float
    snr_db_raw: float
    snr_db_aligned: float


def analyze_pairs(csv_path: str,
                  sr_target: Optional[int] = None,
                  max_pairs: Optional[int] = None,
                  analyze_seconds: Optional[float] = 15.0,
                  max_delay_ms: float = 100.0) -> Tuple[List[PairReport], dict]:
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")
    if max_pairs is not None:
        rows = rows[:max_pairs]

    reports: List[PairReport] = []
    file_missing = 0
    sr_mismatch = 0
    basename_mismatch = 0
    exceptions = 0

    for i, row in enumerate(tqdm(rows, desc="Analyzing pairs")):
        clean_path = row["clean_path"]
        noisy_path = row["noisy_path"]

        if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
            file_missing += 1
            continue

        try:
            clean, sr_c = load_wav_mono(clean_path)
            noisy, sr_n = load_wav_mono(noisy_path)
        except Exception:
            exceptions += 1
            continue

        # Optionally limit length for speed
        def crop_seconds(x: np.ndarray, sr: int, secs: Optional[float]):
            if not secs or secs <= 0:
                return x
            max_len = int(sr * secs)
            return x[:max_len]

        clean = crop_seconds(clean, sr_c, analyze_seconds)
        noisy = crop_seconds(noisy, sr_n, analyze_seconds)

        if sr_target is not None:
            # Do not actually resample the files; diagnostics only flags mismatch
            pass

        if sr_c != sr_n:
            sr_mismatch += 1

        dur_c = len(clean) / sr_c if sr_c > 0 else 0.0
        dur_n = len(noisy) / sr_n if sr_n > 0 else 0.0

        rms_c = rms(clean)
        rms_n = rms(noisy)
        peak_c = peak(clean)
        peak_n = peak(noisy)
        clip_c = clipped_ratio(clean)
        clip_n = clipped_ratio(noisy)
        zero_c = zero_ratio(clean)
        zero_n = zero_ratio(noisy)
        dc_c = dc_offset(clean)
        dc_n = dc_offset(noisy)

        base_c = os.path.splitext(os.path.basename(clean_path))[0]
        base_n = os.path.splitext(os.path.basename(noisy_path))[0]
        base_ok = (base_c == base_n)
        if not base_ok:
            basename_mismatch += 1

        # Align noisy to clean via GCC-PHAT (limit search window)
        delay_samp, gcc_max = compute_gcc_phat(noisy, clean, sr_c, max_delay_s=max_delay_ms / 1000.0)
        delay_ms = 1000.0 * delay_samp / float(sr_c)

        # SNR estimates
        snr_raw = estimate_snr_db(clean, noisy, offset=0)
        snr_aligned = estimate_snr_db(clean, noisy, offset=delay_samp)

        rep = PairReport(
            index=i,
            clean_path=clean_path,
            noisy_path=noisy_path,
            sr_clean=sr_c,
            sr_noisy=sr_n,
            dur_clean_s=dur_c,
            dur_noisy_s=dur_n,
            rms_clean_db=dbfs_from_lin(rms_c),
            rms_noisy_db=dbfs_from_lin(rms_n),
            peak_clean=peak_c,
            peak_noisy=peak_n,
            clipped_pct_clean=clip_c,
            clipped_pct_noisy=clip_n,
            zero_pct_clean=zero_c,
            zero_pct_noisy=zero_n,
            dc_clean=dc_c,
            dc_noisy=dc_n,
            basename_match=base_ok,
            delay_samples=delay_samp,
            delay_ms=delay_ms,
            gcc_max=gcc_max,
            snr_db_raw=snr_raw,
            snr_db_aligned=snr_aligned,
        )
        reports.append(rep)

    summary = {
        "total_pairs": len(rows),
        "analyzed_pairs": len(reports),
        "file_missing": file_missing,
        "sr_mismatch": sr_mismatch,
        "basename_mismatch": basename_mismatch,
        "exceptions": exceptions,
    }
    return reports, summary


def write_report(reports: List[PairReport], summary: dict, out_dir: str, csv_name: str = "diagnostics_report.csv") -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in reports])
    csv_path = os.path.join(out_dir, csv_name)
    df.to_csv(csv_path, index=False)

    # Save summary JSON
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Quick plots
    try:
        if not df.empty:
            plt.figure(figsize=(8,4))
            plt.hist(df["delay_ms"], bins=50, alpha=0.8)
            plt.title("Estimated Delay (ms)")
            plt.xlabel("ms"); plt.ylabel("count")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "hist_delay_ms.png")); plt.close()

            plt.figure(figsize=(8,4))
            plt.hist(df["snr_db_raw"], bins=50, alpha=0.8, label="raw")
            plt.hist(df["snr_db_aligned"], bins=50, alpha=0.5, label="aligned")
            plt.legend(); plt.title("SNR (dB)")
            plt.xlabel("dB"); plt.ylabel("count")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "hist_snr_db.png")); plt.close()

            plt.figure(figsize=(8,4))
            plt.hist(df["rms_clean_db"], bins=50, alpha=0.8, label="clean")
            plt.hist(df["rms_noisy_db"], bins=50, alpha=0.5, label="noisy")
            plt.legend(); plt.title("RMS dBFS")
            plt.xlabel("dBFS"); plt.ylabel("count")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "hist_rms_dbfs.png")); plt.close()
    except Exception as e:
        print(f"[plot] failed: {e}")

    return csv_path


def maybe_write_adjusted_and_aligned(
    df: pd.DataFrame,
    reports: List[PairReport],
    write_adjusted: bool,
    normalized_outdir: Optional[str],
    target_rms_db: float,
    headroom_db: float,
    volume_mode: str,
    write_aligned: bool,
    aligned_outdir: Optional[str],
    make_csv_under: Optional[str],
    orig_csv_path: Optional[str],
    # Alignment acceptance thresholds
    align_accept_gcc_min: float,
    align_accept_snr_improve_min: float,
    align_accept_delay_max_ms: Optional[float],
    # Rejection outputs
    write_rejected: bool,
    reject_csv_name: str = "rejected.csv",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Write normalized/aligned copies and return paths to new CSVs (normalized_csv, aligned_csv, rejected_csv)."""
    if not write_adjusted and not write_aligned:
        return None, None, None

    normalized_rows = []
    aligned_rows = []
    rejected_rows = []

    for r in tqdm(reports, desc="Writing adjusted/aligned"):
        try:
            clean, sr_c = load_wav_mono(r.clean_path)
            noisy, sr_n = load_wav_mono(r.noisy_path)
            if sr_c != sr_n:
                # We do not resample; skip writing to avoid SR drift.
                continue

            # Adjust volume
            if write_adjusted and normalized_outdir:
                if volume_mode == "common":
                    clean_adj, _, _ = volume_normalize(clean, target_rms_db, headroom_db)
                    noisy_adj, _, _ = volume_normalize(noisy, target_rms_db, headroom_db)
                elif volume_mode == "match_clean":
                    # Match noisy RMS to clean RMS (preserve per-file SNR roughly)
                    clean_rms_db = dbfs_from_lin(rms(clean))
                    noisy_rms_db = dbfs_from_lin(rms(noisy))
                    gain_db = (clean_rms_db - noisy_rms_db)
                    noisy_adj = noisy * lin_from_db(gain_db)
                    clean_adj = clean.copy()
                    # Peak limit
                    peak_lim = lin_from_db(-headroom_db)
                    pk = peak(noisy_adj)
                    if pk > peak_lim:
                        noisy_adj = noisy_adj * (peak_lim / (pk + EPS))
                else:
                    raise ValueError("volume_mode must be 'common' or 'match_clean'")

                # Save under normalized_outdir preserving rel path structure by file name only
                base_c = os.path.basename(r.clean_path)
                base_n = os.path.basename(r.noisy_path)
                out_c = os.path.join(normalized_outdir, "clean", base_c)
                out_n = os.path.join(normalized_outdir, "noisy", base_n)
                safe_save_wav(out_c, clean_adj, sr_c)
                safe_save_wav(out_n, noisy_adj, sr_n)
                normalized_rows.append({"clean_path": os.path.abspath(out_c), "noisy_path": os.path.abspath(out_n)})

            # Write aligned noisy (shift to clean)
            if write_aligned and aligned_outdir:
                # Decide acceptance based on thresholds
                improve = float(r.snr_db_aligned - r.snr_db_raw)
                accept = True
                if r.gcc_max < align_accept_gcc_min:
                    accept = False
                if improve < align_accept_snr_improve_min:
                    accept = False
                if align_accept_delay_max_ms is not None and align_accept_delay_max_ms > 0:
                    if abs(r.delay_ms) > align_accept_delay_max_ms:
                        accept = False

                if accept:
                    delay = r.delay_samples
                    noisy_al = apply_delay(noisy, -delay)
                    # Make lengths similar (crop/pad to clean length)
                    if len(noisy_al) < len(clean):
                        noisy_al = np.pad(noisy_al, (0, len(clean) - len(noisy_al)))
                    if len(noisy_al) > len(clean):
                        noisy_al = noisy_al[: len(clean)]
                    base_c = os.path.basename(r.clean_path)
                    base_n = os.path.basename(r.noisy_path)
                    out_c = os.path.join(aligned_outdir, "clean", base_c)
                    out_n = os.path.join(aligned_outdir, "noisy", base_n)
                    safe_save_wav(out_c, clean, sr_c)
                    safe_save_wav(out_n, noisy_al, sr_c)
                    aligned_rows.append({"clean_path": os.path.abspath(out_c), "noisy_path": os.path.abspath(out_n)})
                else:
                    rejected_rows.append({
                        "clean_path": r.clean_path,
                        "noisy_path": r.noisy_path,
                        "reason": f"gcc_max={r.gcc_max:.3f} improve={improve:.3f} delay_ms={r.delay_ms:.2f}",
                    })

        except Exception as e:
            print(f"[write] {r.index} failed: {e}")

    normalized_csv = None
    aligned_csv = None
    rejected_csv = None
    if make_csv_under:
        os.makedirs(make_csv_under, exist_ok=True)
        if normalized_rows:
            normalized_csv = os.path.join(make_csv_under, "normalized.csv")
            pd.DataFrame(normalized_rows).to_csv(normalized_csv, index=False)
        if aligned_rows:
            aligned_csv = os.path.join(make_csv_under, "aligned.csv")
            pd.DataFrame(aligned_rows).to_csv(aligned_csv, index=False)
        if write_rejected and rejected_rows:
            rejected_csv = os.path.join(make_csv_under, reject_csv_name)
            pd.DataFrame(rejected_rows).to_csv(rejected_csv, index=False)

    # Print small stats
    try:
        print(f"Aligned kept: {len(aligned_rows)}, rejected: {len(rejected_rows)}")
    except Exception:
        pass

    return normalized_csv, aligned_csv, rejected_csv


def main():
    ap = argparse.ArgumentParser(description="Dataset diagnostics for speech denoising pairs")
    ap.add_argument("--csv", required=True, help="CSV with clean_path,noisy_path columns")
    ap.add_argument("--sr", type=int, default=None, help="Target SR (info only, no resample)")
    ap.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs to analyze")
    ap.add_argument("--analyze-seconds", type=float, default=15.0, help="Limit per file seconds for speed (<=0 for full)")
    ap.add_argument("--max-delay-ms", type=float, default=100.0, help="Max delay search window for GCC-PHAT")
    ap.add_argument("--write-report", default="diagnostics_output", help="Folder to save CSV/plots")
    ap.add_argument("--volume-normalize", action="store_true", help="Enable RMS volume normalization output")
    ap.add_argument("--target-rms-db", type=float, default=-23.0, help="Target RMS (dBFS) for normalization")
    ap.add_argument("--headroom-db", type=float, default=1.0, help="Peak headroom after normalization (dB)")
    ap.add_argument("--volume-mode", choices=["common", "match_clean"], default="common", help="Volume mode: common target or match noisy to clean RMS")
    ap.add_argument("--write-adjusted", action="store_true", help="Write volume-normalized copies under --normalized-outdir")
    ap.add_argument("--normalized-outdir", default=None, help="Output dir for normalized copies (subfolders clean/noisy)")
    ap.add_argument("--write-aligned", action="store_true", help="Write time-aligned noisy copies under --aligned-outdir")
    ap.add_argument("--aligned-outdir", default=None, help="Output dir for aligned pairs (subfolders clean/noisy)")
    # Alignment acceptance thresholds
    ap.add_argument("--align-accept-gcc-min", type=float, default=0.18, help="Min GCC-PHAT peak to accept alignment")
    ap.add_argument("--align-accept-snr-improve-min", type=float, default=0.5, help="Min SNR(dB) improvement to accept alignment")
    ap.add_argument("--align-accept-delay-max-ms", type=float, default=80.0, help="Reject if |delay| exceeds this (ms)")
    ap.add_argument("--write-rejected", action="store_true", help="Write rejected pairs list to CSV")

    args = ap.parse_args()

    reports, summary = analyze_pairs(
        csv_path=args.csv,
        sr_target=args.sr,
        max_pairs=args.max_pairs,
        analyze_seconds=args.analyze_seconds,
        max_delay_ms=args.max_delay_ms,
    )

    print("Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Save report
    csv_path = write_report(reports, summary, args.write_report)
    print(f"Report saved to: {csv_path}")

    # Optional write outs
    if args.write_adjusted and not args.normalized_outdir:
        args.normalized_outdir = os.path.join(args.write_report, "normalized")
    if args.write_aligned and not args.aligned_outdir:
        args.aligned_outdir = os.path.join(args.write_report, "aligned")

    if args.write_adjusted or args.write_aligned:
        df = pd.DataFrame([asdict(r) for r in reports])
        normalized_csv, aligned_csv, rejected_csv = maybe_write_adjusted_and_aligned(
            df=df,
            reports=reports,
            write_adjusted=args.volume_normalize or args.write_adjusted,
            normalized_outdir=args.normalized_outdir,
            target_rms_db=args.target_rms_db,
            headroom_db=args.headroom_db,
            volume_mode=args.volume_mode,
            write_aligned=args.write_aligned,
            aligned_outdir=args.aligned_outdir,
            make_csv_under=args.write_report,
            orig_csv_path=args.csv,
            align_accept_gcc_min=args.align_accept_gcc_min,
            align_accept_snr_improve_min=args.align_accept_snr_improve_min,
            align_accept_delay_max_ms=args.align_accept_delay_max_ms,
            write_rejected=args.write_rejected,
        )
        print("Adjusted/aligned files written.")
        if normalized_csv:
            print(f"Normalized CSV: {normalized_csv}")
        if aligned_csv:
            print(f"Aligned CSV: {aligned_csv}")
        if rejected_csv:
            print(f"Rejected CSV: {rejected_csv}")


if __name__ == "__main__":
    main()
