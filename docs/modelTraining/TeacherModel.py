import os
import re
import numpy as np
import librosa
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# ----------------- User-configurable parameters -----------------
sr = 16000
segment_length = 16000  # 1 second segments
epochs = 50
batch_size = 1  # not used in current loop, left for future batchification
clean_folder = "./clean_testC"   # folder with clean audio
noisy_folder = "./noise_testC"   # folder with noisy audio
output_folder = "./denoised"
CHECKPOINT_DIR = "./checkpoints"  # directory for saving generator/discriminator weights
LOG_FILE = "training_log.csv"     # csv log file for losses

os.makedirs(output_folder, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------- Helpers: metric computation, audio loading -----------------
def load_audio(file_path, sr=sr):
    """Load audio file and normalize to [-1,1]."""
    audio, _ = librosa.load(file_path, sr=sr)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

def compute_metrics(clean_seg, denoised_seg):
    """
    Compute PESQ and STOI.
    clean_seg and denoised_seg are 1D numpy arrays in range [-1,1].
    Returns numpy array shape (1,2) = [[pesq, stoi]].
    """
    try:
        # convert to 16-bit PCM scaled values for PESQ
        pesq_score = pesq(sr, (clean_seg * 32768).astype(np.int16),
                             (denoised_seg * 32768).astype(np.int16), mode='wb')
    except Exception:
        pesq_score = 1.0
    stoi_score = stoi(clean_seg, denoised_seg, sr, extended=False)
    return np.array([[pesq_score, stoi_score]], dtype=np.float32)

# ----------------- MetricGAN+ architecture (generator & discriminator) -----------------
def causal_conv_block(x, filters, kernel_size, dilation_rate):
    """Residual causal convolution block used in both generator and discriminator."""
    conv = layers.Conv1D(filters, kernel_size, padding='causal',
                         dilation_rate=dilation_rate, activation='relu')(x)
    conv = layers.Conv1D(filters, kernel_size, padding='causal',
                         dilation_rate=dilation_rate, activation='relu')(conv)
    return layers.Add()([x, conv])

def build_generator(input_shape=(None,1), num_blocks=6):
    """Build generator model that outputs denoised waveform."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 31, padding='causal', activation='relu')(inp)
    for i in range(num_blocks):
        x = causal_conv_block(x, 64, 31, dilation_rate=2**i)
    out = layers.Conv1D(1, 31, padding='causal', activation='tanh')(x)
    return Model(inp, out, name="Generator")

def build_discriminator(input_shape=(None,1), num_blocks=6):
    """
    Build discriminator that predicts PESQ & STOI (two outputs).
    Compiled with MSE loss since we regress to metric values.
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 31, padding='causal', activation='relu')(inp)
    for i in range(num_blocks):
        x = causal_conv_block(x, 64, 31, dilation_rate=2**i)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(2, activation='linear')(x)  # PESQ and STOI
    model = Model(inp, out, name="Discriminator")
    model.compile(optimizer=Adam(1e-4), loss='mse')
    return model

# ----------------- File pairing verification -----------------
def extract_id_from_filename(fname):
    """
    Try to extract final digit-group (e.g. '0001') from filename; returns string key.
    If no digits found, return full basename (fallback).
    """
    m = re.search(r'(\d+)(?:\.\w+)?$', fname)
    if m:
        return m.group(1)
    return os.path.basename(fname)

def pair_clean_noisy(clean_files, noisy_files):
    """
    Build matched list of (clean_path, noisy_path) pairs based on numeric id extracted from filename.
    Raises if mismatch.
    """
    clean_map = {}
    for p in clean_files:
        key = extract_id_from_filename(os.path.basename(p))
        clean_map[key] = p

    noisy_map = {}
    for p in noisy_files:
        key = extract_id_from_filename(os.path.basename(p))
        noisy_map[key] = p

    # Intersection of keys
    common_keys = sorted(list(set(clean_map.keys()) & set(noisy_map.keys())))
    if not common_keys:
        raise ValueError("No matching file IDs found between clean and noisy folders.")

    # Warn if counts mismatch
    if len(common_keys) != len(clean_files) or len(common_keys) != len(noisy_files):
        print("Warning: Not all files have matching pairs. Using matched subset of size:", len(common_keys))

    pairs = [(clean_map[k], noisy_map[k]) for k in common_keys]
    # Sort pairs by numeric id for deterministic order
    try:
        pairs = sorted(pairs, key=lambda t: int(extract_id_from_filename(os.path.basename(t[0]))))
    except Exception:
        pairs = sorted(pairs, key=lambda t: extract_id_from_filename(os.path.basename(t[0])))
    return pairs

# ----------------- Checkpoint utilities -----------------
def extract_epoch_from_ckpt(fname, prefix):
    """
    Extract integer epoch from filenames like 'generator_epoch_12.h5'
    prefix should be 'generator' or 'discriminator'
    Returns epoch int or -1 if not matched.
    """
    pattern = rf'{re.escape(prefix)}_epoch_(\d+)\.h5$'
    m = re.search(pattern, fname)
    return int(m.group(1)) if m else -1

def find_latest_common_epoch(checkpoint_dir):
    """
    Find the maximum epoch number for which BOTH generator and discriminator weight files exist.
    Returns epoch_number (int) or 0 if none found.
    """
    files = os.listdir(checkpoint_dir)
    gen_epochs = {extract_epoch_from_ckpt(f, "generator") for f in files}
    disc_epochs = {extract_epoch_from_ckpt(f, "discriminator") for f in files}
    # Keep only positive epoch numbers
    gen_epochs = {e for e in gen_epochs if e >= 1}
    disc_epochs = {e for e in disc_epochs if e >= 1}
    common = gen_epochs & disc_epochs
    if not common:
        return 0
    return max(common)

def load_weights_for_epoch(generator, discriminator, checkpoint_dir, epoch):
    """
    Load generator and discriminator weights for a given epoch.
    Assumes files generator_epoch_{epoch}.h5 and discriminator_epoch_{epoch}.h5 exist.
    """
    gen_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.h5")
    disc_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.h5")
    if not os.path.exists(gen_path) or not os.path.exists(disc_path):
        raise FileNotFoundError(f"Checkpoint pair for epoch {epoch} not found in {checkpoint_dir}")
    generator.load_weights(gen_path)
    discriminator.load_weights(disc_path)
    print(f"Loaded weights: {gen_path} & {disc_path}")

# ----------------- Prepare file lists and verify pairing -----------------
clean_files_all = sorted([os.path.join(clean_folder, f) for f in os.listdir(clean_folder) if f.endswith('.wav')])
noisy_files_all = sorted([os.path.join(noisy_folder, f) for f in os.listdir(noisy_folder) if f.endswith('.wav')])

pairs = pair_clean_noisy(clean_files_all, noisy_files_all)
print(f"Using {len(pairs)} matched clean/noisy file pairs for training.")

# ----------------- Build models and try to resume -----------------
generator = build_generator()
discriminator = build_discriminator()  # compiled inside builder
discriminator.trainable = False  # ensure discriminator is non-trainable for metric_gan construction

# find latest epoch that has both generator and discriminator weights
latest_epoch = find_latest_common_epoch(CHECKPOINT_DIR)
start_epoch = latest_epoch  # last completed epoch number (0 if none)

if start_epoch > 0:
    # load weights (we restore only weights, not optimizer state)
    load_weights_for_epoch(generator, discriminator, CHECKPOINT_DIR, start_epoch)
    discriminator.trainable = False  # ensure it stays non-trainable
    print(f"✅ Resumed from epoch {start_epoch}. Next epoch will be {start_epoch + 1}")
else:
    print("No existing checkpoints found — training from scratch.")

# build metric_gan model (generator -> discriminator) used to train generator
input_layer = layers.Input(shape=(None,1))
generated_audio = generator(input_layer)
metric_pred = discriminator(generated_audio)
metric_gan = Model(input_layer, metric_pred)
metric_gan.compile(optimizer=Adam(1e-4), loss='mse')

# ensure log file exists (header only if new)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("epoch,avg_g_loss,avg_d_loss\n")
else:
    print(f"Existing log file found: {LOG_FILE}")

# ----------------- Training loop -----------------
for epoch in range(start_epoch + 1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    epoch_g_losses = []
    epoch_d_losses = []

    for clean_path, noisy_path in pairs:
        clean_audio = load_audio(clean_path)
        noisy_audio = load_audio(noisy_path)
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        num_segments = min_len // segment_length
        if num_segments == 0:
            print(f"  Skipping file (too short): {os.path.basename(clean_path)}")
            continue

        file_g_losses = []
        file_d_losses = []
        for i in range(num_segments):
            s = i * segment_length
            e = s + segment_length
            clean_seg = clean_audio[s:e].reshape(1, -1, 1)
            noisy_seg = noisy_audio[s:e].reshape(1, -1, 1)

            # ---- Discriminator training ----
            # use generator in inference mode to produce denoised audio (no gradients)
            denoised_seg_tf = generator(noisy_seg, training=False)
            denoised_np = denoised_seg_tf.numpy().flatten()
            metric_target = compute_metrics(clean_seg.flatten(), denoised_np)
            # train discriminator to regress to the true metrics
            d_loss = discriminator.train_on_batch(denoised_seg_tf, metric_target)

            # ---- Generator training ----
            # train generator through metric_gan (discriminator fixed)
            g_loss = metric_gan.train_on_batch(noisy_seg, metric_target)

            # collect stats
            file_g_losses.append(float(g_loss))
            file_d_losses.append(float(d_loss))

        # aggregate per-file losses and print
        mean_g_file = np.mean(file_g_losses) if file_g_losses else float('nan')
        mean_d_file = np.mean(file_d_losses) if file_d_losses else float('nan')
        print(f"  File {os.path.basename(clean_path)} -> G_loss: {mean_g_file:.4f}, D_loss: {mean_d_file:.4f}")

        epoch_g_losses.extend(file_g_losses)
        epoch_d_losses.extend(file_d_losses)

    # end of epoch
    avg_g = np.mean(epoch_g_losses) if epoch_g_losses else 0.0
    avg_d = np.mean(epoch_d_losses) if epoch_d_losses else 0.0
    print(f"Epoch {epoch} finished. Avg G_loss: {avg_g:.6f}, Avg D_loss: {avg_d:.6f}")

    # append log (no duplication because epoch numbering comes from checkpoint)
    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_g:.6f},{avg_d:.6f}\n")

    # save checkpoints every N epochs (5 here)
    if epoch % 5 == 0 or epoch == epochs:
        gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.h5")
        disc_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.h5")
        generator.save_weights(gen_path)
        discriminator.save_weights(disc_path)
        print(f"✅ Saved checkpoints: {gen_path} , {disc_path}")

# ----------------- Batch Denoising after training -----------------
print("\nStarting batch denoising using final generator weights...")
for _, noisy_path in pairs:
    noisy_audio = load_audio(noisy_path)
    noisy_input = noisy_audio.reshape(1, -1, 1)
    denoised_audio = generator(noisy_input, training=False).numpy().flatten()
    out_path = os.path.join(output_folder, os.path.basename(noisy_path))
    sf.write(out_path, denoised_audio, sr)
    print(f"Denoised {os.path.basename(noisy_path)} -> {out_path}")

print("\nAll done.")
