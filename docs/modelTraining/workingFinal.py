from google.colab import drive
drive.mount('/content/drive')

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
import csv

# ----------------- Parameters -----------------
sr = 16000
segment_length = 16000
epochs = 100
batch_size = 16
clean_folder = "/content/drive/MyDrive/MetricGAN+/clean_testC"
noisy_folder = "/content/drive/MyDrive/MetricGAN+/noise_testC"
output_folder = "/content/drive/MyDrive/MetricGAN+/denoised_new"
CHECKPOINT_DIR = "/content/drive/MyDrive/MetricGAN+/checkpoints"
LOG_FILE = "/content/drive/MyDrive/MetricGAN+/training_log.csv"
METRIC_FILE = "/content/drive/MyDrive/MetricGAN+/epoch_metrics.csv"

METRIC_WEIGHT = 0.05
RECONSTRUCTION_WEIGHT = 1.0
METRIC_RAMPUP_EPOCHS = 20

os.makedirs(output_folder, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------- Additional func -----------------
def load_audio(file_path, sr=sr):
    audio, _ = librosa.load(file_path, sr=sr)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio

def compute_metrics(clean_seg, denoised_seg):
    try:
        pesq_score = pesq(sr, (clean_seg*32768).astype(np.int16),
                             (denoised_seg*32768).astype(np.int16), mode='wb')
    except:
        pesq_score = 1.0
    stoi_score = stoi(clean_seg, denoised_seg, sr, extended=False)
    return np.array([pesq_score, stoi_score], dtype=np.float32)

def causal_conv_block(x, filters, kernel_size, dilation_rate):
    conv = layers.Conv1D(filters, kernel_size, padding='causal',
                         dilation_rate=dilation_rate, activation='relu')(x)
    conv = layers.Conv1D(filters, kernel_size, padding='causal',
                         dilation_rate=dilation_rate, activation='relu')(conv)
    return layers.Add()([x, conv])

def build_generator(input_shape=(None,1), num_blocks=6):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 31, padding='causal', activation='relu')(inp)
    for i in range(num_blocks):
        x = causal_conv_block(x, 64, 31, dilation_rate=2**i)
    out = layers.Conv1D(1, 31, padding='causal', activation=None)(x)
    return Model(inp, out, name="Generator")

def build_discriminator(input_shape=(None,1), num_blocks=6):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 31, padding='causal', activation='relu')(inp)
    for i in range(num_blocks):
        x = causal_conv_block(x, 64, 31, dilation_rate=2**i)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(2, activation='linear')(x)
    model = Model(inp, out, name="Discriminator")
    return model

def extract_id_from_filename(fname):
    m = re.search(r'(\d+)(?:\.\w+)?$', fname)
    return m.group(1) if m else os.path.basename(fname)

def pair_clean_noisy(clean_files, noisy_files):
    clean_map = {extract_id_from_filename(os.path.basename(f)): f for f in clean_files}
    noisy_map = {extract_id_from_filename(os.path.basename(f)): f for f in noisy_files}
    common_keys = sorted(list(set(clean_map.keys()) & set(noisy_map.keys())))
    if not common_keys:
        raise ValueError("No matching file IDs found.")
    if len(common_keys) != len(clean_files) or len(common_keys) != len(noisy_files):
        print(f"Warning: Not all files have matching pairs. Using subset: {len(common_keys)}")
    pairs = [(clean_map[k], noisy_map[k]) for k in common_keys]
    try:
        pairs = sorted(pairs, key=lambda t: int(extract_id_from_filename(os.path.basename(t[0]))))
    except:
        pairs = sorted(pairs, key=lambda t: extract_id_from_filename(os.path.basename(t[0])))
    return pairs

# ----------------- File Preparation -----------------
clean_files_all = sorted([os.path.join(clean_folder, f) for f in os.listdir(clean_folder) if f.endswith('.wav')])
noisy_files_all = sorted([os.path.join(noisy_folder, f) for f in os.listdir(noisy_folder) if f.endswith('.wav')])
pairs = pair_clean_noisy(clean_files_all, noisy_files_all)
print(f"Using {len(pairs)} matched clean/noisy file pairs.")

# ----------------- Models -----------------
generator = build_generator()
discriminator = build_discriminator()

#d_optimizer = Adam(1e-4)
#g_optimizer = Adam(5e-5) #!!!
d_optimizer = Adam(5e-5)   # ↓ в 2 раза
g_optimizer = Adam(1e-4)   # ↑ в 2 раза

loss_fn = tf.keras.losses.MeanSquaredError()

# ----------------- Resume training -----------------
def get_last_epoch(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("generator_epoch_")]
    if not files:
        return 0
    epochs_list = [int(re.search(r"(\d+)", f).group(1)) for f in files]
    return max(epochs_list)

last_epoch = get_last_epoch(CHECKPOINT_DIR)

if last_epoch > 0:
    gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{last_epoch}.weights.h5")
    disc_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{last_epoch}.weights.h5")
    generator.load_weights(gen_path)
    discriminator.load_weights(disc_path)
    print(f"✅ Resuming training from epoch {last_epoch}")
else:
    print("ℹ️ Starting training from scratch")

# ----------------- Logg files -----------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("epoch,avg_g_loss,avg_d_loss\n")

if not os.path.exists(METRIC_FILE):
    with open(METRIC_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        # Добавлено логирование по файлам и отдельные лоссы для дискриминатора
        writer.writerow(["epoch", "file_name", "G_loss_total", "D_loss_total", "D_loss_real", "D_loss_gen", "PESQ", "STOI"])


# ----------------- Training loop -----------------
for epoch in range(last_epoch + 1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")

    if epoch > METRIC_RAMPUP_EPOCHS:
    # увеличиваем вес раз в 5 эпох, шаг 0.05, максимум = 1.0
    #increse the metric weight (once every 5 epoch) - firstly wwe need to observe the results and then change it for out needs
        metric_weight_current = METRIC_WEIGHT + ((epoch - METRIC_RAMPUP_EPOCHS+5) // 5) * 0.05 # ВЕС
        metric_weight_current = min(metric_weight_current, 1.0)
    else:
        metric_weight_current = METRIC_WEIGHT

    print(f"  Metric weight: {metric_weight_current:.2f}")

    epoch_g_losses_total = []
    epoch_d_losses_total = []

    for clean_path, noisy_path in pairs:
        clean_audio = load_audio(clean_path)
        noisy_audio = load_audio(noisy_path)
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        num_segments = min_len // segment_length
        if num_segments == 0:
            continue

        file_g_losses = []
        file_d_losses = []
        file_d_real_losses = []
        file_d_gen_losses = []

        batch_clean = []
        batch_noisy = []

        for i in range(num_segments):
            s, e = i * segment_length, (i + 1) * segment_length
            batch_clean.append(clean_audio[s:e])
            batch_noisy.append(noisy_audio[s:e])

            if len(batch_clean) == batch_size:
                batch_clean_np = np.array(batch_clean).reshape(batch_size, segment_length, 1)
                batch_noisy_np = np.array(batch_noisy).reshape(batch_size, segment_length, 1)

                # --- 1. Обучение дискриминатора ---
                with tf.GradientTape() as d_tape:
                    denoised_tf = generator(batch_noisy_np, training=True)

                    metric_target_gen = tf.constant(np.array([compute_metrics(c.flatten(), d.flatten())
                                                              for c, d in zip(batch_clean_np, denoised_tf.numpy())]), dtype=tf.float32)

                    disc_pred_gen = discriminator(denoised_tf, training=True)
                    d_loss_gen = loss_fn(disc_pred_gen, metric_target_gen)

                    metric_target_real = tf.constant(np.array([4.5, 1.0]), dtype=tf.float32)
                    metric_target_real = tf.repeat(tf.expand_dims(metric_target_real, axis=0), batch_size, axis=0)

                    disc_pred_real = discriminator(batch_clean_np, training=True)
                    d_loss_real = loss_fn(disc_pred_real, metric_target_real)

                    d_loss_total = d_loss_gen + d_loss_real

                d_grads = d_tape.gradient(d_loss_total, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

                # --- 2. Обучение генератора ---
                with tf.GradientTape() as g_tape:
                    denoised_tf = generator(batch_noisy_np, training=True)
                    disc_pred_gen = discriminator(denoised_tf, training=False)

                    metric_target_high = tf.constant(np.array([4.5, 1.0]), dtype=tf.float32)
                    metric_target_high = tf.repeat(tf.expand_dims(metric_target_high, axis=0), batch_size, axis=0)
                    g_loss_metric = loss_fn(disc_pred_gen, metric_target_high)

                    g_loss_reconst = loss_fn(denoised_tf, batch_clean_np)

                    g_loss_total = metric_weight_current * g_loss_metric + RECONSTRUCTION_WEIGHT * g_loss_reconst

                g_grads = g_tape.gradient(g_loss_total, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

                epoch_g_losses_total.append(float(g_loss_total))
                epoch_d_losses_total.append(float(d_loss_total))

                file_g_losses.append(float(g_loss_total))
                file_d_losses.append(float(d_loss_total))
                file_d_real_losses.append(float(d_loss_real))
                file_d_gen_losses.append(float(d_loss_gen))

                batch_clean, batch_noisy = [], []

        if batch_clean:
            batch_clean_np = np.array(batch_clean).reshape(len(batch_clean), segment_length, 1)
            batch_noisy_np = np.array(batch_noisy).reshape(len(batch_noisy), segment_length, 1)

            with tf.GradientTape() as d_tape:
                denoised_tf = generator(batch_noisy_np, training=True)
                metric_target_gen = tf.constant(np.array([compute_metrics(c.flatten(), d.flatten())
                                                          for c, d in zip(batch_clean_np, denoised_tf.numpy())]), dtype=tf.float32)

                disc_pred_gen = discriminator(denoised_tf, training=True)
                d_loss_gen = loss_fn(disc_pred_gen, metric_target_gen)

                metric_target_real = tf.constant(np.array([4.5, 1.0]), dtype=tf.float32)
                metric_target_real = tf.repeat(tf.expand_dims(metric_target_real, axis=0), len(batch_clean), axis=0)

                disc_pred_real = discriminator(batch_clean_np, training=True)
                d_loss_real = loss_fn(disc_pred_real, metric_target_real)
                d_loss_total = d_loss_gen + d_loss_real

            d_grads = d_tape.gradient(d_loss_total, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            with tf.GradientTape() as g_tape:
                denoised_tf = generator(batch_noisy_np, training=True)
                disc_pred_gen = discriminator(denoised_tf, training=False)

                metric_target_high = tf.constant(np.array([4.5, 1.0]), dtype=tf.float32)
                metric_target_high = tf.repeat(tf.expand_dims(metric_target_high, axis=0), len(batch_clean), axis=0)
                g_loss_metric = loss_fn(disc_pred_gen, metric_target_high)

                g_loss_reconst = loss_fn(denoised_tf, batch_clean_np)
                g_loss_total = metric_weight_current * g_loss_metric + RECONSTRUCTION_WEIGHT * g_loss_reconst

            g_grads = g_tape.gradient(g_loss_total, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

            epoch_g_losses_total.append(float(g_loss_total))
            epoch_d_losses_total.append(float(d_loss_total))

            file_g_losses.append(float(g_loss_total))
            file_d_losses.append(float(d_loss_total))
            file_d_real_losses.append(float(d_loss_real))
            file_d_gen_losses.append(float(d_loss_gen))

        # --- Print loss records for each file ---
        file_name = os.path.basename(clean_path)
        mean_g_loss = np.mean(file_g_losses) if file_g_losses else float('nan')
        mean_d_loss = np.mean(file_d_losses) if file_d_losses else float('nan')
        mean_d_real_loss = np.mean(file_d_real_losses) if file_d_real_losses else float('nan')
        mean_d_gen_loss = np.mean(file_d_gen_losses) if file_d_gen_losses else float('nan')

        print(f"  File {file_name} -> G_loss: {mean_g_loss:.4f}, D_loss_total: {mean_d_loss:.4f}")

        # Calculating metrics for the current file
        noisy_audio_eval = load_audio(noisy_path)
        clean_audio_eval = load_audio(clean_path)
        denoised_audio_eval = generator(noisy_audio_eval.reshape(1, -1, 1), training=False).numpy().flatten()
        min_len_eval = min(len(clean_audio_eval), len(denoised_audio_eval))
        metrics = compute_metrics(clean_audio_eval[:min_len_eval], denoised_audio_eval[:min_len_eval])
        pesq_file = metrics[0]
        stoi_file = metrics[1]

        # Record to  METRIC_FILE
        with open(METRIC_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, file_name, mean_g_loss, mean_d_loss, mean_d_real_loss, mean_d_gen_loss, pesq_file, stoi_file])


    # ---- Records of final metrics for the epoch ----
    avg_g = np.mean(epoch_g_losses_total)
    avg_d = np.mean(epoch_d_losses_total)

    # ---- Average PESQ & STOI for an epoch ----
    pesq_epoch = []
    stoi_epoch = []
    with open(METRIC_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["epoch"]) == epoch:
                try:
                    pesq_epoch.append(float(row["PESQ"]))
                    stoi_epoch.append(float(row["STOI"]))
                except:
                    pass
    avg_pesq = np.mean(pesq_epoch) if pesq_epoch else float('nan')
    avg_stoi = np.mean(stoi_epoch) if stoi_epoch else float('nan')

    print(f"\nEpoch {epoch} finished.")
    print(f"  Avg G_loss: {avg_g:.6f}, Avg D_loss: {avg_d:.6f}")
    print(f"  Avg PESQ: {avg_pesq:.3f}, Avg STOI: {avg_stoi:.3f}")

    # ---- LOgging ----
    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_g:.6f},{avg_d:.6f},{avg_pesq:.3f},{avg_stoi:.3f}\n")

    # ---- Checkpoints ----
    gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.weights.h5")
    disc_path = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.weights.h5")
    generator.save_weights(gen_path)
    discriminator.save_weights(disc_path)
    print(f"✅ Saved checkpoints: {gen_path}, {disc_path}")

    # ---- Save denoised audio ----
    epoch_output_dir = os.path.join(output_folder, f"epoch_{epoch}")
    os.makedirs(epoch_output_dir, exist_ok=True)
    for _, noisy_path in pairs:
        noisy_audio = load_audio(noisy_path)
        denoised_audio = generator(noisy_audio.reshape(1, -1, 1), training=False).numpy().flatten()
        out_path = os.path.join(epoch_output_dir, os.path.basename(noisy_path))
        sf.write(out_path, denoised_audio, sr)
