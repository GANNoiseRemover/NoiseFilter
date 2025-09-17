# System Architecture (for GAN + Hearing Aid)

## 1. Data Collection
- **Noisy audio**: recorded through a microphone connected to the hearing aid prototype while playing sound (e.g., from a laptop speaker).  
- **Clean audio**: recorded in parallel from a phone or another “reference” system under the same conditions (so that the signals match).  
- **Audio format**: WAV, sampling rate 16 kHz, 16-bit PCM.  
  - WAV is preferred because it is uncompressed (unlike MP3).  

## 2. Preprocessing and Data Loading
- Audio files are separated into **clean** and **noisy** folders.  
- **librosa** library is used for loading and processing audio.  
- All files are normalized to the range [-1, 1].  
- Signals are split into 1-second segments (16,000 samples at 16 kHz).  

## 3. Training the Reference Model (MetricGAN+)
- **Framework**: TensorFlow, Keras  
- **Imports**: os, keras, numpy, librosa, soundfile, tensorflow.keras, pesq, pystoi  
- **Logic**:  
  1. **Generator** → receives noisy audio and attempts to denoise it.  
  2. **Discriminator** → predicts quality metrics (PESQ, STOI).  
  3. **Loss** → difference between predicted metrics and the true metrics.  
- **Goal**: train the generator to maximize PESQ/STOI for the output signal.  
- **Training metrics**:  
  - PESQ before/after → target metric for this criteria??? 
  - STOI before/after  - not  that important

## 4. Using PNNoisy (Student)
- After the reference model (MetricGAN+) is trained, a lighter **PNNoisy** model is trained:  
  - It learns to mimic the outputs of the MetricGAN+ generator but runs faster (necessary for deployment on the device).  
- Thus:  
  - **Teacher** = MetricGAN+  
  - **Student** = PNNoisy  

## 5. Saving and Sharing Results
- Save weights in **.h5** format:  
  ```python
  generator.save("generator_metricgan_full.h5")
- Sharing can be done via **GitHub** or **Google Drive**.  
- In the README, include instructions for loading the model:  
  ```python
  from tensorflow.keras.models import load_model
  generator = load_model("generator_metricgan_full.h5")
- This way, you share not only the code but also the trained model.  

## 6. Embedding in the Device
- After training, optimization is needed:  
  - Convert to **TensorFlow Lite (TFLite)** or **ONNX** for deployment on a microcontroller.   - WHICH ONE WE GONNA USE???
  - Optionally, perform **model quantization** (reducing weights from float32 to int8) to save resources.  
- The microchip/processor in the hearing aid will:  
  1. Capture audio from the microphone → buffer (16 kHz).  
  2. Process it through the model (TFLite/ONNX runtime).  
  3. Output the denoised signal to the speaker with minimal latency (<10 ms).  

## 7. Final Result
- **Input**: noisy real-time audio.  
- **Output**: amplified and denoised audio delivered directly to the hearing aid speaker.  
- The system works **autonomously**, without a laptop or internet, fully on the device.
