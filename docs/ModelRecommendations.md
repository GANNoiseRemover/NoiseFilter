# 모델 추천 / Model Recommendations

* https://arxiv.org/abs/2212.09019
  이건 GAN기반은 아니긴 한데, 성능이 좋다고 하네요. 일반적인 기준에서는 빠르다고 하는데 보청기 수준에서 돌아갈 정도로 모델이 작은지는 잘 모르겠네요. CPU(i7-9700)에서 RTF가 0.082라고 해요.


* The SEGAN model (the first Speech Enhancement GAN model) https://github.com/santi-pdp/segan
-> may not be the best option for our project
   old model -
   requires a lot of data -
   simple architecture +

* MetricGAN (and updated version MetricGAN+) https://github.com/JasonSWFu/MetricGAN ???
-> learning on PESQ and STOI criteria
-> real life use

* CGA-MGAN(Convolution-Augmented Gated-Attention MetricGAN) https://www.mdpi.com/1099-4300/25/4/628
  -> MetricGAN + CGAU 결합한 구조


DeepFilterNet  [GitHub Repository](https://github.com/Rikorose/DeepFilterNet)  
- Lightweight deep filtering framework for real-time full-band (48kHz) speech noise suppression.  
- Supports both Rust and Python with command-line tools and LADSPA plugins for real-time microphone noise suppression.  
- Pretrained models available; works on CPU and GPU, suitable for embedded environments such as hearing aids.

Cycle-Consitency-Audio-Noise-Filter  [GitHub Repository](https://github.com/bboycoi/Cycle-Consitency-Audio-Noise-Filter)  
- An open-source audio filtering project applying a modified CycleGAN architecture for speech noise removal.  
- Supports frequency and waveform transformations and can learn from asymmetric datasets.  
- Simple and effective for various noise conditions, ideal for research and rapid prototyping.

HiFi-GAN  [Hugging Face Model](https://huggingface.co/nvidia/tts_hifigan)  
- High-performance GAN vocoder model converting mel spectrograms into high-quality speech signals.  
- Offers dramatically faster inference speed compared to WaveNet, optimized for real-time speech synthesis and enhancement.  
- Uses a generator combined with multi-scale and multi-period discriminators for simultaneous model compression and speech quality.  
- Widely used for speech restoration, text-to-speech, and noise reduction enhancement applications.

RNnoisy - mentor recommendations?