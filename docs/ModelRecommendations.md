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
  MetricGAN + CGAU 결합한 구조
