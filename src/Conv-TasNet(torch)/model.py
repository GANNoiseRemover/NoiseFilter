import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)
    입력 (B, C, T)에 대해 샘플 단위로 (C, T) 전체를 정규화합니다.
    TasNet 계열에서 널리 사용됩니다.
    """
    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.norm = GlobalLayerNorm(out_channels)
        self.prelu = nn.PReLU()
        # Squeeze-and-Excitation block: 채널별 어텐션을 학습합니다.
        # reduction=8은 파라미터가 적고 실험적으로도 효과적입니다.
        class SEBlock(nn.Module):
            def __init__(self, channels, reduction=8):
                super().__init__()
                hidden = max(1, channels // reduction)
                # 1x1 conv를 FC 대용으로 사용 (시간 차원 유지)
                self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
                self.relu = nn.ReLU(inplace=True)
                self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # x: (B, C, T)
                s = x.mean(dim=2, keepdim=True)  # (B, C, 1)
                s = self.fc1(s)
                s = self.relu(s)
                s = self.fc2(s)
                s = self.sigmoid(s)
                return x * s

        # instantiate SE module for this conv's output channels
        self.se = SEBlock(out_channels, reduction=8)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        # apply channel-wise attention (SE)
        x = self.se(x)
        x = self.norm(x)
        x = self.prelu(x)
        return x

class TCNBlock(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_size, dilation, skip_channels=None, dropout=0.1):
        super(TCNBlock, self).__init__()
        # 1x1 bottleneck -> depthwise separable -> split to residual/skip (Conv-TasNet style)
        self.conv1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(conv_channels)
        self.d_conv = DepthwiseSeparableConv(
            conv_channels,
            conv_channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(conv_channels)
        # Separate heads for residual and skip (skip dim can be smaller)
        self.res_out = nn.Conv1d(conv_channels, in_channels, 1)
        self.skip_channels = in_channels if skip_channels is None else int(skip_channels)
        self.skip_out = nn.Conv1d(conv_channels, self.skip_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        h = self.conv1(x)
        h = self.prelu1(h)
        h = self.norm1(h)
        h = self.d_conv(h)
        # Stabilize before heads
        h = self.prelu2(h)
        h = self.norm2(h)
        res = self.res_out(h)
        skip = self.skip_out(h)
        skip = self.dropout(skip)
        return identity + res, skip

class ConvTasNet(nn.Module):
    def __init__(
        self,
        enc_dim=512,
        win_len=16,
        num_spk=1,
        num_layers=3,
        num_blocks=8,
        conv_channels=512,
        kernel_size=3,
        skip_channels=None,
        mask_activation: str = 'sigmoid',
        use_refiner: bool = False,
        refiner_channels: int = 32,
        use_skip_gates: bool = False,
        # New options
        use_dual_mask_head: bool = False,
        use_postfilter: bool = False,
    ):
        super(ConvTasNet, self).__init__()
        self.enc_dim = enc_dim
        self.win_len = win_len
        self.num_spk = num_spk
        self.skip_channels = enc_dim if skip_channels is None else int(skip_channels)
        self.mask_activation = mask_activation.lower()
        self.use_refiner = bool(use_refiner)
        self.use_skip_gates = bool(use_skip_gates)
        self.use_dual_mask_head = bool(use_dual_mask_head)
        self.use_postfilter = bool(use_postfilter)

        self.encoder = nn.Conv1d(1, enc_dim, win_len, bias=False, stride=win_len // 2)
        # gLN으로 교체: 입력 (B, C, T) 그대로 처리
        self.ln = GlobalLayerNorm(enc_dim)
        self.bottleneck = nn.Conv1d(enc_dim, enc_dim, 1)

        # Use a global expanding dilation schedule across all blocks (not resetting per layer)
        # This increases receptive field without changing parameter count and can help PESQ.
        self.tcn_blocks = nn.ModuleList()
        total_blocks = num_layers * num_blocks
        for idx in range(total_blocks):
            dil = 2 ** idx
            # cap dilation to avoid excessive padding/instability; cycle after reaching a max exponent
            if dil > 512:
                # cycle within [1..512]
                # find exponent modulo range so that pattern repeats
                from math import log2
                max_exp = int(log2(512))
                exp = (idx % (max_exp + 1))
                dil = 2 ** exp
            self.tcn_blocks.append(TCNBlock(enc_dim, conv_channels, kernel_size, dilation=dil, skip_channels=self.skip_channels))

        # Optional learnable positive gates for each skip connection
        if self.use_skip_gates:
            # initialize near-zero so early training isn't dominated
            self.skip_gates = nn.Parameter(torch.zeros(len(self.tcn_blocks), dtype=torch.float32))
        else:
            self.register_parameter('skip_gates', None)

        # mask는 skip 경로의 피처를 사용해 예측합니다 (더 가벼운 skip로 파라미터 절약)
        self.mask_conv = nn.Conv1d(self.skip_channels, num_spk * enc_dim, 1)
        # Optional second mask head for activation ensemble (sigmoid + softplus)
        if self.use_dual_mask_head:
            self.mask_conv_2 = nn.Conv1d(self.skip_channels, num_spk * enc_dim, 1)
            # learnable blend between [0,1] via sigmoid
            self.mask_blend = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.register_parameter('mask_blend', None)
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, win_len, bias=False, stride=win_len // 2)

        # Lightweight time-domain residual post-refiner (very small parameter increase)
        # Simple structure: Conv1d(1 -> refiner_channels, k=3) -> PReLU -> Conv1d(refiner_channels -> 1, k=1)
        # Applied as decoded + alpha * refiner(decoded) where alpha is a small learnable scalar.
        if self.use_refiner:
            # deeper but still lightweight refiner: two 1D conv layers to model fine residual corrections
            class RefinerBlock(nn.Module):
                def __init__(self, in_ch, mid_ch):
                    super().__init__()
                    self.conv1 = nn.Conv1d(in_ch, mid_ch, kernel_size=3, padding=1)
                    self.prelu1 = nn.PReLU()
                    self.conv2 = nn.Conv1d(mid_ch, mid_ch, kernel_size=3, padding=1)
                    self.prelu2 = nn.PReLU()
                    self.conv3 = nn.Conv1d(mid_ch, 1, kernel_size=1)

                def forward(self, x):
                    # x: (B, 1, T)
                    h = self.conv1(x)
                    h = self.prelu1(h)
                    h = self.conv2(h)
                    h = self.prelu2(h)
                    out = self.conv3(h)
                    return out

            self.refiner = RefinerBlock(1, int(refiner_channels))
            # small initialized scale to avoid destabilizing pre-trained weights
            self.refiner_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Optional lightweight differentiable Wiener-like post-filter in STFT domain
        if self.use_postfilter:
            # single learnable strength parameter; softplus to keep positive
            self.postfilter_strength = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        else:
            self.register_parameter('postfilter_strength', None)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        
        mixture_w = self.encoder(x)
        
        norm_mixture_w = self.ln(mixture_w)
        bottleneck_output = self.bottleneck(norm_mixture_w)

        tcn_output = bottleneck_output
        skip_sum = None
        for i, block in enumerate(self.tcn_blocks):
            tcn_output, skip = block(tcn_output)
            # ensure positive gating; softplus keeps it >0 and is smooth
            if self.skip_gates is not None:
                gate = F.softplus(self.skip_gates[i])
                skip = skip * gate
            if skip_sum is None:
                skip_sum = skip
            else:
                skip_sum = skip_sum + skip

        feat_for_mask = skip_sum if skip_sum is not None else tcn_output
        masks_1 = self.mask_conv(feat_for_mask)
        # 기본 활성화
        def _activate(m):
            if self.mask_activation == 'relu':
                return F.relu(m)
            elif self.mask_activation == 'softplus':
                return F.softplus(m)
            else:
                return torch.sigmoid(m)
        masks_1 = _activate(masks_1)
        if self.use_dual_mask_head:
            # second head with complementary activation
            masks_2 = self.mask_conv_2(feat_for_mask)
            # choose activation different from primary for diversity
            if self.mask_activation == 'softplus':
                masks_2 = torch.sigmoid(masks_2)
            else:
                masks_2 = F.softplus(masks_2)
            w = torch.sigmoid(self.mask_blend)
            masks = w * masks_1 + (1.0 - w) * masks_2
        else:
            masks = masks_1
        masks = masks.view(x.shape[0], self.num_spk, self.enc_dim, -1)
        
        estimated_sources = masks * mixture_w.unsqueeze(1)
        
        if self.num_spk == 1:
            estimated_sources = estimated_sources.squeeze(1)

        decoded_sources = self.decoder(estimated_sources)
        
        # Ensure output length matches input length (pad or crop safely)
        T = x.shape[-1]
        out_len = decoded_sources.shape[-1]
        if out_len < T:
            decoded_sources = F.pad(decoded_sources, (0, T - out_len))
        elif out_len > T:
            decoded_sources = decoded_sources[..., :T]
        
        # Apply post-refiner if enabled
        if self.use_refiner:
            # refiner expects (B, 1, T)
            ref_out = self.refiner(decoded_sources)
            decoded_sources = decoded_sources + self.refiner_scale * ref_out

        # Optional Wiener-like post-filter in STFT domain using residual estimate
        if self.use_postfilter:
            # Compute residual (approx noise) and build Wiener gain in FP32 outside autocast
            B, _, T = decoded_sources.shape
            mix = x
            if mix.shape[-1] != T:
                if mix.shape[-1] < T:
                    mix = F.pad(mix, (0, T - mix.shape[-1]))
                else:
                    mix = mix[..., :T]
            resid = (mix - decoded_sources).squeeze(1)
            sig = decoded_sources.squeeze(1)
            # Disable autocast for complex STFT ops and force float32
            try:
                ctx = torch.amp.autocast('cuda', enabled=False) if sig.is_cuda else torch.cuda.amp.autocast(enabled=False)
            except Exception:
                # Fallback: no context manager
                class _N:
                    def __enter__(self):
                        return None
                    def __exit__(self, *args):
                        return False
                ctx = _N()
            with ctx:
                sig32 = sig.float()
                resid32 = resid.float()
                win = 1024
                hop = 256
                window = torch.hann_window(win, device=sig32.device, dtype=sig32.dtype)
                S = torch.stft(sig32, n_fft=win, hop_length=hop, win_length=win, window=window, return_complex=True, center=True, pad_mode='reflect')
                R = torch.stft(resid32, n_fft=win, hop_length=hop, win_length=win, window=window, return_complex=True, center=True, pad_mode='reflect')
                Syy = (S.real**2 + S.imag**2)
                Rnn = (R.real**2 + R.imag**2)
                alpha = F.softplus(self.postfilter_strength) + 1e-5
                G = Syy / (Syy + alpha * Rnn + 1e-8)
                S_filt = G * S
                y_filt = torch.istft(S_filt, n_fft=win, hop_length=hop, win_length=win, window=window, center=True)
            if y_filt.shape[-1] < T:
                y_filt = F.pad(y_filt, (0, T - y_filt.shape[-1]))
            elif y_filt.shape[-1] > T:
                y_filt = y_filt[..., :T]
            y_filt = y_filt.unsqueeze(1)
            decoded_sources = y_filt

        return decoded_sources


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [spectral_norm(nn.Conv1d(in_filters, out_filters, 3, 2, 1)), nn.LeakyReLU(0.2, inplace=False)]
            if bn:
                block.append(nn.InstanceNorm1d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=False),
            # [핵심 수정] 시간 축에 대해 평균 풀링을 수행하여 (B, C, T) -> (B, C, 1)로 만듭니다.
            nn.AdaptiveAvgPool1d(1),
            # 최종 출력을 (B, 1, 1)로 만듭니다.
            spectral_norm(nn.Conv1d(128, 1, kernel_size=1))
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.model(x)

