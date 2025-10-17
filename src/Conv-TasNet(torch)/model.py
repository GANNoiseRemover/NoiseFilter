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
    def __init__(self, in_channels, conv_channels, kernel_size, dilation):
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
        # Separate heads for residual and skip
        self.res_out = nn.Conv1d(conv_channels, in_channels, 1)
        self.skip_out = nn.Conv1d(conv_channels, in_channels, 1)
        self.dropout = nn.Dropout(0.1)

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
    def __init__(self, enc_dim=512, win_len=16, num_spk=1, num_layers=3, num_blocks=8, conv_channels=512, kernel_size=3, max_dilation=128, use_refiner: bool = True, refiner_channels: int = 64):
        super(ConvTasNet, self).__init__()
        self.enc_dim = enc_dim
        self.win_len = win_len
        self.num_spk = num_spk

        self.encoder = nn.Conv1d(1, enc_dim, win_len, bias=False, stride=win_len // 2)
        # gLN으로 교체: 입력 (B, C, T) 그대로 처리
        self.ln = GlobalLayerNorm(enc_dim)
        self.bottleneck = nn.Conv1d(enc_dim, enc_dim, 1)

        # Use a global cumulative dilation schedule across layers/blocks
        # This increases receptive field without changing parameter count.
        self.tcn_blocks = nn.ModuleList()
        global_idx = 0
        for _ in range(num_layers):
            for i in range(num_blocks):
                dilation = min(2 ** global_idx, max_dilation)
                self.tcn_blocks.append(TCNBlock(enc_dim, conv_channels, kernel_size, dilation=dilation))
                global_idx += 1

        # Learnable per-block skip gates (scalar per block). Using a small number
        # of parameters to gate skip contributions allows the network to downweight
        # harmful skip signals while keeping overall parameter cost negligible.
        self.skip_gates = nn.Parameter(torch.ones(len(self.tcn_blocks)))

        self.mask_conv = nn.Conv1d(enc_dim, num_spk * enc_dim, 1)
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, win_len, bias=False, stride=win_len // 2)

        # Lightweight time-domain residual post-refiner (very small parameter increase)
        # Simple structure: Conv1d(1 -> refiner_channels, k=3) -> PReLU -> Conv1d(refiner_channels -> 1, k=1)
        # Applied as decoded + alpha * refiner(decoded) where alpha is a small learnable scalar.
        self.use_refiner = use_refiner
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

            self.refiner = RefinerBlock(1, refiner_channels)
            # small initialized scale to avoid destabilizing pre-trained weights
            self.refiner_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

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
            gate = F.softplus(self.skip_gates[i])
            skip = skip * gate
            if skip_sum is None:
                skip_sum = skip
            else:
                skip_sum = skip_sum + skip

        feat_for_mask = skip_sum if skip_sum is not None else tcn_output
        masks = self.mask_conv(feat_for_mask)
        masks = torch.sigmoid(masks).view(x.shape[0], self.num_spk, self.enc_dim, -1)
        
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

