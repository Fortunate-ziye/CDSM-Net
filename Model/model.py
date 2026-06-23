import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError("Please install the official package: pip install mamba-ssm") from exc


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 1-D feature sequences."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.transpose(0, 1).unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :, : x.size(2)]


class SEBlock(nn.Module):
    """Squeeze-and-excitation block for channel recalibration."""

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = max(in_channels // reduction, 4)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, reduced_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pool(x)
        return x * self.fc(y)


class DSConv1d(nn.Module):
    """Depthwise separable 1-D convolution."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size,
            stride,
            padding,
            groups=in_ch,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class BlurPool1d(nn.Module):
    """Anti-aliased downsampling for 1-D features."""

    def __init__(self, channels, stride=4):
        super().__init__()
        filt = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        filt = filt / filt.sum()
        self.register_buffer("filter", filt[None, None, :].repeat((channels, 1, 1)))
        self.stride = stride
        self.groups = channels

    def forward(self, x):
        x_pad = F.pad(x, (2, 2), mode="reflect")
        return F.conv1d(x_pad, self.filter, stride=self.stride, groups=self.groups)


class ResBlockOptimized(nn.Module):
    """Residual block with separable convolutions and SE attention."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = DSConv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm1d(out_ch, affine=True)

        self.conv2 = DSConv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm1d(out_ch, affine=True)

        self.se = SEBlock(out_ch)
        self.act = nn.GELU()

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm1d(out_ch, affine=True),
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.InstanceNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Small residual-branch scale stabilizes early optimization.
        nn.init.constant_(self.norm2.weight, 1e-2)

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.act(out)


class BiMambaAdapter(nn.Module):
    """Bidirectional Mamba mixer implemented with forward and reversed scans."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x):
        out_fwd = self.mamba_fwd(x)
        x_rev = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd, dims=[1])
        combined = torch.cat([out_fwd, out_bwd], dim=-1)
        return self.norm(self.fusion(combined))


class AttentionGateLite(nn.Module):
    """Lightweight additive attention gate for decoder skip features."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv1d(F_g, F_int, 1), nn.InstanceNorm1d(F_int, affine=True))
        self.W_x = nn.Sequential(nn.Conv1d(F_l, F_int, 1), nn.InstanceNorm1d(F_int, affine=True))
        self.psi = nn.Sequential(nn.Conv1d(F_int, 1, 1), nn.InstanceNorm1d(1, affine=True), nn.Sigmoid())
        self.act = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2] != x1.shape[2]:
            g1 = F.interpolate(g1, size=x1.shape[2], mode="linear", align_corners=False)
        psi = self.act(g1 + x1)
        return x * self.psi(psi)


class DecoderBlockOptimized(nn.Module):
    """Upsampling decoder block with attention-gated skip fusion."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            DSConv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
        )

        self.att_gate = AttentionGateLite(
            F_g=in_channels,
            F_l=skip_channels,
            F_int=max(skip_channels // 4, 1),
        )

        self.conv = nn.Sequential(
            DSConv1d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.GELU(),
            DSConv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, x, skip_connection):
        x_up = self.upsample(x)

        if x_up.size(2) != skip_connection.size(2):
            x_up = F.interpolate(
                x_up,
                size=skip_connection.size(2),
                mode="linear",
                align_corners=False,
            )

        skip_filtered = self.att_gate(g=x_up, x=skip_connection)
        return self.conv(torch.cat([x_up, skip_filtered], dim=1))


class DecoderBranchDeepSupOptimized(nn.Module):
    """Task-specific decoder branch with an auxiliary deep-supervision head."""

    def __init__(self, bottleneck_dim, skip2_ch, skip1_ch, dropout=0):
        super().__init__()
        self.block1 = DecoderBlockOptimized(bottleneck_dim, skip2_ch, 32)
        self.ds_head1 = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(32, 1, 1))
        self.block2 = DecoderBlockOptimized(32, skip1_ch, 16)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(16, 1, 1))

        nn.init.constant_(self.head[-1].bias, -4.5)
        nn.init.constant_(self.ds_head1[-1].bias, -4.5)

    def forward(self, x, skip2, skip1):
        d1 = self.block1(x, skip2)
        d2 = self.block2(d1, skip1)

        final_out = self.head(d2)
        aux_out = self.ds_head1(d1)
        aux_out = F.interpolate(aux_out, size=final_out.shape[2], mode="linear", align_corners=False)
        return final_out, aux_out


class CDSMNet(nn.Module):
    """Continuous Dynamic State-Space Modeling Network."""

    def __init__(
        self,
        in_channels=3,
        unet_base_filters=32,
        cfc_hidden=48,
        mamba_dim=96,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        dropout=0,
        first_kernel_size=11,
    ):
        super().__init__()
        pad = (first_kernel_size - 1) // 2

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, unet_base_filters, kernel_size=first_kernel_size, padding=pad, bias=False),
            nn.InstanceNorm1d(unet_base_filters, affine=True),
            nn.GELU(),
            ResBlockOptimized(unet_base_filters, unet_base_filters),
        )
        nn.init.kaiming_normal_(self.enc1[0].weight, mode="fan_out", nonlinearity="relu")

        self.pool1 = BlurPool1d(unet_base_filters, stride=4)
        self.enc2 = ResBlockOptimized(unet_base_filters, unet_base_filters * 2)
        self.pool2 = BlurPool1d(unet_base_filters * 2, stride=4)
        self.enc3 = ResBlockOptimized(unet_base_filters * 2, cfc_hidden)

        self.cfc_fwd = CfC(input_size=cfc_hidden, units=cfc_hidden, batch_first=True)
        self.cfc_bwd = CfC(input_size=cfc_hidden, units=cfc_hidden, batch_first=True)
        cfc_out_dim = cfc_hidden * 2
        self.pos_encoder = PositionalEncoding(cfc_out_dim, max_len=2000)

        self.proj_cfc_mamba = nn.Sequential(
            nn.Linear(cfc_out_dim, mamba_dim),
            nn.LayerNorm(mamba_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        nn.init.trunc_normal_(self.proj_cfc_mamba[0].weight, std=0.02)
        nn.init.constant_(self.proj_cfc_mamba[0].bias, 0)

        self.mamba_mixer = BiMambaAdapter(
            d_model=mamba_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

        self.decoder_det = DecoderBranchDeepSupOptimized(
            mamba_dim,
            unet_base_filters * 2,
            unet_base_filters,
            dropout,
        )
        self.decoder_p = DecoderBranchDeepSupOptimized(
            mamba_dim,
            unet_base_filters * 2,
            unet_base_filters,
            dropout,
        )
        self.decoder_s = DecoderBranchDeepSupOptimized(
            mamba_dim,
            unet_base_filters * 2,
            unet_base_filters,
            dropout,
        )

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        z = self.enc3(p2)

        z_rnn_in = z.permute(0, 2, 1)
        out_fwd, _ = self.cfc_fwd(z_rnn_in)
        z_rnn_bwd = torch.flip(z_rnn_in, [1])
        out_bwd, _ = self.cfc_bwd(z_rnn_bwd)
        out_bwd = torch.flip(out_bwd, [1])
        z_bi = torch.cat([out_fwd, out_bwd], dim=2)

        # Positional encoding is applied in [batch, channels, length] layout.
        z_bi = z_bi.permute(0, 2, 1)
        z_bi = self.pos_encoder(z_bi)
        z_bi = z_bi.permute(0, 2, 1)

        z_proj = self.proj_cfc_mamba(z_bi)
        z_mamba = self.mamba_mixer(z_proj)
        dec_in = z_mamba.permute(0, 2, 1)

        det_final, det_aux = self.decoder_det(dec_in, s2, s1)
        p_final, p_aux = self.decoder_p(dec_in, s2, s1)
        s_final, s_aux = self.decoder_s(dec_in, s2, s1)

        if self.training:
            return {
                "det": det_final,
                "det_aux": det_aux,
                "p": p_final,
                "p_aux": p_aux,
                "s": s_final,
                "s_aux": s_aux,
            }

        return torch.cat([det_final, p_final, s_final], dim=1)
