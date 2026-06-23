import torch
import torch.nn as nn


class SeismicAugmentor(nn.Module):
    """Online waveform augmentation for seismic detection and phase picking."""

    def __init__(
        self,
        p_stack=0.6,
        p_noise=0.5,
        p_shift=0.99,
        p_gap=0.2,
        p_drop=0.3,
        max_shift=1000,
        max_gap_size=1000,
        snr_min=1.0,
        snr_max=20.0,
    ):
        super().__init__()
        self.p_stack = p_stack
        self.p_noise = p_noise
        self.p_shift = p_shift
        self.p_gap = p_gap
        self.p_drop = p_drop
        self.max_shift = max_shift
        self.max_gap_size = max_gap_size
        self.snr_min = snr_min
        self.snr_max = snr_max

    @staticmethod
    def _normalize(x):
        """Apply per-channel z-score normalization."""
        mean = torch.mean(x, dim=2, keepdim=True)
        std = torch.std(x, dim=2, keepdim=True)
        return (x - mean) / (std + 1e-6)

    def forward(self, waveforms, targets):
        """
        Args:
            waveforms: Tensor with shape [batch, 3, time].
            targets: Tensor with shape [batch, 3, time] for detection, P, and S.
        """
        if not self.training:
            return self._normalize(waveforms), targets

        batch_size, channels, trace_len = waveforms.shape
        device = waveforms.device
        aug_size = batch_size // 2
        if aug_size == 0:
            return self._normalize(waveforms), targets

        x_aug = waveforms[:aug_size].clone()
        y_aug = targets[:aug_size].clone()

        # Random circular shifts move arrivals within the fixed input window.
        if torch.rand(1, device=device) < self.p_shift:
            shifts = torch.randint(-self.max_shift, self.max_shift, (aug_size, 1), device=device)
            arange = torch.arange(trace_len, device=device).unsqueeze(0)
            indices = (arange - shifts) % trace_len
            indices = indices.unsqueeze(1).expand(aug_size, channels, trace_len)

            x_aug = torch.gather(x_aug, 2, indices)
            y_aug = torch.gather(y_aug, 2, indices)

        # Stack a delayed secondary event to simulate overlapping events.
        if torch.rand(1, device=device) < self.p_stack:
            perm_idx = torch.randperm(aug_size, device=device)
            secondary_wave = x_aug[perm_idx].clone()
            secondary_target = y_aug[perm_idx].clone()

            std_curr = torch.std(x_aug, dim=2, keepdim=True) + 1e-6
            noise_floor = torch.randn_like(secondary_wave) * std_curr
            shifts = torch.randint(0, trace_len // 2, (aug_size,), device=device)

            for i in range(aug_size):
                shift = shifts[i].item()
                if shift <= 0:
                    continue

                secondary_wave[i] = torch.cat(
                    [noise_floor[i, :, :shift], secondary_wave[i, :, :-shift]],
                    dim=1,
                )
                zeros_label = torch.zeros((3, shift), device=device)
                secondary_target[i] = torch.cat(
                    [zeros_label, secondary_target[i, :, :-shift]],
                    dim=1,
                )

            std_sec = torch.std(secondary_wave, dim=2, keepdim=True) + 1e-6
            alpha = torch.rand(aug_size, 1, 1, device=device) * 0.4 + 0.1

            x_aug = x_aug + secondary_wave * (std_curr / std_sec) * alpha
            y_aug = torch.max(y_aug, secondary_target)

        # Add Gaussian noise at a randomly sampled target SNR.
        if torch.rand(1, device=device) < self.p_noise:
            signal_power = torch.mean(x_aug ** 2, dim=2, keepdim=True)
            target_snr_db = torch.rand(aug_size, 1, 1, device=device) * (
                self.snr_max - self.snr_min
            ) + self.snr_min
            target_snr_linear = 10 ** (target_snr_db / 10.0)
            noise_power = signal_power / (target_snr_linear + 1e-6)
            x_aug = x_aug + torch.randn_like(x_aug) * torch.sqrt(noise_power)

        # Random gaps force the model to use context around short signal dropouts.
        if torch.rand(1, device=device) < self.p_gap:
            max_gap = min(self.max_gap_size, trace_len - 1)
            if max_gap > 20:
                gap_len = torch.randint(20, max_gap, (aug_size,), device=device)
                gap_start = (torch.rand(aug_size, device=device) * (trace_len - gap_len)).long()
                arange = torch.arange(trace_len, device=device).unsqueeze(0)
                mask_gap = (arange < gap_start.unsqueeze(1)) | (
                    arange >= (gap_start + gap_len).unsqueeze(1)
                )
                x_aug = x_aug * mask_gap.unsqueeze(1).float()

        # Drop waveform channels while always keeping at least one component.
        if torch.rand(1, device=device) < self.p_drop:
            drop_mask = (torch.rand(aug_size, channels, device=device) > 0.5).float()
            keep_idx = torch.randint(0, channels, (aug_size,), device=device)
            drop_mask[torch.arange(aug_size, device=device), keep_idx] = 1.0
            x_aug = x_aug * drop_mask.unsqueeze(2)

        x_aug = self._normalize(x_aug)
        x_clean = self._normalize(waveforms[aug_size:])

        waveforms_out = torch.cat([x_aug, x_clean], dim=0)
        targets_out = torch.cat([y_aug, targets[aug_size:]], dim=0)

        waveforms_out = torch.clamp(waveforms_out, -15.0, 15.0)
        waveforms_out = torch.nan_to_num(waveforms_out, nan=0.0)
        return waveforms_out, targets_out
