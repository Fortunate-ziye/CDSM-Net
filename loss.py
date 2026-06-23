import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for sparse phase labels."""

    def __init__(self, alpha=0.75, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LogCoshDiceLoss(nn.Module):
    """Smooth Dice-style loss used to preserve phase-label shape."""

    def __init__(self, smooth=10.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        return torch.log(torch.cosh(dice_loss))


class Loss(nn.Module):
    """Combined loss for detection, P picking, and S picking."""

    def __init__(
        self,
        task_weights=None,
        aux_weight=0.4,
        hard_mining_weight=8.0,
        miss_thresh=0.2,
        reg_weight=0.15,
        device=None,
    ):
        super().__init__()
        if task_weights is None:
            task_weights = [0.05, 0.40, 0.55]

        self.task_weights = task_weights
        self.aux_weight = aux_weight
        self.hm_weight = hard_mining_weight
        self.miss_thresh = miss_thresh
        self.reg_weight = reg_weight

        init_device = device
        if init_device == "cuda" and not torch.cuda.is_available():
            init_device = "cpu"
        self.det_pos_weight = torch.tensor([1.0], device=init_device)

        self.bce_det = nn.BCEWithLogitsLoss(pos_weight=self.det_pos_weight, reduction="none")
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0, reduction="none")
        self.dice_loss = LogCoshDiceLoss(smooth=10.0)

    def calculate_single_loss(self, logits, targets):
        raw_loss_det = self.bce_det(logits[:, 0:1, :], targets[:, 0:1, :])
        raw_loss_p = self.focal_loss(logits[:, 1:2, :], targets[:, 1:2, :])
        raw_loss_s = self.focal_loss(logits[:, 2:3, :], targets[:, 2:3, :])

        # Increase the penalty on high-confidence missed phase labels.
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            mask_miss_p = (targets[:, 1:2, :] > 0.5) & (
                probs[:, 1:2, :] < self.miss_thresh
            )
            mask_miss_s = (targets[:, 2:3, :] > 0.5) & (
                probs[:, 2:3, :] < self.miss_thresh
            )
            w_p = torch.ones_like(raw_loss_p)
            w_s = torch.ones_like(raw_loss_s)
            w_p[mask_miss_p] = self.hm_weight
            w_s[mask_miss_s] = self.hm_weight

        dice_p = self.dice_loss(logits[:, 1:2, :], targets[:, 1:2, :])
        dice_s = self.dice_loss(logits[:, 2:3, :], targets[:, 2:3, :])

        def soft_argmax_loss(pred_logits, true_targets):
            probs = torch.sigmoid(pred_logits)
            batch_size, _, trace_len = probs.shape
            grid = torch.arange(trace_len, device=probs.device).float().view(1, 1, trace_len)

            pred_center = (probs * grid).sum(dim=-1) / (probs.sum(dim=-1) + 1e-6)
            true_center = (true_targets * grid).sum(dim=-1) / (
                true_targets.sum(dim=-1) + 1e-6
            )
            has_event = true_targets.view(batch_size, -1).max(dim=-1).values > 0.5

            if has_event.sum() > 0:
                return F.l1_loss(
                    pred_center[has_event] / trace_len,
                    true_center[has_event] / trace_len,
                )
            return torch.tensor(0.0, device=probs.device)

        reg_p = soft_argmax_loss(logits[:, 1:2, :], targets[:, 1:2, :])
        reg_s = soft_argmax_loss(logits[:, 2:3, :], targets[:, 2:3, :])

        final_loss_det = raw_loss_det.mean()
        final_loss_p = (raw_loss_p * w_p).mean() + 0.5 * dice_p + self.reg_weight * reg_p
        final_loss_s = (raw_loss_s * w_s).mean() + 0.5 * dice_s + self.reg_weight * reg_s

        return (
            self.task_weights[0] * final_loss_det
            + self.task_weights[1] * final_loss_p
            + self.task_weights[2] * final_loss_s
        )

    def forward(self, outputs, targets):
        if self.det_pos_weight.device != targets.device:
            self.det_pos_weight = self.det_pos_weight.to(targets.device)
            self.bce_det.pos_weight = self.det_pos_weight
            self.focal_loss.to(targets.device)
            self.dice_loss.to(targets.device)

        if isinstance(outputs, dict):
            loss_main = self.calculate_single_loss(
                torch.cat([outputs["det"], outputs["p"], outputs["s"]], dim=1),
                targets,
            )
            loss_aux = self.calculate_single_loss(
                torch.cat([outputs["det_aux"], outputs["p_aux"], outputs["s_aux"]], dim=1),
                targets,
            )
            return loss_main + self.aux_weight * loss_aux

        return self.calculate_single_loss(outputs, targets)
