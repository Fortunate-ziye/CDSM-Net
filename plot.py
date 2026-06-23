import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import wandb
import copy
from scipy import signal
from matplotlib.lines import Line2D
import io
from PIL import Image
from scipy.signal import find_peaks


# ==========================================
# ==========================================
def smooth_curve(x, window_len=11):
    if window_len < 3: return x
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2):-int(window_len / 2)]


# ==========================================
# ==========================================
def plot_single_panel(waveform, target, pred_prob, title_suffix="", sample_rate=100,
                      plot_mode='time', clean_mode=False):


    COLOR_MANUAL_P = '#FFD700'
    COLOR_MANUAL_S = '#FF4500'
    COLOR_PRED_P = 'cyan'
    COLOR_PRED_S = 'magenta'
    COLOR_PROB_DET = 'green'
    COLOR_PROB_P = 'blue'
    COLOR_PROB_S = 'red'

    idx_det, idx_p, idx_s = 0, 1, 2
    length = waveform.shape[1]

    smooth_det = smooth_curve(pred_prob[idx_det, :], window_len=15)
    smooth_p = smooth_curve(pred_prob[idx_p, :], window_len=15)
    smooth_s = smooth_curve(pred_prob[idx_s, :], window_len=15)

    picks = {'man_p': [], 'man_s': [], 'pred_p': [], 'pred_s': []}

    man_p_peaks, _ = find_peaks(target[idx_p, :], height=0.3, distance=50)
    man_s_peaks, _ = find_peaks(target[idx_s, :], height=0.3, distance=50)
    picks['man_p'].extend(man_p_peaks)
    picks['man_s'].extend(man_s_peaks)

    pred_p_peaks, _ = find_peaks(smooth_p, height=0.3, distance=50)
    pred_s_peaks, _ = find_peaks(smooth_s, height=0.3, distance=50)
    picks['pred_p'].extend(pred_p_peaks)
    picks['pred_s'].extend(pred_s_peaks)

    if clean_mode:
        fig = plt.figure(figsize=(10, 6), constrained_layout=False)
        widths = [6, 1]
        heights = [1, 1, 1, 1.2]  # E, N, Z, Prob
        spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths, height_ratios=heights, hspace=0.1)
    else:
        if plot_mode == 'time_frequency':
            fig = plt.figure(figsize=(10, 12), constrained_layout=False)
            widths = [6, 1]
            heights = [1, 1, 1, 1, 1, 1, 1.8]
            spec = fig.add_gridspec(ncols=2, nrows=7, width_ratios=widths, height_ratios=heights, hspace=0.15)
        else:
            fig = plt.figure(figsize=(10, 8), constrained_layout=False)
            widths = [6, 1]
            heights = [1, 1, 1, 1.5]
            spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths, height_ratios=heights, hspace=0.1)

    labels = ['E', 'N', 'Z']

    for i in range(3):
        if clean_mode:
            row_idx = i
        else:
            row_idx = i * 2 if plot_mode == 'time_frequency' else i

        ax_wave = fig.add_subplot(spec[row_idx, 0])
        ax_wave.plot(waveform[i], 'k', linewidth=0.8)
        ax_wave.set_xlim(0, length)
        ax_wave.set_ylabel('Amp')
        if i == 0: ax_wave.set_title(title_suffix, loc='left', fontweight='bold')

        is_active_channel = np.max(np.abs(waveform[i])) > 1e-6
        limit = max(abs(waveform[i].min()), abs(waveform[i].max())) * 1.1
        if limit == 0: limit = 1
        ax_wave.set_ylim(-limit, limit)

        if is_active_channel:
            ymin, ymax = -limit, limit
            for p in picks['man_p']:
                ax_wave.vlines(p, ymin, ymax, color=COLOR_MANUAL_P, alpha=0.9, lw=2, label='Manual P')
            for s in picks['man_s']:
                ax_wave.vlines(s, ymin, ymax, color=COLOR_MANUAL_S, alpha=0.9, lw=2, label='Manual S')

        ax_wave.set_xticks([])

        ax_leg = fig.add_subplot(spec[row_idx, 1])
        ax_leg.axis('off')
        if clean_mode:
            ax_leg.text(0, 0.5, labels[i], fontweight='bold', va='center')
        else:
            custom_lines = [
                Line2D([0], [0], color='k', lw=0),
                Line2D([0], [0], color=COLOR_MANUAL_P, lw=2),
                Line2D([0], [0], color=COLOR_MANUAL_S, lw=2)
            ]
            ax_leg.legend(custom_lines, [labels[i], 'Manual P', 'Manual S'],
                          loc='center left', fontsize=8, frameon=False)

        if not clean_mode and plot_mode == 'time_frequency':
            ax_stft = fig.add_subplot(spec[row_idx + 1, 0])
            f, t, Pxx = signal.stft(waveform[i], fs=sample_rate, nperseg=80)

            Pxx_log = 10 * np.log10(np.abs(Pxx) + 1e-6)
            t_samples = t * sample_rate

            ax_stft.pcolormesh(t_samples, f, Pxx_log, alpha=None, cmap='inferno', shading='auto')

            ax_stft.set_xlim(0, length)
            ax_stft.set_ylim(0, 40)
            ax_stft.set_ylabel('Hz')
            ax_stft.set_xticks([])
            if i == 0:
                ax_stft.text(0.02, 0.8, 'STFT (dB)', transform=ax_stft.transAxes, color='white', fontweight='bold',
                             fontsize=10)

    prob_row = 3 if clean_mode else (6 if plot_mode == 'time_frequency' else 3)
    ax_prob = fig.add_subplot(spec[prob_row, 0])
    x_axis = np.arange(length)

    ax_prob.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    if not clean_mode:
        ax_prob.text(0, 0.52, 'Threshold 0.5', fontsize=8, color='gray', fontweight='bold')

    ax_prob.fill_between(x_axis, smooth_det, color=COLOR_PROB_DET, alpha=0.1)
    ax_prob.plot(x_axis, smooth_det, '-', color=COLOR_PROB_DET, alpha=0.8, lw=1.5, label='Earthquake')

    ax_prob.plot(x_axis, smooth_p, '--', color=COLOR_PROB_P, alpha=0.8, lw=1.5, label='P_arrival')
    ax_prob.plot(x_axis, smooth_s, '--', color=COLOR_PROB_S, alpha=0.8, lw=1.5, label='S_arrival')

    for p in picks['pred_p']:
        ax_prob.vlines(p, 0, 1, color=COLOR_PRED_P, lw=2, linestyle='-', alpha=0.8)
    for s in picks['pred_s']:
        ax_prob.vlines(s, 0, 1, color=COLOR_PRED_S, lw=2, linestyle='-', alpha=0.8)

    ax_prob.set_xlim(0, length)
    ax_prob.set_ylim(-0.1, 1.1)
    ax_prob.set_xlabel('Sample', fontweight='bold')
    ax_prob.set_ylabel('Probability', fontweight='bold')
    ax_prob.grid(True, axis='y', color='lightgray', linestyle='-', linewidth=0.5)

    ax_prob_leg = fig.add_subplot(spec[prob_row, 1])
    ax_prob_leg.axis('off')

    if clean_mode:
        ax_prob_leg.text(0, 0.8, 'Det (Green)', color=COLOR_PROB_DET, fontsize=8)
        ax_prob_leg.text(0, 0.5, 'P (Blue)', color=COLOR_PROB_P, fontsize=8)
        ax_prob_leg.text(0, 0.2, 'S (Red)', color=COLOR_PROB_S, fontsize=8)
    else:
        prob_lines = [
            Line2D([0], [0], color=COLOR_PRED_P, lw=2),
            Line2D([0], [0], color=COLOR_PRED_S, lw=2),
            Line2D([0], [0], linestyle='--', color=COLOR_PROB_P, lw=2),
            Line2D([0], [0], linestyle='--', color=COLOR_PROB_S, lw=2),
            Line2D([0], [0], linestyle='--', color=COLOR_PROB_DET, lw=2)
        ]
        ax_prob_leg.legend(prob_lines,
                           ['Pred P', 'Pred S', 'P Prob', 'S Prob', 'Det Prob'],
                           loc='center left', fontsize=8, frameon=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return wandb.Image(img)


# ==========================================
# ==========================================
def generate_stress_batch(clean_waveforms, clean_targets):
    noisy_wav = copy.deepcopy(clean_waveforms).cpu().numpy()
    noisy_tar = copy.deepcopy(clean_targets).cpu().numpy()
    batch_size = noisy_wav.shape[0]

    eq_indices = [i for i in range(batch_size) if np.max(noisy_tar[i, 1, :]) > 0.5]
    if len(eq_indices) < 2: eq_indices = list(range(batch_size)) * 2

    titles = []
    final_wavs = []
    final_tars = []

    # (a-d) Normal Samples
    for i in range(4):
        idx = eq_indices[i % len(eq_indices)]
        final_wavs.append(noisy_wav[idx])
        final_tars.append(noisy_tar[idx])
        titles.append(f"({chr(97 + i)}) Normal Sample {i + 1}")

    # (e) Multi-event Overlap
    idx1 = eq_indices[0]
    idx2 = eq_indices[1] if len(eq_indices) > 1 else eq_indices[0]
    w_e = noisy_wav[idx1].copy()
    t_e = noisy_tar[idx1].copy()

    p2_loc = np.argmax(noisy_tar[idx2, 1, :])
    shift = 3500 - p2_loc
    w2_shifted = np.roll(noisy_wav[idx2], shift, axis=1)
    t2_shifted = np.roll(noisy_tar[idx2], shift, axis=1)

    if shift > 0:
        w2_shifted[:, :shift] = 0
        t2_shifted[:, :shift] = 0
    elif shift < 0:
        w2_shifted[:, shift:] = 0
        t2_shifted[:, shift:] = 0

    amp1 = np.max(np.abs(w_e)) + 1e-6
    amp2 = np.max(np.abs(w2_shifted)) + 1e-6

    scale_factor = (amp1 / amp2) * 0.8
    w_e = w_e + w2_shifted * scale_factor

    t_e = np.maximum(t_e, t2_shifted)

    final_wavs.append(w_e)
    final_tars.append(t_e)
    titles.append("(e) Multi-event Overlap (Matched)")

    # (f) Two Channels Noise
    idx = eq_indices[2 % len(eq_indices)]
    w_f = noisy_wav[idx].copy()
    ref_std = np.std(w_f[2, :]) if np.std(w_f[2, :]) > 0 else 1.0
    w_f[0, :] = np.random.normal(0, ref_std * 1.5, w_f.shape[1])
    w_f[1, :] = np.random.normal(0, ref_std * 1.5, w_f.shape[1])
    final_wavs.append(w_f)
    final_tars.append(noisy_tar[idx])
    titles.append("(f) Two-Ch Gaussian Noise")

    # (g) Broken/Noisy Channels
    idx = eq_indices[3 % len(eq_indices)]
    w_g = noisy_wav[idx].copy()
    w_g[0, :] = 0
    noise_std_g = np.std(w_g[2, :]) if np.std(w_g[2, :]) > 0 else 1.0
    w_g[1, :] = np.random.normal(0, noise_std_g * 2.0, w_g.shape[1])
    final_wavs.append(w_g)
    final_tars.append(noisy_tar[idx])
    titles.append("(g) Broken/Noisy Channels")

    # (h) Single Component
    idx = eq_indices[4 % len(eq_indices)]
    w_h = noisy_wav[idx].copy()
    w_h[0, :] = 0
    w_h[1, :] = 0
    final_wavs.append(w_h)
    final_tars.append(noisy_tar[idx])
    titles.append("(h) Single Component (Z)")

    return (torch.tensor(np.array(final_wavs)),
            torch.tensor(np.array(final_tars)),
            titles)


# ==========================================
# ==========================================
def log_fig4_to_wandb(model, waveforms, targets, device, freq=100, plot_mode='time_frequency'):
    model.eval()
    with torch.no_grad():
        stress_inputs, stress_targets, titles = generate_stress_batch(waveforms, targets)
        stress_inputs = stress_inputs.to(device)

        outputs = model(stress_inputs)
        probs = torch.sigmoid(outputs)

        inputs_np = stress_inputs.cpu().numpy()
        targets_np = stress_targets.cpu().numpy()
        probs_np = probs.cpu().numpy()

        wandb_images = []
        for i in range(len(titles)):
            img = plot_single_panel(
                waveform=inputs_np[i],
                target=targets_np[i],
                pred_prob=probs_np[i],
                title_suffix=titles[i],
                sample_rate=freq,
                plot_mode=plot_mode,
                clean_mode=False
            )
            wandb_images.append(img)

        wandb.log({f"Fig4_Performance ({plot_mode})": wandb_images}, commit=False)


# ==========================================
# ==========================================
def plot_error_distribution(p_res, s_res, metrics_dict, bins=50, range_limit=(-1.0, 1.0)):
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        def _draw_hist(ax, residuals, phase_name, color, stats_prefix):
            data = np.array(residuals)
            plot_data = data[(data >= range_limit[0]) & (data <= range_limit[1])]

            ax.hist(plot_data, bins=bins, range=range_limit, color=color, edgecolor='black', alpha=0.7)
            ax.set_title(f"{phase_name} picks", fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Residuals (s)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)

            prefix = stats_prefix.lower()
            stats_text = (
                f"F1 = {metrics_dict[f'{prefix}_f1']:.2f}\n"
                f"MAE = {metrics_dict[f'{prefix}_mae']:.3f}\n"
                f"Mean = {metrics_dict[f'{prefix}_mean_error']:.3f} s\n"
                f"Std = {metrics_dict[f'{prefix}_std_dev']:.3f} s"
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')

        _draw_hist(axes[0], p_res, "P", color='mediumslateblue', stats_prefix='p')
        _draw_hist(axes[1], s_res, "S", color='indianred', stats_prefix='s')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return wandb.Image(img, caption="Error Distribution Analysis")


# ==========================================
# ==========================================
def plot_event_level_confusion_matrix(y_true_labels, y_pred_labels):
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[1, 0])
    labels = ['Earthquake', 'Noise']

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 18, "weight": "bold"}, cbar=False, ax=ax,
                linewidths=1, linecolor='black')

    ax.set_title('Detection Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return wandb.Image(img, caption="Event Detection CM")
