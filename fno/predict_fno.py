import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from train_fno import FNO2d  # noqa: E402


# ================= 配置 =================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(REPO_ROOT, "checkpoints_fno", "fno_final.pth"))
_DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "processed_data", "26ms", "340.npy")
if not os.path.exists(_DEFAULT_DATA_PATH):
    _DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "processed_data", "340.npy")
DATA_PATH = os.getenv("DATA_PATH", _DEFAULT_DATA_PATH)

SAVE_GIF = os.getenv("SAVE_GIF", os.path.join(REPO_ROOT, "results_fno", "prediction_9s.gif"))
SAVE_ERROR_CURVE = os.getenv("SAVE_ERROR_CURVE", os.path.join(REPO_ROOT, "results_fno", "error_curve_9s.png"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRED_STEPS = int(os.getenv("PRED_STEPS", "4500"))
GIF_STRIDE = int(os.getenv("GIF_STRIDE", "25"))

NORM_FACTOR = float(os.getenv("NORM_FACTOR", "30.0"))

MODES = int(os.getenv("MODES", "12"))
WIDTH = int(os.getenv("WIDTH", "32"))
PADDING = int(os.getenv("PADDING", "20"))

# 物理范围（用于防止爆掉；默认 hard clamp）
CLAMP_ENABLE = int(os.getenv("CLAMP_ENABLE", "1"))
CLAMP_MIN = float(os.getenv("CLAMP_MIN", "-0.5"))
CLAMP_MAX = float(os.getenv("CLAMP_MAX", "1.5"))

# 数值阻尼（可选）：next = (1-d)*pred + d*prev
DAMPING = float(os.getenv("DAMPING", "0.0"))

# 船体/死水区 Mask（从 t=0 自动估计并锁死）
MASK_ENABLE = int(os.getenv("MASK_ENABLE", "1"))
MASK_MODE = os.getenv("MASK_MODE", "u").lower()  # u | speed
MASK_U_THRESHOLD = float(os.getenv("MASK_U_THRESHOLD", "0.2"))
MASK_SPEED_THRESHOLD = float(os.getenv("MASK_SPEED_THRESHOLD", "0.2"))
MASK_LEFT_COLS = int(os.getenv("MASK_LEFT_COLS", "24"))
MASK_LOCK = os.getenv("MASK_LOCK", "initial").lower()  # initial | zero

# 开放边界 Dirichlet：右/上/下边缘每步重置到 t=0 背景
BC_ENABLE = int(os.getenv("BC_ENABLE", "1"))
BC_PAD_RIGHT = int(os.getenv("BC_PAD_RIGHT", "4"))
BC_PAD_TOP = int(os.getenv("BC_PAD_TOP", "4"))
BC_PAD_BOTTOM = int(os.getenv("BC_PAD_BOTTOM", "4"))


def compute_hull_mask(initial_tensor):
    if not MASK_ENABLE:
        return None
    _, _, h, w = initial_tensor.shape
    left_cols = max(0, min(int(MASK_LEFT_COLS), w))
    if left_cols <= 0:
        return None

    if MASK_MODE == "u":
        u0 = initial_tensor[:, 0:1]
        low = u0 < MASK_U_THRESHOLD
    elif MASK_MODE == "speed":
        speed0 = torch.linalg.vector_norm(initial_tensor, dim=1, keepdim=True)
        low = speed0 < MASK_SPEED_THRESHOLD
    else:
        raise ValueError(f"Unknown MASK_MODE={MASK_MODE}")

    region = torch.zeros((1, 1, h, w), device=initial_tensor.device, dtype=torch.bool)
    region[:, :, :, :left_cols] = True
    return low & region


def apply_hard_constraints(x, initial_tensor, hull_mask):
    if BC_ENABLE:
        if BC_PAD_RIGHT > 0:
            x[:, :, :, -BC_PAD_RIGHT:] = initial_tensor[:, :, :, -BC_PAD_RIGHT:]
        if BC_PAD_TOP > 0:
            x[:, :, :BC_PAD_TOP, :] = initial_tensor[:, :, :BC_PAD_TOP, :]
        if BC_PAD_BOTTOM > 0:
            x[:, :, -BC_PAD_BOTTOM:, :] = initial_tensor[:, :, -BC_PAD_BOTTOM:, :]

    if hull_mask is not None:
        if MASK_LOCK == "zero":
            lock_val = torch.zeros_like(x)
        elif MASK_LOCK == "initial":
            lock_val = initial_tensor
        else:
            raise ValueError(f"Unknown MASK_LOCK={MASK_LOCK}")
        x = torch.where(hull_mask.expand_as(x), lock_val, x)

    return x


def apply_clamp(x):
    if not CLAMP_ENABLE:
        return x
    return torch.clamp(x, min=CLAMP_MIN, max=CLAMP_MAX)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


if __name__ == "__main__":
    print(f"🚀 Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

    ensure_parent_dir(SAVE_GIF)
    ensure_parent_dir(SAVE_ERROR_CURVE)

    print(f"🔄 Loading FNO: {MODEL_PATH}")
    model = FNO2d(modes=MODES, width=WIDTH, padding=PADDING).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print(f"📂 Loading data: {DATA_PATH}")
    data = np.load(DATA_PATH, mmap_mode="r")
    test_seq = data[: PRED_STEPS + 1]
    print(f"✅ test_seq: {test_seq.shape}")

    current_frame = test_seq[0]
    current_tensor = (
        torch.from_numpy(current_frame / NORM_FACTOR)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )
    initial_tensor = current_tensor.clone()
    hull_mask = compute_hull_mask(initial_tensor)
    if hull_mask is not None:
        print(f"🧱 Hull mask ratio={hull_mask.float().mean().item():.4f} (left_cols={MASK_LEFT_COLS})")
    if BC_ENABLE:
        print(f"🧷 Dirichlet BC right/top/bottom={BC_PAD_RIGHT}/{BC_PAD_TOP}/{BC_PAD_BOTTOM}px")

    gif_steps = list(range(0, PRED_STEPS + 1, max(GIF_STRIDE, 1)))
    if gif_steps[-1] != PRED_STEPS:
        gif_steps.append(PRED_STEPS)
    gif_step_set = set(gif_steps)

    gif_truth_u = {0: test_seq[0, :, :, 0]}
    gif_pred_u = {0: test_seq[0, :, :, 0]}
    errors = [0.0]

    print(f"🔮 Rolling prediction: steps={PRED_STEPS}")
    with torch.no_grad():
        for t in range(PRED_STEPS):
            step_idx = t + 1
            next_tensor = model(current_tensor)
            next_tensor = apply_clamp(next_tensor)
            next_tensor = apply_hard_constraints(next_tensor, initial_tensor, hull_mask)
            if DAMPING > 0:
                next_tensor = (1.0 - DAMPING) * next_tensor + DAMPING * current_tensor
            next_tensor = apply_clamp(next_tensor)

            pred_np = next_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * NORM_FACTOR
            truth_np = test_seq[step_idx]
            diff_norm = np.linalg.norm(pred_np - truth_np)
            truth_norm = np.linalg.norm(truth_np)
            errors.append(diff_norm / truth_norm)

            if step_idx in gif_step_set:
                gif_truth_u[step_idx] = truth_np[:, :, 0]
                gif_pred_u[step_idx] = pred_np[:, :, 0]

            current_tensor = next_tensor

            if step_idx % 500 == 0:
                print(
                    f"  Step {step_idx}/{PRED_STEPS}: "
                    f"max={next_tensor.max().item():.3f}, min={next_tensor.min().item():.3f}"
                )

    errors = np.array(errors)
    print("✅ Done")
    print(f"   mean={errors.mean():.6f}, max={errors.max():.6f}, final={errors[-1]:.6f}")

    print("🎨 Rendering GIF ...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin, vmax = test_seq[:, :, :, 0].min(), test_seq[:, :, :, 0].max()

    def update(frame_i):
        for ax in axes:
            ax.clear()
        step = gif_steps[frame_i]
        real_u = gif_truth_u[step]
        pred_u = gif_pred_u[step]
        axes[0].imshow(real_u, cmap="jet", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"GT (step={step}, t={step*0.002:.3f}s)")
        axes[0].axis("off")
        axes[1].imshow(pred_u, cmap="jet", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"FNO (step={step}, t={step*0.002:.3f}s)")
        axes[1].axis("off")

    ani = animation.FuncAnimation(fig, update, frames=len(gif_steps), interval=50)
    ani.save(SAVE_GIF, writer="pillow", fps=20)
    plt.close(fig)
    print(f"🎉 Saved: {SAVE_GIF}")

    print("📈 Plotting error curve ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.arange(len(errors)) * 0.002
    ax.plot(time_axis, errors, linewidth=2, color="#E74C3C", label="L2 Relative Error")
    ax.axhline(y=errors.mean(), color="#3498DB", linestyle="--", linewidth=1.5, label=f"Mean={errors.mean():.6f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("FNO Long-term Prediction Error (9s @ Δt=2ms)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(SAVE_ERROR_CURVE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"🎉 Saved: {SAVE_ERROR_CURVE}")
