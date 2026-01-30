import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from train_time_resnet import TimeCondResNet  # noqa: E402


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ================= 配置 =================
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(REPO_ROOT, "checkpoints_time", "time_resnet_final.pth"))
_DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "processed_data", "26ms", "340.npy")
if not os.path.exists(_DEFAULT_DATA_PATH):
    _DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "processed_data", "340.npy")
DATA_PATH = os.getenv("DATA_PATH", _DEFAULT_DATA_PATH)

SAVE_GIF = os.getenv("SAVE_GIF", os.path.join(REPO_ROOT, "results_time", "prediction_9s.gif"))
SAVE_ERROR_CURVE = os.getenv("SAVE_ERROR_CURVE", os.path.join(REPO_ROOT, "results_time", "error_curve_9s.png"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM_FACTOR = float(os.getenv("NORM_FACTOR", "30.0"))
DT = float(os.getenv("DT", "0.002"))

PRED_STEPS = int(os.getenv("PRED_STEPS", "4500"))
GIF_STRIDE = int(os.getenv("GIF_STRIDE", "25"))
BATCH_TIMES = int(os.getenv("BATCH_TIMES", "128"))

TIME_NORM_MODE = os.getenv("TIME_NORM_MODE", "index").lower()  # index | seconds

# 推理端硬约束（可选）
CLAMP_ENABLE = int(os.getenv("CLAMP_ENABLE", "1"))
CLAMP_MIN = float(os.getenv("CLAMP_MIN", "-0.5"))
CLAMP_MAX = float(os.getenv("CLAMP_MAX", "1.5"))

MASK_ENABLE = int(os.getenv("MASK_ENABLE", "1"))
MASK_MODE = os.getenv("MASK_MODE", "u").lower()  # u | speed
MASK_U_THRESHOLD = float(os.getenv("MASK_U_THRESHOLD", "0.2"))
MASK_SPEED_THRESHOLD = float(os.getenv("MASK_SPEED_THRESHOLD", "0.2"))
MASK_LEFT_COLS = int(os.getenv("MASK_LEFT_COLS", "24"))
MASK_LOCK = os.getenv("MASK_LOCK", "initial").lower()  # initial | zero

BC_ENABLE = int(os.getenv("BC_ENABLE", "1"))
BC_PAD_RIGHT = int(os.getenv("BC_PAD_RIGHT", "4"))
BC_PAD_TOP = int(os.getenv("BC_PAD_TOP", "4"))
BC_PAD_BOTTOM = int(os.getenv("BC_PAD_BOTTOM", "4"))


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def apply_clamp(x):
    if not CLAMP_ENABLE:
        return x
    return torch.clamp(x, min=CLAMP_MIN, max=CLAMP_MAX)


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


def make_time_norm(step_idx, T_total):
    if TIME_NORM_MODE == "index":
        return float(step_idx) / float(T_total)
    if TIME_NORM_MODE == "seconds":
        t_seconds = float(step_idx) * DT
        t_max = float(T_total) * DT
        return t_seconds / max(t_max, 1e-12)
    raise ValueError(f"Unknown TIME_NORM_MODE={TIME_NORM_MODE}")


if __name__ == "__main__":
    ensure_parent(SAVE_GIF)
    ensure_parent(SAVE_ERROR_CURVE)

    print(f"🚀 Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

    print(f"🔄 Loading model: {MODEL_PATH}")
    model = TimeCondResNet().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print(f"📂 Loading data: {DATA_PATH}")
    data = np.load(DATA_PATH, mmap_mode="r")
    test_seq = data[: PRED_STEPS + 1]
    T_total = PRED_STEPS
    print(f"✅ test_seq: {test_seq.shape}")

    x0 = test_seq[0].astype(np.float32) / NORM_FACTOR  # (H,W,3)
    x0_tensor = torch.from_numpy(x0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    initial_tensor = x0_tensor.clone()
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
    errors = np.zeros((PRED_STEPS + 1,), dtype=np.float64)

    print(f"🧠 One-shot inference for {PRED_STEPS} steps (batched={BATCH_TIMES}) ...")
    with torch.no_grad():
        step = 1
        while step <= PRED_STEPS:
            end = min(PRED_STEPS, step + BATCH_TIMES - 1)
            batch_steps = np.arange(step, end + 1, dtype=np.int64)
            t_norm = np.array([make_time_norm(int(s), T_total) for s in batch_steps], dtype=np.float32)

            b = len(batch_steps)
            x0_rep = x0_tensor.repeat(b, 1, 1, 1)  # (B,3,H,W)
            time_map = torch.from_numpy(t_norm).to(DEVICE).view(b, 1, 1, 1)
            time_map = time_map.expand(b, 1, x0_rep.shape[2], x0_rep.shape[3])

            inp = torch.cat([x0_rep, time_map], dim=1)  # (B,4,H,W)
            pred = model(inp)  # (B,3,H,W) normalized
            pred = apply_clamp(pred)
            pred = apply_hard_constraints(pred, initial_tensor.expand_as(pred[:1]), hull_mask)
            pred = apply_clamp(pred)

            pred_np = pred.permute(0, 2, 3, 1).cpu().numpy() * NORM_FACTOR  # (B,H,W,3)
            truth_np = test_seq[batch_steps]  # (B,H,W,3)

            diff = pred_np - truth_np
            diff_norm = np.linalg.norm(diff.reshape(b, -1), axis=1)
            truth_norm = np.linalg.norm(truth_np.reshape(b, -1), axis=1)
            errors[batch_steps] = diff_norm / truth_norm

            for i, s in enumerate(batch_steps):
                if int(s) in gif_step_set:
                    gif_truth_u[int(s)] = truth_np[i, :, :, 0]
                    gif_pred_u[int(s)] = pred_np[i, :, :, 0]

            step = end + 1

    print(f"✅ Done. mean={errors.mean():.6f}, max={errors.max():.6f}, final={errors[-1]:.6f}")

    print("🎨 Rendering GIF ...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin, vmax = test_seq[:, :, :, 0].min(), test_seq[:, :, :, 0].max()

    def update(frame_i):
        for ax in axes:
            ax.clear()
        s = gif_steps[frame_i]
        real_u = gif_truth_u[s]
        pred_u = gif_pred_u[s]
        axes[0].imshow(real_u, cmap="jet", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"GT (step={s}, t={s*DT:.3f}s)")
        axes[0].axis("off")
        axes[1].imshow(pred_u, cmap="jet", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"ResNet G(x0,t) (step={s}, t={s*DT:.3f}s)")
        axes[1].axis("off")

    ani = animation.FuncAnimation(fig, update, frames=len(gif_steps), interval=50)
    ani.save(SAVE_GIF, writer="pillow", fps=20)
    plt.close(fig)
    print(f"🎉 Saved: {SAVE_GIF}")

    print("📈 Plotting error curve ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.arange(len(errors)) * DT
    ax.plot(time_axis, errors, linewidth=2, color="#E74C3C", label="L2 Relative Error")
    ax.axhline(y=float(errors.mean()), color="#3498DB", linestyle="--", linewidth=1.5, label=f"Mean={errors.mean():.6f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Time-Conditioned ResNet Error (One-shot, 9s)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(SAVE_ERROR_CURVE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"🎉 Saved: {SAVE_ERROR_CURVE}")
