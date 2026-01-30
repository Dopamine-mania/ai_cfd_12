import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_time_resnet import TimeCondResNet  # noqa: E402


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "processed_data", "26ms", "340.npy")
if not os.path.exists(_DEFAULT_DATA_PATH):
    _DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "processed_data", "340.npy")
DATA_PATH = os.getenv("DATA_PATH", _DEFAULT_DATA_PATH)
BEFORE_MODEL_PATH = os.getenv("BEFORE_MODEL_PATH", os.path.join(REPO_ROOT, "checkpoints_time", "time_resnet_final.pth"))
AFTER_MODEL_PATH = os.getenv("AFTER_MODEL_PATH", os.path.join(REPO_ROOT, "checkpoints_time_phys", "time_resnet_final.pth"))

SAVE_FIG = os.getenv("SAVE_FIG", os.path.join(REPO_ROOT, "results_time_phys", "triptych_step4500.png"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM_FACTOR = float(os.getenv("NORM_FACTOR", "30.0"))
DT = float(os.getenv("DT", "0.002"))

STEP_IDX = int(os.getenv("STEP_IDX", "4500"))
TIME_NORM_MODE = os.getenv("TIME_NORM_MODE", "index").lower()

COMPONENT = os.getenv("COMPONENT", "u").lower()  # u | v | w

# 推理端硬约束（保持与 one-shot 推理一致）
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


def time_norm(step_idx, T_total):
    if TIME_NORM_MODE == "index":
        return float(step_idx) / float(T_total)
    if TIME_NORM_MODE == "seconds":
        t_seconds = float(step_idx) * DT
        t_max = float(T_total) * DT
        return t_seconds / max(t_max, 1e-12)
    raise ValueError(f"Unknown TIME_NORM_MODE={TIME_NORM_MODE}")


def load_model(path):
    model = TimeCondResNet().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def predict_at_step(model, x0_tensor, t_norm_value, initial_tensor, hull_mask):
    b, _, h, w = x0_tensor.shape
    time_map = torch.full((b, 1, h, w), float(t_norm_value), device=x0_tensor.device, dtype=x0_tensor.dtype)
    inp = torch.cat([x0_tensor, time_map], dim=1)  # (B,4,H,W)
    with torch.no_grad():
        pred = model(inp)
        pred = apply_clamp(pred)
        pred = apply_hard_constraints(pred, initial_tensor, hull_mask)
        pred = apply_clamp(pred)
    return pred


if __name__ == "__main__":
    ensure_parent(SAVE_FIG)

    comp_to_idx = {"u": 0, "v": 1, "w": 2}
    if COMPONENT not in comp_to_idx:
        raise ValueError("COMPONENT must be one of u|v|w")
    cidx = comp_to_idx[COMPONENT]

    data = np.load(DATA_PATH, mmap_mode="r")
    if STEP_IDX < 0 or STEP_IDX >= data.shape[0]:
        raise ValueError(f"STEP_IDX out of range: {STEP_IDX} for {data.shape[0]}")

    x0 = (data[0].astype(np.float32) / NORM_FACTOR)
    x0_tensor = torch.from_numpy(x0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    initial_tensor = x0_tensor.clone()
    hull_mask = compute_hull_mask(initial_tensor)

    T_total = 4500
    t_norm_value = time_norm(STEP_IDX, T_total)

    model_before = load_model(BEFORE_MODEL_PATH)
    model_after = load_model(AFTER_MODEL_PATH)

    pred_before = predict_at_step(model_before, x0_tensor, t_norm_value, initial_tensor, hull_mask)
    pred_after = predict_at_step(model_after, x0_tensor, t_norm_value, initial_tensor, hull_mask)

    gt = data[STEP_IDX].astype(np.float32)
    before = pred_before.squeeze(0).permute(1, 2, 0).cpu().numpy() * NORM_FACTOR
    after = pred_after.squeeze(0).permute(1, 2, 0).cpu().numpy() * NORM_FACTOR

    gt_c = gt[:, :, cidx]
    b_c = before[:, :, cidx]
    a_c = after[:, :, cidx]

    vmin = min(gt_c.min(), b_c.min(), a_c.min())
    vmax = max(gt_c.max(), b_c.max(), a_c.max())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    axes[0].imshow(gt_c, cmap="jet", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth (t={STEP_IDX*DT:.3f}s)")
    axes[0].axis("off")

    axes[1].imshow(b_c, cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].set_title("Before FT (One-shot)")
    axes[1].axis("off")

    axes[2].imshow(a_c, cmap="jet", vmin=vmin, vmax=vmax)
    axes[2].set_title("After FT (+Physics/Texture)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(SAVE_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved triptych: {SAVE_FIG}")
