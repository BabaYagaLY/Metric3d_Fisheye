import os
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

try:
    from mmcv.utils import Config
except Exception:
    from mmengine import Config

from mono.model.monodepth_model import get_configured_monodepth_model


def resolve_config_path(cfg_path):
    """
    尝试定位配置文件：
    1) 直接使用用户给的路径；
    2) 相对于项目根目录；
    3) 相对于 training/ 子目录（因为 RAFTDecoder 配置实际在 training/mono/configs 下）。
    """
    if os.path.isfile(cfg_path):
        return cfg_path

    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, cfg_path),
        os.path.join(root, 'training', cfg_path),
        os.path.join(root, 'training', 'mono', 'configs', cfg_path),
        os.path.join(root, 'mono', 'configs', cfg_path),
    ]
    for c in candidates:
        if os.path.isfile(c):
            print(f"[Info] 配置文件自动定位到: {c}")
            return c

    raise FileNotFoundError(
        f"找不到配置文件: {cfg_path}\n"
        f"请确认路径正确。常用位置在 training/mono/configs/RAFTDecoder/ 或 mono/configs/HourglassDecoder/"
    )


def load_model(cfg_path, ckpt_path):
    """加载 Metric3D 模型与配置（强制使用 GPU）。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 GPU 驱动和 torch 是否编译了 CUDA。")

    cfg_path = resolve_config_path(cfg_path)
    cfg = Config.fromfile(cfg_path)
    model = get_configured_monodepth_model(cfg)

    print(f"正在读取权重: {ckpt_path}")
    # 尝试用 mmap 读取，减少 WSL 下的大文件内存占用（需要 PyTorch >= 2.0）
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', mmap=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    print("权重读取完成，正在加载到模型...")
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)

    # 立即释放 checkpoint 占用的 CPU 内存
    del ckpt, state
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("模型权重加载完成，正在移动到 GPU...")

    model = model.cuda().eval()
    print(f"模型已加载到 GPU: {next(model.parameters()).device}")
    return cfg, model


def parse_input_size(value):
    """兼容 '616,1064' 字符串 与 [616, 1064] 列表。"""
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        parts = value.split(',')
        return int(parts[0]), int(parts[1])
    raise ValueError(f"无法解析 input_size: {value}")


def preprocess_image(img_path, input_size):
    """
    读取图像、等比缩放、padding、归一化。
    返回字典，包含原图、tensor、缩放比例、padding 信息等。
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = img_rgb.shape[:2]

    input_h, input_w = parse_input_size(input_size)
    scale = min(input_h / ori_h, input_w / ori_w)
    new_h, new_w = int(ori_h * scale + 0.5), int(ori_w * scale + 0.5)
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    mean_rgb = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std_rgb = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    pad_h = input_h - new_h
    pad_w = input_w - new_w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2

    img_pad = cv2.copyMakeBorder(
        img_resized,
        pad_h_half, pad_h - pad_h_half,
        pad_w_half, pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=mean_rgb.tolist()
    )

    img_tensor = torch.from_numpy(img_pad.transpose(2, 0, 1)).float()
    img_tensor = (img_tensor - torch.from_numpy(mean_rgb).view(3, 1, 1)) / \
                 torch.from_numpy(std_rgb).view(3, 1, 1)
    img_tensor = img_tensor.unsqueeze(0).cuda()

    return {
        'img_rgb': img_rgb,
        'img_pad': img_pad,        # 未归一化的 padded 图像（RGB, 0~255），TRT 用
        'img_tensor': img_tensor,  # 归一化后的 tensor，PyTorch 用
        'ori_h': ori_h,
        'ori_w': ori_w,
        'scale': scale,
        'new_h': new_h,
        'new_w': new_w,
        'pad_h_half': pad_h_half,
        'pad_w_half': pad_w_half,
    }


def recover_depth(pred_depth, K_scaled, cfg, ori_h, ori_w,
                  pad_h_half, pad_w_half, new_h, new_w, max_depth):
    """
    去 padding、插值回原分辨率、用真实焦距恢复尺度、裁剪到 max_depth。
    """
    canonical_focal = cfg.data_basic.canonical_space.focal_length
    real_focal_x = K_scaled[0, 0]

    pred_depth = pred_depth.squeeze().cpu()
    pred_depth = pred_depth[
        pad_h_half: pad_h_half + new_h,
        pad_w_half: pad_w_half + new_w
    ]
    pred_depth = F.interpolate(
        pred_depth.unsqueeze(0).unsqueeze(0),
        size=(ori_h, ori_w),
        mode='nearest'
    ).squeeze()

    pred_depth = pred_depth * (real_focal_x / canonical_focal)
    pred_depth = torch.clamp(pred_depth, 0, max_depth)
    return pred_depth.numpy()


def save_depth_outputs(pred_depth_np, img_rgb, out_dir, base_name, max_depth):
    """保存 npy、16-bit PNG、伪彩色深度图。"""
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, f'{base_name}_depth.npy')
    np.save(npy_path, pred_depth_np)
    print(f"深度图 (.npy) 已保存: {npy_path}")

    png_path = os.path.join(out_dir, f'{base_name}_depth.png')
    depth_uint16 = (pred_depth_np * 256.0).astype(np.uint16)
    cv2.imwrite(png_path, depth_uint16)
    print(f"深度图 (.png, scale=256) 已保存: {png_path}")

    depth_color = cv2.applyColorMap(
        (np.clip(pred_depth_np / max_depth, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    color_path = os.path.join(out_dir, f'{base_name}_depth_color.png')
    cv2.imwrite(color_path, depth_color)
    print(f"伪彩色深度图已保存: {color_path}")


def save_ply(filename, points, colors):
    """保存点云为 PLY 格式（需要 plyfile）。"""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("[Warning] 未安装 plyfile，跳过 PLY 保存。可执行 pip install plyfile")
        return

    vertices = np.hstack([points.astype(np.float32), colors.astype(np.uint8)])
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_array = np.array([tuple(v) for v in vertices], dtype=dtype)
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el]).write(filename)


def save_pinhole_pcd(pred_depth_np, K, img_rgb, out_dir, base_name, max_depth,
                     extra_mask=None):
    """针孔 / 去畸变图像反投影为点云。extra_mask 为可选的额外掩码（True 表示保留）。"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ori_h, ori_w = pred_depth_np.shape

    xs = np.arange(ori_w)
    ys = np.arange(ori_h)
    xv, yv = np.meshgrid(xs, ys)
    x_norm = (xv - cx) / fx
    y_norm = (yv - cy) / fy

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    valid_mask = (pred_depth_np > 0.1) & (pred_depth_np < max_depth) & (gray > 15)
    if extra_mask is not None:
        valid_mask &= extra_mask

    z = pred_depth_np[valid_mask]
    x = x_norm[valid_mask] * z
    y = y_norm[valid_mask] * z
    colors = img_rgb[valid_mask]

    pcd_cam = np.stack([x, y, z], axis=-1)
    ply_path = os.path.join(out_dir, f'{base_name}_pointcloud.ply')
    save_ply(ply_path, pcd_cam, colors)
    print(f"点云已保存: {ply_path}")


def _kb_forward(theta, D):
    """Kannala-Brandt 正向投影：rho = theta * (1 + k1*theta^2 + k2*theta^4 + ...)."""
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    return theta * (1.0 + D[0]*t2 + D[1]*t4 + D[2]*t6 + D[3]*t8)


def _kb_forward_derivative(theta, D):
    """KB 正向投影对 theta 的导数。"""
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    P = 1.0 + D[0]*t2 + D[1]*t4 + D[2]*t6 + D[3]*t8
    dP = (2.0*D[0]*theta + 4.0*D[1]*theta**3 +
          6.0*D[2]*theta**5 + 8.0*D[3]*theta**7)
    return P + theta * dP


def solve_kb_inverse(rho, D, theta_max=1.75, max_iter=30, eps=1e-12):
    """
    求解 rho = theta * (1 + k1*theta^2 + k2*theta^4 + ...) 的逆。
    使用带 clip 的 Newton 法，对无效 rho 返回 NaN。
    """
    rho = np.asarray(rho, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if D.size < 4:
        D = np.pad(D, (0, 4 - D.size), mode='constant')

    # 有效范围：rho <= theta_max 对应的最大正向半径
    max_r = _kb_forward(theta_max, D)
    active = (rho > 1e-9) & (rho <= max_r)

    theta = np.full_like(rho, np.nan, dtype=np.float64)
    theta[active] = np.minimum(rho[active], theta_max)

    for _ in range(max_iter):
        if not np.any(active):
            break
        f = _kb_forward(theta[active], D) - rho[active]
        df = _kb_forward_derivative(theta[active], D)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = f / df
        theta_new = np.clip(theta[active] - delta, 0.0, theta_max)
        converged = np.abs(delta) < eps
        theta[active] = theta_new
        active[active] = ~converged
    return theta


def fisheye_undistort_points(pts2d, K, D, theta_max=1.75):
    """
    自定义鱼眼反投影：把鱼眼图像素映射到归一化平面 (xn, yn)。
    不依赖 cv2.fisheye.undistortPoints，避免该函数在大角度下收敛错误。
    """
    pts2d = np.asarray(pts2d, dtype=np.float64)
    if pts2d.ndim == 3 and pts2d.shape[1] == 1:
        u = pts2d[:, 0, 0]
        v = pts2d[:, 0, 1]
    else:
        u = pts2d[:, 0]
        v = pts2d[:, 1]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    rho = np.sqrt(x*x + y*y)

    theta = solve_kb_inverse(rho, D, theta_max=theta_max)
    scale = np.ones_like(rho, dtype=np.float64)
    nonzero = rho > 1e-9
    scale[nonzero] = np.tan(theta[nonzero]) / rho[nonzero]

    xn = x * scale
    yn = y * scale
    return xn, yn


def save_fisheye_pcd(pred_depth_np, K, D, img_rgb, out_dir, base_name,
                     max_depth, depth_is_z=False, theta_max=1.75, no_filter=False):
    """鱼眼原图反投影为点云。"""
    ori_h, ori_w = pred_depth_np.shape

    u = np.arange(ori_w)
    v = np.arange(ori_h)
    uu, vv = np.meshgrid(u, v)
    pts2d = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
    pts2d = pts2d.reshape(-1, 1, 2)

    xn, yn = fisheye_undistort_points(pts2d, K, D, theta_max=theta_max)
    d = pred_depth_np.ravel()

    if depth_is_z:
        X = xn * d
        Y = yn * d
        Z = d
    else:
        r2 = xn * xn + yn * yn
        inv_norm = 1.0 / np.sqrt(1.0 + r2)
        ray_x = xn * inv_norm
        ray_y = yn * inv_norm
        ray_z = inv_norm
        X = d * ray_x
        Y = d * ray_y
        Z = d * ray_z

    if no_filter:
        valid_mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    else:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        valid_mask = (d > 0.1) & (d < max_depth) & (gray.ravel() > 15)
        valid_mask &= np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)

    points = np.stack([X, Y, Z], axis=1)[valid_mask]
    colors = img_rgb.reshape(-1, 3)[valid_mask]

    ply_path = os.path.join(out_dir, f'{base_name}_pointcloud.ply')
    save_ply(ply_path, points, colors)
    print(f"点云已保存: {ply_path}")
