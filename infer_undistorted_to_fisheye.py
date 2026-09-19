"""
Metric3D 去畸变图批量推理 -> 鱼眼深度图 + 鱼眼点云。

数据结构与 infer_undistorted.py 保持一致：
  --dataset-root
      ├── undistort/{view}/{timestamp}.jpg
      ├── new_K/{view}/new_K.npy
      ├── origin/{view}/{timestamp}.jpg      （原始鱼眼图，用于颜色/FOV）
      └── cloud/{view}/...                    （输出目录）

核心流程：
  1. 对去畸变图做 Metric3D 推理，得到针孔 Z 深度图。
  2. 用 calibration.yml 中的鱼眼参数把针孔 Z 深度图反投影回鱼眼坐标系。
  3. 保存鱼眼深度图，并用鱼眼模型（depth_is_z=True）生成点云。

与 infer_undistorted.py 的区别：
  - 生成点云时不再使用针孔内参，而是使用鱼眼内参和畸变系数。
  - 额外输出鱼眼格式的深度图。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

try:
    import yaml
except ImportError:
    yaml = None

from infer_common import (
    load_model,
    preprocess_image,
    recover_depth,
    save_fisheye_pcd,
    fisheye_undistort_points,
)


def guess_cam_key(img_path):
    """根据图片名推断 calibration.yml 中的相机 key。"""
    base = os.path.basename(img_path).lower()
    mapping = {
        'front': 'cam_hy_n5_avm_front',
        'back': 'cam_hy_n5_avm_back',
        'left': 'cam_hy_n5_avm_left',
        'right': 'cam_hy_n5_avm_right',
    }
    for k, v in mapping.items():
        if k in base:
            return v
    return None


def load_fisheye_calib(calib_path, view=None, cam_key=None, img_path=None):
    """从 calibration.yml 读取指定相机的鱼眼内参与畸变系数。"""
    if yaml is None:
        raise RuntimeError("需要安装 PyYAML 才能读取 YAML 标定文件: pip install pyyaml")

    with open(calib_path, 'r', encoding='utf-8') as f:
        calib = yaml.safe_load(f)

    if 'rig' not in calib:
        raise ValueError("calibration.yml 顶层缺少 'rig' 字段")

    rig = calib['rig']

    if cam_key is not None:
        print(f"使用指定相机: {cam_key}")
    elif view is not None:
        cam_key = f'cam_hy_n5_avm_{view}'
        print(f"根据视角 {view} 选择相机: {cam_key}")
    else:
        cam_key = guess_cam_key(img_path)
        if cam_key is None:
            raise ValueError(f"无法从图片名推断相机 key，请显式指定 --cam 或 --view")
        print(f"根据图片名自动选择相机: {cam_key}")

    if cam_key not in rig:
        available = [k for k in rig.keys() if 'avm' in k or 'fisheye' in k.lower()]
        raise ValueError(f"calibration.yml 中找不到 {cam_key}，可用相机: {available}")

    cam = rig[cam_key]
    focal = cam['focal']
    pp = cam['pp']
    inv_poly = cam.get('inv_poly', [0.0, 0.0, 0.0, 0.0])

    K = np.array([
        [focal[0], 0.0,      pp[0]],
        [0.0,      focal[1], pp[1]],
        [0.0,      0.0,      1.0]
    ], dtype=np.float64)

    D = np.array(inv_poly, dtype=np.float64)

    print(f"从 {cam_key} 加载鱼眼参数:")
    print(f"  标定图像尺寸: {cam.get('image_size')}")
    print(f"  K:\n{K}")
    print(f"  D: {D}")

    return K, D, cam.get('image_size')


def load_pinhole_K(K_path):
    """加载去畸变后的针孔内参 K。支持 .npy 和 .yml/.yaml。"""
    if not os.path.isfile(K_path):
        raise FileNotFoundError(f"找不到针孔内参文件: {K_path}")

    ext = os.path.splitext(K_path)[1].lower()
    if ext in ['.yml', '.yaml']:
        if yaml is None:
            raise RuntimeError("需要安装 PyYAML 才能读取 YAML 标定文件")
        with open(K_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        K = np.array(data['K'], dtype=np.float64)
    elif ext == '.npy':
        K = np.load(K_path).astype(np.float64)
    else:
        raise ValueError(f"不支持的针孔内参文件格式: {ext}")

    if K.shape != (3, 3):
        raise ValueError(f"K 矩阵形状异常: {K.shape}，应为 (3, 3)")
    print(f"加载针孔内参: {K_path}\n{K}")
    return K


def scale_K_to_image(K, calib_size, actual_size):
    """如果实际图像尺寸与标定尺寸不同，按比例缩放 K。"""
    if calib_size is None or actual_size is None:
        return K
    w_cal, h_cal = calib_size
    h_act, w_act = actual_size
    if h_cal == h_act and w_cal == w_act:
        return K

    sx = w_act / w_cal
    sy = h_act / h_cal
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    print(f"[Warning] 实际图像尺寸 {actual_size} 与标定尺寸 {calib_size} 不同，"
          f"已按比例缩放 K ({sx:.4f}, {sy:.4f})")
    return K_scaled


def remap_depth_to_fisheye(depth_pin, K_fisheye, D_fisheye, K_pin,
                           img_fisheye_shape, max_depth, theta_max=1.75):
    """
    将针孔 Z 深度图反投影回鱼眼图像坐标系。
    使用自定义 KB 反解，避免 cv2.fisheye.undistortPoints 在大角度下的精度问题。
    返回鱼眼 Z 深度图和有效掩码。
    """
    H_f, W_f = img_fisheye_shape[:2]
    H_p, W_p = depth_pin.shape

    u_f, v_f = np.meshgrid(np.arange(W_f, dtype=np.float64),
                           np.arange(H_f, dtype=np.float64))
    pts_f = np.stack([u_f, v_f], axis=-1).reshape(-1, 1, 2)

    # 鱼眼像素 -> 归一化平面（自定义反解）
    xn, yn = fisheye_undistort_points(pts_f, K_fisheye, D_fisheye, theta_max=theta_max)
    xn = xn.reshape(H_f, W_f)
    yn = yn.reshape(H_f, W_f)

    # 归一化平面 -> 针孔图像坐标
    u_p = xn * K_pin[0, 0] + K_pin[0, 2]
    v_p = yn * K_pin[1, 1] + K_pin[1, 2]

    # 从针孔深度图采样 Z
    depth_fisheye = cv2.remap(
        depth_pin.astype(np.float32),
        u_p.astype(np.float32),
        v_p.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    valid_mask = (
        np.isfinite(u_p) & np.isfinite(v_p) &
        (u_p >= 0) & (u_p < W_p - 1) &
        (v_p >= 0) & (v_p < H_p - 1) &
        (depth_fisheye > 0.1) & (depth_fisheye < max_depth) &
        np.isfinite(depth_fisheye)
    )

    return depth_fisheye, valid_mask


def save_fisheye_depth_outputs(depth, valid_mask, out_dir, base_name, max_depth):
    """保存鱼眼深度图，无效区域设为 0。"""
    os.makedirs(out_dir, exist_ok=True)
    depth_clean = np.where(valid_mask, depth, 0.0)

    npy_path = os.path.join(out_dir, f'{base_name}_depth.npy')
    np.save(npy_path, depth_clean)
    print(f"  鱼眼深度图 (.npy) 已保存: {npy_path}")

    png_path = os.path.join(out_dir, f'{base_name}_depth.png')
    depth_uint16 = (depth_clean * 256.0).astype(np.uint16)
    cv2.imwrite(png_path, depth_uint16)
    print(f"  鱼眼深度图 (.png, scale=256) 已保存: {png_path}")

    depth_color = cv2.applyColorMap(
        (np.clip(depth_clean / max_depth, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    color_path = os.path.join(out_dir, f'{base_name}_depth_color.png')
    cv2.imwrite(color_path, depth_color)
    print(f"  鱼眼伪彩色深度图已保存: {color_path}")


def process_single(args):
    """单图模式。"""
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.img_fisheye))[0]

    # 1. 加载两种标定
    K_fisheye, D_fisheye, fisheye_calib_size = load_fisheye_calib(
        args.calib, cam_key=args.cam, img_path=args.img_fisheye
    )
    K_pin = load_pinhole_K(args.K)

    # 2. 加载模型并推理去畸变图
    cfg, model = load_model(args.config, args.ckpt)
    info = preprocess_image(args.img, args.input_size)
    img_rgb = info['img_rgb']
    img_tensor = info['img_tensor']
    ori_h, ori_w = info['ori_h'], info['ori_w']
    scale = info['scale']

    K_pin_scaled = K_pin.copy()
    K_pin_scaled[0, 0] *= scale
    K_pin_scaled[1, 1] *= scale
    K_pin_scaled[0, 2] *= scale
    K_pin_scaled[1, 2] *= scale

    print("\n开始推理去畸变图像...")
    pred_depth, _, _ = model.inference({'input': img_tensor})

    pred_depth_np = recover_depth(
        pred_depth, K_pin_scaled, cfg,
        ori_h, ori_w,
        info['pad_h_half'], info['pad_w_half'],
        info['new_h'], info['new_w'],
        args.max_depth
    )

    # 可选：按上下比例过滤针孔深度图
    pinhole_mask = np.ones((ori_h, ori_w), dtype=bool)
    if args.ignore_top_ratio > 0:
        top_rows = int(ori_h * args.ignore_top_ratio)
        pinhole_mask[:top_rows, :] = False
    if args.ignore_bottom_ratio > 0:
        bottom_rows = int(ori_h * args.ignore_bottom_ratio)
        pinhole_mask[-bottom_rows:, :] = False
    pred_depth_np_masked = np.where(pinhole_mask, pred_depth_np, 0.0)

    # 3. 读取原始鱼眼图
    img_fisheye_bgr = cv2.imread(args.img_fisheye)
    if img_fisheye_bgr is None:
        raise FileNotFoundError(f"无法读取鱼眼图像: {args.img_fisheye}")
    img_fisheye_rgb = cv2.cvtColor(img_fisheye_bgr, cv2.COLOR_BGR2RGB)
    print(f"\n读取鱼眼图像: {args.img_fisheye}, 尺寸: {img_fisheye_rgb.shape[:2]}")

    K_fisheye_scaled = scale_K_to_image(
        K_fisheye, fisheye_calib_size, img_fisheye_rgb.shape[:2]
    )

    # 4. 针孔深度图 -> 鱼眼深度图
    print("\n将针孔深度图反投影回鱼眼坐标系...")
    depth_fisheye, valid_mask = remap_depth_to_fisheye(
        pred_depth_np_masked, K_fisheye_scaled, D_fisheye, K_pin,
        img_fisheye_rgb.shape[:2], args.max_depth
    )

    save_fisheye_depth_outputs(
        depth_fisheye, valid_mask, args.out_dir,
        f'{base_name}_fisheye', args.max_depth
    )

    # 5. 鱼眼深度图 -> 点云
    if args.save_pcd:
        print("\n生成鱼眼点云...")
        save_fisheye_pcd(
            depth_fisheye,
            K_fisheye_scaled, D_fisheye, img_fisheye_rgb,
            args.out_dir, f'{base_name}_fisheye', args.max_depth,
            depth_is_z=True, no_filter=True
        )

    print("\n单图推理完成。")


def process_batch(args):
    """批量模式，结构与 infer_undistorted.py 完全一致。"""
    undistort_root = args.undistort_root
    newK_root = args.newK_root
    cloud_root = args.cloud_root
    origin_root = args.origin_root
    views = args.views

    os.makedirs(cloud_root, exist_ok=True)

    cfg, model = load_model(args.config, args.ckpt)
    print("模型加载完成。")

    for view in views:
        undistort_view_dir = os.path.join(undistort_root, view)
        if not os.path.isdir(undistort_view_dir):
            print(f"[Warning] 跳过不存在的目录: {undistort_view_dir}")
            continue

        K_path = os.path.join(newK_root, view, 'new_K.npy')
        if not os.path.isfile(K_path):
            print(f"[Warning] 找不到针孔内参: {K_path}，跳过视角 {view}")
            continue
        K_pin = load_pinhole_K(K_path)

        origin_view_dir = os.path.join(origin_root, view)
        if not os.path.isdir(origin_view_dir):
            print(f"[Warning] 找不到原始鱼眼图目录: {origin_view_dir}，跳过视角 {view}")
            continue

        K_fisheye, D_fisheye, fisheye_calib_size = load_fisheye_calib(
            args.calib, view=view
        )

        cloud_view_dir = os.path.join(cloud_root, view)
        os.makedirs(cloud_view_dir, exist_ok=True)

        img_files = sorted([f for f in os.listdir(undistort_view_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"\n[{view}] 发现 {len(img_files)} 张去畸变图像")

        for img_file in img_files:
            img_path = os.path.join(undistort_view_dir, img_file)
            img_fisheye_path = os.path.join(origin_view_dir, img_file)
            if not os.path.isfile(img_fisheye_path):
                print(f"[Warning] 找不到对应鱼眼图: {img_fisheye_path}，跳过 {img_file}")
                continue

            base_name = os.path.splitext(img_file)[0]
            print(f"\n[{view}] 处理: {img_file}")

            info = preprocess_image(img_path, args.input_size)
            img_rgb = info['img_rgb']
            img_tensor = info['img_tensor']
            ori_h, ori_w = info['ori_h'], info['ori_w']
            scale = info['scale']

            K_pin_scaled = K_pin.copy()
            K_pin_scaled[0, 0] *= scale
            K_pin_scaled[1, 1] *= scale
            K_pin_scaled[0, 2] *= scale
            K_pin_scaled[1, 2] *= scale

            pred_depth, _, _ = model.inference({'input': img_tensor})

            pred_depth_np = recover_depth(
                pred_depth, K_pin_scaled, cfg,
                ori_h, ori_w,
                info['pad_h_half'], info['pad_w_half'],
                info['new_h'], info['new_w'],
                args.max_depth
            )

            # 可选：上下区域过滤
            pinhole_mask = np.ones((ori_h, ori_w), dtype=bool)
            if args.ignore_top_ratio > 0:
                top_rows = int(ori_h * args.ignore_top_ratio)
                pinhole_mask[:top_rows, :] = False
            if args.ignore_bottom_ratio > 0:
                bottom_rows = int(ori_h * args.ignore_bottom_ratio)
                pinhole_mask[-bottom_rows:, :] = False
            pred_depth_np_masked = np.where(pinhole_mask, pred_depth_np, 0.0)

            # 读取原始鱼眼图（颜色与尺寸）
            img_fisheye_bgr = cv2.imread(img_fisheye_path)
            if img_fisheye_bgr is None:
                print(f"[Warning] 无法读取鱼眼图: {img_fisheye_path}")
                continue
            img_fisheye_rgb = cv2.cvtColor(img_fisheye_bgr, cv2.COLOR_BGR2RGB)

            K_fisheye_scaled = scale_K_to_image(
                K_fisheye, fisheye_calib_size, img_fisheye_rgb.shape[:2]
            )

            # 针孔深度 -> 鱼眼深度
            depth_fisheye, valid_mask = remap_depth_to_fisheye(
                pred_depth_np_masked, K_fisheye_scaled, D_fisheye, K_pin,
                img_fisheye_rgb.shape[:2], args.max_depth
            )

            save_fisheye_depth_outputs(
                depth_fisheye, valid_mask, cloud_view_dir,
                f'{base_name}_fisheye', args.max_depth
            )

            if args.save_pcd:
                save_fisheye_pcd(
                    depth_fisheye,
                    K_fisheye_scaled, D_fisheye, img_fisheye_rgb,
                    cloud_view_dir, f'{base_name}_fisheye', args.max_depth,
                    depth_is_z=True, no_filter=True
                )

            print(f"[{view}] 完成: {img_file}")

    print("\n批量推理完成。")


def main():
    parser = argparse.ArgumentParser(
        description='Metric3D 去畸变图 -> 鱼眼深度图 + 鱼眼点云（批量/单图）'
    )
    # 单图模式
    parser.add_argument('--img', default=None, help='输入去畸变图像路径')
    parser.add_argument('--K', default=None,
                        help='去畸变后的针孔内参：.npy 或 .yml/.yaml 文件路径')
    parser.add_argument('--img-fisheye', default=None, help='原始鱼眼图像路径')
    parser.add_argument('--cam', default=None,
                        help='calibration.yml 中的相机 key，例如 cam_hy_n5_avm_front')
    parser.add_argument('--out-dir', default='./output_undistorted_to_fisheye',
                        help='单图模式输出目录')

    # 批量模式
    parser.add_argument('--dataset-root', default=None,
                        help='批量模式：数据集根目录，会自动推导 undistort/newK/origin/cloud 路径')
    parser.add_argument('--undistort-root', default=None,
                        help='批量模式：去畸变图根目录（默认 dataset-root/undistort）')
    parser.add_argument('--newK-root', default=None,
                        help='批量模式：针孔内参根目录（默认 dataset-root/new_K）')
    parser.add_argument('--origin-root', default=None,
                        help='批量模式：原始鱼眼图根目录（默认 dataset-root/origin）')
    parser.add_argument('--cloud-root', default=None,
                        help='批量模式：输出根目录（默认 dataset-root/cloud）')
    parser.add_argument('--views', nargs='+', default=['front', 'back', 'left', 'right'],
                        help='要处理的视角列表')
    parser.add_argument('--ignore-top-ratio', type=float, default=0.0,
                        help='针孔深度图上方忽略比例（去除天空）')
    parser.add_argument('--ignore-bottom-ratio', type=float, default=0.0,
                        help='针孔深度图下方忽略比例（去除车体/地面）')

    # 公共参数
    parser.add_argument('--calib', required=True, help='鱼眼 calibration.yml 路径')
    parser.add_argument('--config', required=True, help='模型配置文件路径')
    parser.add_argument('--ckpt', required=True, help='模型权重路径')
    parser.add_argument('--input-size', default='616,1064', help='模型输入尺寸 (H,W)')
    parser.add_argument('--max-depth', type=float, default=200.0, help='深度上限')
    parser.add_argument('--save-pcd', action='store_true', help='是否保存点云')
    args = parser.parse_args()

    # dataset-root 自动推导
    if args.dataset_root is not None:
        if args.undistort_root is None:
            args.undistort_root = os.path.join(args.dataset_root, 'undistort')
        if args.newK_root is None:
            args.newK_root = os.path.join(args.dataset_root, 'new_K')
        if args.origin_root is None:
            args.origin_root = os.path.join(args.dataset_root, 'origin')
        if args.cloud_root is None:
            args.cloud_root = os.path.join(args.dataset_root, 'cloud')
    else:
        # 默认批量路径
        if args.undistort_root is None:
            args.undistort_root = '/root/ly/Map/Datas/undistort'
        if args.newK_root is None:
            args.newK_root = '/root/ly/Map/Datas/new_K'
        if args.origin_root is None:
            args.origin_root = '/root/ly/Map/Datas/origin'
        if args.cloud_root is None:
            args.cloud_root = '/root/ly/Map/Metric3D/cloud'

    if args.img or args.K or args.img_fisheye:
        if not args.img or not args.K or not args.img_fisheye:
            raise ValueError("单图模式需要同时提供 --img、--K 和 --img-fisheye")
        process_single(args)
    else:
        process_batch(args)


if __name__ == '__main__':
    main()
