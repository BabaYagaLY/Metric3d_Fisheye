"""
Metric3D 鱼眼原图直接推理。

与 infer_undistorted.py 保持一致的输入/输出结构：
  - 单图模式：--img + --calib
  - 批量模式：--dataset-root（自动推导 origin/cloud）

输出格式：
  - 批量模式只生成点云：cloud/{view}/{timestamp}_pointcloud.ply
  - 单图模式额外生成深度图：*_depth.npy/png/color.png

反投影默认使用 Z 深度（depth_is_z=True），因为 Metric3D 原生输出就是 Z 深度。
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
    save_depth_outputs,
    save_fisheye_pcd,
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
            raise ValueError("无法从图片名推断相机 key，请显式指定 --cam 或 --view")
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


def process_single(args):
    """单图模式：输出深度图 + 点云（与 infer_undistorted.py 单图模式一致）。"""
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.img))[0]

    K, D, calib_size = load_fisheye_calib(args.calib, cam_key=args.cam, img_path=args.img)

    cfg, model = load_model(args.config, args.ckpt)
    print("模型加载完成。")

    info = preprocess_image(args.img, args.input_size)
    img_rgb = info['img_rgb']
    img_tensor = info['img_tensor']
    ori_h, ori_w = info['ori_h'], info['ori_w']
    scale = info['scale']

    # K 对齐到实际图像尺寸，并按模型输入缩放
    K_actual = scale_K_to_image(K, calib_size, (ori_h, ori_w))
    K_scaled = K_actual.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale

    print("\n开始推理...")
    pred_depth, _, _ = model.inference({'input': img_tensor})

    pred_depth_np = recover_depth(
        pred_depth, K_scaled, cfg,
        ori_h, ori_w,
        info['pad_h_half'], info['pad_w_half'],
        info['new_h'], info['new_w'],
        args.max_depth
    )

    # 可选：上下区域过滤（与 infer_undistorted.py 一致）
    extra_mask = np.ones((ori_h, ori_w), dtype=bool)
    if args.ignore_top_ratio > 0:
        top_rows = int(ori_h * args.ignore_top_ratio)
        extra_mask[:top_rows, :] = False
    if args.ignore_bottom_ratio > 0:
        bottom_rows = int(ori_h * args.ignore_bottom_ratio)
        extra_mask[-bottom_rows:, :] = False
    pred_depth_np = np.where(extra_mask, pred_depth_np, 0.0)

    # 保存深度图
    save_depth_outputs(pred_depth_np, img_rgb, args.out_dir, base_name, args.max_depth)

    # 保存鱼眼点云（使用 Z 深度）
    if args.save_pcd:
        save_fisheye_pcd(
            pred_depth_np, K_actual, D, img_rgb,
            args.out_dir, base_name, args.max_depth,
            depth_is_z=True
        )

    print("\n单图推理完成。")


def process_batch(args):
    """批量模式：结构与 infer_undistorted.py 一致。"""
    origin_root = args.origin_root
    cloud_root = args.cloud_root
    views = args.views

    os.makedirs(cloud_root, exist_ok=True)

    cfg, model = load_model(args.config, args.ckpt)
    print("模型加载完成。")

    for view in views:
        origin_view_dir = os.path.join(origin_root, view)
        if not os.path.isdir(origin_view_dir):
            print(f"[Warning] 跳过不存在的目录: {origin_view_dir}")
            continue

        K, D, calib_size = load_fisheye_calib(args.calib, view=view, cam_key=args.cam)

        cloud_view_dir = os.path.join(cloud_root, view)
        os.makedirs(cloud_view_dir, exist_ok=True)

        img_files = sorted([f for f in os.listdir(origin_view_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"\n[{view}] 发现 {len(img_files)} 张鱼眼图像")

        for img_file in img_files:
            img_path = os.path.join(origin_view_dir, img_file)
            base_name = os.path.splitext(img_file)[0]

            info = preprocess_image(img_path, args.input_size)
            img_rgb = info['img_rgb']
            img_tensor = info['img_tensor']
            ori_h, ori_w = info['ori_h'], info['ori_w']
            scale = info['scale']

            K_actual = scale_K_to_image(K, calib_size, (ori_h, ori_w))
            K_scaled = K_actual.copy()
            K_scaled[0, 0] *= scale
            K_scaled[1, 1] *= scale
            K_scaled[0, 2] *= scale
            K_scaled[1, 2] *= scale

            pred_depth, _, _ = model.inference({'input': img_tensor})

            pred_depth_np = recover_depth(
                pred_depth, K_scaled, cfg,
                ori_h, ori_w,
                info['pad_h_half'], info['pad_w_half'],
                info['new_h'], info['new_w'],
                args.max_depth
            )

            extra_mask = np.ones((ori_h, ori_w), dtype=bool)
            if args.ignore_top_ratio > 0:
                top_rows = int(ori_h * args.ignore_top_ratio)
                extra_mask[:top_rows, :] = False
            if args.ignore_bottom_ratio > 0:
                bottom_rows = int(ori_h * args.ignore_bottom_ratio)
                extra_mask[-bottom_rows:, :] = False
            pred_depth_np = np.where(extra_mask, pred_depth_np, 0.0)

            # 批量模式只保存点云（与 infer_undistorted.py 一致）
            save_fisheye_pcd(
                pred_depth_np, K_actual, D, img_rgb,
                cloud_view_dir, base_name, args.max_depth,
                depth_is_z=True
            )

            print(f"[{view}] 点云完成: {img_file}")

    print("\n批量推理完成。")


def main():
    parser = argparse.ArgumentParser(
        description='Metric3D 鱼眼原图直接推理（Z 深度反投影）'
    )
    # 单图模式
    parser.add_argument('--img', default=None, help='输入鱼眼图像路径')
    parser.add_argument('--cam', default=None,
                        help='calibration.yml 中的相机 key，例如 cam_hy_n5_avm_front')
    parser.add_argument('--out-dir', default='./output_fisheye_infer',
                        help='单图模式输出目录')

    # 批量模式（与 infer_undistorted.py 对齐）
    parser.add_argument('--dataset-root', default=None,
                        help='批量模式：数据集根目录，会自动推导 origin/cloud 路径')
    parser.add_argument('--origin-root', default=None,
                        help='批量模式：鱼眼原图根目录（默认 dataset-root/origin）')
    parser.add_argument('--cloud-root', default=None,
                        help='批量模式：点云输出根目录（默认 dataset-root/cloud）')
    parser.add_argument('--views', nargs='+', default=['front', 'back', 'left', 'right'],
                        help='要处理的视角列表')
    parser.add_argument('--ignore-top-ratio', type=float, default=0.0,
                        help='点云构建时忽略图像上方多少比例区域（0.0~1.0），用于去除天空')
    parser.add_argument('--ignore-bottom-ratio', type=float, default=0.0,
                        help='点云构建时忽略图像下方多少比例区域（0.0~1.0），用于去除车体/地面')

    # 公共参数
    parser.add_argument('--calib', required=True, help='鱼眼 calibration.yml 路径')
    parser.add_argument('--config', required=True, help='模型配置文件路径')
    parser.add_argument('--ckpt', required=True, help='模型权重路径')
    parser.add_argument('--input-size', default='616,1064', help='模型输入尺寸 (H,W)，需为 14 的倍数')
    parser.add_argument('--max-depth', type=float, default=200.0, help='深度上限')
    parser.add_argument('--save-pcd', action='store_true',
                        help='单图模式下是否保存点云（批量模式始终保存点云）')
    args = parser.parse_args()

    # dataset-root 自动推导
    if args.dataset_root is not None:
        if args.origin_root is None:
            args.origin_root = os.path.join(args.dataset_root, 'origin')
        if args.cloud_root is None:
            args.cloud_root = os.path.join(args.dataset_root, 'cloud')
    else:
        if args.origin_root is None:
            args.origin_root = '/root/ly/Map/Datas/origin'
        if args.cloud_root is None:
            args.cloud_root = '/root/ly/Map/Metric3D/cloud_fisheye'

    if args.img is not None:
        process_single(args)
    else:
        process_batch(args)


if __name__ == '__main__':
    main()
