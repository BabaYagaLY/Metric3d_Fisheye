"""
Metric3D 去畸变/针孔图像推理：
  1) 单图模式：--img + --K
  2) 批量模式：--origin-root + --mask-root + --calib，自动完成 mask -> 去畸变 -> 推理 -> 点云
  3) 批量模式（旧）：--undistort-root + --newK-root + --cloud-root，处理预去畸变图
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from infer_common import (
    load_model,
    preprocess_image,
    recover_depth,
    save_depth_outputs,
    save_pinhole_pcd,
)

from infer_fisheye import load_fisheye_calib, scale_K_to_image


def parse_K(K_str):
    """
    支持两种输入：
      1) 逗号分隔的 9 个数字
      2) .npy 文件路径
    """
    if os.path.isfile(K_str):
        K = np.load(K_str)
    else:
        vals = [float(x) for x in K_str.split(',')]
        if len(vals) != 9:
            raise ValueError("--K 必须是 9 个数字（逗号分隔）或一个 .npy 文件路径")
        K = np.array(vals, dtype=np.float32).reshape(3, 3)
    return K.astype(np.float32)


def load_mask(mask_path, target_size):
    """加载单视角静态掩码并二值化。如果尺寸不匹配，用最近邻插值调整。"""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[0:2] != target_size:
        mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def undistort_fisheye_image(img_bgr, K, D, balance=0.0):
    """对鱼眼图去畸变，返回去畸变图、新针孔内参和 remap 映射。"""
    h, w = img_bgr.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
    )
    undistorted = cv2.remap(img_bgr, map1, map2, cv2.INTER_LINEAR)
    return undistorted, new_K.astype(np.float32), map1, map2


def warp_mask_to_undistorted(mask, map1, map2):
    """把鱼眼掩码用同一组去畸变映射 warp 到去畸变空间。"""
    warped = cv2.remap(
        mask, map1, map2, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return warped.astype(bool)


def process_single(args):
    """单图模式。"""
    os.makedirs(args.out_dir, exist_ok=True)

    K = parse_K(args.K)
    print(f"去畸变相机内参 K:\n{K}")

    cfg, model = load_model(args.config, args.ckpt)
    print("模型加载完成。")

    info = preprocess_image(args.img, args.input_size)
    img_rgb = info['img_rgb']
    img_tensor = info['img_tensor']
    ori_h, ori_w = info['ori_h'], info['ori_w']
    scale = info['scale']

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale

    print("开始推理...")
    pred_depth, _, _ = model.inference({'input': img_tensor})

    pred_depth_np = recover_depth(
        pred_depth, K_scaled, cfg,
        ori_h, ori_w,
        info['pad_h_half'], info['pad_w_half'],
        info['new_h'], info['new_w'],
        args.max_depth
    )

    base_name = os.path.splitext(os.path.basename(args.img))[0]
    save_depth_outputs(pred_depth_np, img_rgb, args.out_dir, base_name, args.max_depth)

    if args.save_pcd:
        save_pinhole_pcd(pred_depth_np, K, img_rgb, args.out_dir, base_name, args.max_depth)

    print("\n推理完成。")


def process_batch(args):
    """批量模式入口：根据是否提供 --calib 选择走鱼眼全管线还是预去畸变图。"""
    if args.calib is not None:
        process_batch_from_fisheye(args)
    else:
        process_batch_from_undistort(args)


def process_batch_from_undistort(args):
    """批量模式：处理 --undistort-root 下各视角的预去畸变图。"""
    undistort_root = args.undistort_root
    newK_root = args.newK_root
    cloud_root = args.cloud_root
    depth_root = args.depth_root
    views = args.views

    os.makedirs(cloud_root, exist_ok=True)
    os.makedirs(depth_root, exist_ok=True)

    cfg, model = load_model(args.config, args.ckpt)
    print("模型加载完成。")

    for view in views:
        undistort_view_dir = os.path.join(undistort_root, view)
        if not os.path.isdir(undistort_view_dir):
            print(f"[Warning] 跳过不存在的目录: {undistort_view_dir}")
            continue

        K_path = os.path.join(newK_root, view, 'new_K.npy')
        if not os.path.isfile(K_path):
            print(f"[Warning] 找不到内参文件: {K_path}，跳过视角 {view}")
            continue
        K = np.load(K_path).astype(np.float32)
        print(f"\n[{view}] 加载内参:\n{K}")

        cloud_view_dir = os.path.join(cloud_root, view)
        os.makedirs(cloud_view_dir, exist_ok=True)

        depth_view_dir = os.path.join(depth_root, view)
        os.makedirs(depth_view_dir, exist_ok=True)

        img_files = sorted([f for f in os.listdir(undistort_view_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"[{view}] 发现 {len(img_files)} 张图像")

        for img_file in img_files:
            img_path = os.path.join(undistort_view_dir, img_file)
            base_name = os.path.splitext(img_file)[0]

            info = preprocess_image(img_path, args.input_size)
            img_rgb = info['img_rgb']
            img_tensor = info['img_tensor']
            ori_h, ori_w = info['ori_h'], info['ori_w']
            scale = info['scale']

            K_scaled = K.copy()
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

            # 批量模式下同时保存深度图
            save_depth_outputs(pred_depth_np, img_rgb, depth_view_dir, base_name, args.max_depth)

            # 构建额外掩码：可以去掉天空（上方）或地面（下方）区域
            extra_mask = np.ones((ori_h, ori_w), dtype=bool)
            if args.ignore_top_ratio > 0:
                top_rows = int(ori_h * args.ignore_top_ratio)
                extra_mask[:top_rows, :] = False
            if args.ignore_bottom_ratio > 0:
                bottom_rows = int(ori_h * args.ignore_bottom_ratio)
                extra_mask[-bottom_rows:, :] = False

            # 批量模式下保存点云和深度图
            save_pinhole_pcd(pred_depth_np, K, img_rgb, cloud_view_dir, base_name,
                             args.max_depth, extra_mask=extra_mask)

            print(f"[{view}] 点云完成: {img_file}")

    print("\n批量推理完成。")


def process_batch_from_fisheye(args):
    """批量模式：从原始鱼眼图开始，mask -> 去畸变 -> 推理 -> 深度图/点云。"""
    origin_root = args.origin_root
    mask_root = args.mask_root
    undistort_root = args.undistort_root
    newK_root = args.newK_root
    cloud_root = args.cloud_root
    depth_root = args.depth_root
    views = args.views

    os.makedirs(undistort_root, exist_ok=True)
    os.makedirs(newK_root, exist_ok=True)
    os.makedirs(cloud_root, exist_ok=True)
    os.makedirs(depth_root, exist_ok=True)

    cfg, model = load_model(args.config, args.ckpt)
    print("模型加载完成。")

    for view in views:
        origin_view_dir = os.path.join(origin_root, view)
        if not os.path.isdir(origin_view_dir):
            print(f"[Warning] 跳过不存在的目录: {origin_view_dir}")
            continue

        # 加载掩码
        mask_path = os.path.join(mask_root, f'cam_hy_n5_avm_{view}.png')
        if not os.path.isfile(mask_path):
            print(f"[Warning] 找不到掩码: {mask_path}，跳过视角 {view}")
            continue

        # 加载鱼眼 K/D
        K_fisheye, D_fisheye, calib_size = load_fisheye_calib(args.calib, view=view)

        img_files = sorted([f for f in os.listdir(origin_view_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"\n[{view}] 发现 {len(img_files)} 张鱼眼图")

        if not img_files:
            continue

        # 用第一张图估计尺寸、缩放 K、计算新针孔 K 和 remap 映射
        img0 = cv2.imread(os.path.join(origin_view_dir, img_files[0]))
        if img0 is None:
            print(f"[Warning] 无法读取首张图像，跳过视角 {view}")
            continue
        H, W = img0.shape[:2]
        K_fisheye_actual = scale_K_to_image(K_fisheye, calib_size, (H, W))

        mask = load_mask(mask_path, (H, W))

        _, new_K, map1, map2 = undistort_fisheye_image(
            img0, K_fisheye_actual, D_fisheye, balance=args.balance
        )

        # 保存新针孔内参
        newK_view_dir = os.path.join(newK_root, view)
        os.makedirs(newK_view_dir, exist_ok=True)
        np.save(os.path.join(newK_view_dir, 'new_K.npy'), new_K.astype(np.float32))
        print(f"[{view}] 新针孔内参 K:\n{new_K}")

        # 创建各输出目录
        undistort_view_dir = os.path.join(undistort_root, view)
        cloud_view_dir = os.path.join(cloud_root, view)
        depth_view_dir = os.path.join(depth_root, view)
        for d in [undistort_view_dir, cloud_view_dir, depth_view_dir]:
            os.makedirs(d, exist_ok=True)

        for img_file in img_files:
            img_path = os.path.join(origin_view_dir, img_file)
            base_name = os.path.splitext(img_file)[0]

            fisheye_img = cv2.imread(img_path)
            if fisheye_img is None:
                print(f"[Warning] 无法读取图像: {img_path}，跳过")
                continue

            # 如果该图尺寸与首张不同，重新 resize 掩码并重新计算映射
            if fisheye_img.shape[:2] != (H, W):
                cur_H, cur_W = fisheye_img.shape[:2]
                cur_mask = cv2.resize(mask, (cur_W, cur_H), interpolation=cv2.INTER_NEAREST)
                K_fisheye_cur = scale_K_to_image(K_fisheye, calib_size, (cur_H, cur_W))
                undistorted, cur_new_K, cur_map1, cur_map2 = undistort_fisheye_image(
                    fisheye_img, K_fisheye_cur, D_fisheye, balance=args.balance
                )
                cur_K = cur_new_K
                warped_mask = warp_mask_to_undistorted(cur_mask, cur_map1, cur_map2)
            else:
                undistorted = cv2.remap(fisheye_img, map1, map2, cv2.INTER_LINEAR)
                cur_K = new_K
                warped_mask = warp_mask_to_undistorted(mask, map1, map2)

            # 保存去畸变图
            undist_path = os.path.join(undistort_view_dir, img_file)
            cv2.imwrite(undist_path, undistorted)

            # 对去畸变图做 Metric3D 推理
            info = preprocess_image(undist_path, args.input_size)
            img_rgb = info['img_rgb']
            img_tensor = info['img_tensor']
            ori_h, ori_w = info['ori_h'], info['ori_w']
            scale = info['scale']

            K_scaled = cur_K.copy()
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

            # 保存深度图（未加掩码）
            save_depth_outputs(pred_depth_np, img_rgb, depth_view_dir, base_name, args.max_depth)

            # 构建额外掩码：warped mask + 上下比例
            extra_mask = warped_mask.copy()
            if args.ignore_top_ratio > 0:
                top_rows = int(ori_h * args.ignore_top_ratio)
                extra_mask[:top_rows, :] = False
            if args.ignore_bottom_ratio > 0:
                bottom_rows = int(ori_h * args.ignore_bottom_ratio)
                extra_mask[-bottom_rows:, :] = False

            # 保存点云（使用 warp 后的掩码过滤）
            save_pinhole_pcd(pred_depth_np, cur_K, img_rgb, cloud_view_dir, base_name,
                             args.max_depth, extra_mask=extra_mask)

            print(f"[{view}] 完成: {img_file}")

    print("\n批量推理完成。")


def main():
    parser = argparse.ArgumentParser(description='Metric3D 去畸变/针孔图像推理')
    # 单图模式
    parser.add_argument('--img', default=None, help='输入图像路径')
    parser.add_argument('--K', default=None, help='去畸变后的针孔内参：.npy 文件路径，或逗号分隔的 9 个数')
    parser.add_argument('--out-dir', default='./output_infer', help='单图模式输出目录')

    # 批量模式
    parser.add_argument('--dataset-root', default=None,
                        help='批量模式：数据集根目录，会自动推导 undistort/newK/cloud 路径')
    parser.add_argument('--undistort-root', default=None,
                        help='批量模式：去畸变图根目录（默认 dataset-root/undistort）')
    parser.add_argument('--newK-root', default=None,
                        help='批量模式：新内参根目录（默认 dataset-root/new_K）')
    parser.add_argument('--cloud-root', default=None,
                        help='批量模式：点云输出根目录（默认 dataset-root/cloud）')
    parser.add_argument('--depth-root', default=None,
                        help='批量模式：深度图输出根目录（默认 dataset-root/depth）')
    parser.add_argument('--origin-root', default=None,
                        help='批量模式：原始鱼眼图根目录（默认 dataset-root/origin）。提供后会自动执行 mask -> 去畸变 -> 推理')
    parser.add_argument('--mask-root', default=None,
                        help='批量模式：静态掩码根目录（默认 dataset-root/mask），文件名为 cam_hy_n5_avm_{view}.png')
    parser.add_argument('--calib', default=None,
                        help='鱼眼 calibration.yml 路径（需要 --origin-root）')
    parser.add_argument('--balance', type=float, default=0.0,
                        help='去畸变 balance 参数')
    parser.add_argument('--views', nargs='+', default=['front', 'back', 'left', 'right'],
                        help='要处理的视角列表')
    parser.add_argument('--ignore-top-ratio', type=float, default=0.0,
                        help='点云构建时忽略图像上方多少比例区域（0.0~1.0），用于去除天空')
    parser.add_argument('--ignore-bottom-ratio', type=float, default=0.0,
                        help='点云构建时忽略图像下方多少比例区域（0.0~1.0），用于去除车体/地面')

    # 公共参数
    parser.add_argument('--config', required=True, help='模型配置文件路径')
    parser.add_argument('--ckpt', required=True, help='模型权重路径')
    parser.add_argument('--input-size', default='616,1064', help='模型输入尺寸 (H,W)，需为 14 的倍数')
    parser.add_argument('--max-depth', type=float, default=200.0, help='深度上限')
    parser.add_argument('--save-pcd', action='store_true', help='是否保存点云')
    args = parser.parse_args()

    # 如果指定了 origin-root 但没有 dataset-root，从 origin-root 推断 dataset-root
    if args.dataset_root is None and args.origin_root is not None:
        args.dataset_root = os.path.dirname(args.origin_root)

    # 如果指定了 dataset-root，自动推导目录
    if args.dataset_root is not None:
        if args.undistort_root is None:
            args.undistort_root = os.path.join(args.dataset_root, 'undistort')
        if args.newK_root is None:
            args.newK_root = os.path.join(args.dataset_root, 'new_K')
        if args.cloud_root is None:
            args.cloud_root = os.path.join(args.dataset_root, 'cloud')
        if args.depth_root is None:
            args.depth_root = os.path.join(args.dataset_root, 'depth')
        # 只有启用 fisheye 模式时才默认使用 origin/mask
        if args.calib is not None:
            if args.origin_root is None:
                args.origin_root = os.path.join(args.dataset_root, 'origin')
            if args.mask_root is None:
                args.mask_root = os.path.join(args.dataset_root, 'mask')
    else:
        # 没有 dataset-root 时使用默认路径
        if args.undistort_root is None:
            args.undistort_root = '/root/ly/Map/Datas/undistort'
        if args.newK_root is None:
            args.newK_root = '/root/ly/Map/Datas/new_K'
        if args.cloud_root is None:
            args.cloud_root = '/root/ly/Map/Metric3D/cloud'
        if args.depth_root is None:
            args.depth_root = '/root/ly/Map/Metric3D/depth'

    if args.img or args.K:
        if not args.img or not args.K:
            raise ValueError("单图模式需要同时提供 --img 和 --K")
        process_single(args)
    else:
        process_batch(args)


if __name__ == '__main__':
    main()
