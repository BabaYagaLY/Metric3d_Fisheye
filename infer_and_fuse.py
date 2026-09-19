"""
================================================================================
 端到端：原始鱼眼图 → 去畸变 → Metric3D 推理 → ego 坐标变换 → 重叠过滤 → 4 路融合
================================================================================

把 infer_undistorted.py 的"推理 + 相机坐标点云"和
visualize_two_clouds_in_ego.py 的"ego 变换 + 重叠过滤 + 合并"串成一条流水线。

数据流（Datas01 为例）：

  Datas01/origin/{view}/{ts}.jpg                  ← 输入
      ↓ mask + 去畸变（infer_undistorted 逻辑）
  Datas01/undistort/{view}/{ts}.jpg
  Datas01/new_K/{view}/new_K.npy
      ↓ Metric3D 推理
  Datas01/depth/{view}/{ts}_depth.png + .npy + _color.png
  Datas01/cloud/{view}/{ts}_pointcloud.ply        （相机坐标系）
      ↓ cam -> ego 外参变换（visualize_two_clouds_in_ego 逻辑）
  Datas01/cloud_ego/{view}/{ts}_pointcloud.ply    （ego 坐标系）
      ↓ 重叠过滤（左右 vs 前后图像投影）
  Datas01/{ts}_fused.ply                          （4 路融合点云）

注：所有底层函数直接 import 自现有模块，逻辑保持不变。
================================================================================
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
import open3d as o3d
from scipy.spatial import cKDTree

# === 来自 infer_common.py ===
from infer_common import (
    load_model,
    preprocess_image,
    recover_depth,
    save_depth_outputs,
    save_pinhole_pcd,
)
# === 来自 infer_fisheye.py ===
from infer_fisheye import load_fisheye_calib, scale_K_to_image
# === 来自 infer_undistorted.py（低层工具函数）===
from infer_undistorted import (
    load_mask,
    undistort_fisheye_image,
    warp_mask_to_undistorted,
)
# === 来自 visualize_two_clouds_in_ego.py（坐标变换 + 重叠过滤 + 合并）===
from visualize_two_clouds_in_ego import (
    CAMERA_SENSOR_IDS,
    cam_to_ego,
    load_extrinsic_from_calibration,
    load_intrinsics_from_calibration,
    project_to_camera_image,
    is_covered_by_image_projection,
    save_merged_point_cloud,
)


# =====================================================================
# 推理后端封装：把 PyTorch 和 TensorRT 统一到一个接口后面
# =====================================================================
# 这样 step1 的代码不用关心用的是哪个后端，统一调用 backend['infer'](img_path, K)。
# 切换后端只需改 Config.backend 一行。
# =====================================================================
from types import SimpleNamespace


def load_backend(args):
    """根据 Config.backend 加载 PyTorch 模型或 TRT engine。

    Returns:
        backend: dict，包含
            - 'cfg': 配置对象（TRT 时是带 canonical_focal 的 SimpleNamespace）
            - 'infer': 函数 (img_path, K, input_size, max_depth) -> (pred_depth_np, img_rgb, info)
    """
    if args.backend == 'trt':
        # === TensorRT 后端 ===
        # 把 onnx/ 目录加到 sys.path 以便 import TrtRunner
        onnx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onnx')
        if onnx_dir not in sys.path:
            sys.path.insert(0, onnx_dir)
        from infer_trt import TrtRunner

        print(f"[Backend] 加载 TensorRT engine: {args.trt_engine}")
        runner = TrtRunner(args.trt_engine)

        # TRT 没有真正的 cfg，构造一个只含 canonical_focal 的假 cfg
        # 让 recover_depth 内部的 cfg.data_basic.canonical_space.focal_length 能正常访问
        fake_cfg = SimpleNamespace(
            data_basic=SimpleNamespace(
                canonical_space=SimpleNamespace(
                    focal_length=args.canonical_focal
                )
            )
        )

        def trt_infer(img_path, K, input_size, max_depth):
            """TRT 推理：输入 raw RGB（归一化烤在 ONNX 里），输出深度图。"""
            info = preprocess_image(img_path, input_size)
            img_rgb = info['img_rgb']
            ori_h, ori_w = info['ori_h'], info['ori_w']
            scale = info['scale']

            # K 按模型输入尺寸缩放（和 PyTorch 路径完全一致）
            K_scaled = K.copy()
            K_scaled[0, 0] *= scale
            K_scaled[1, 1] *= scale
            K_scaled[0, 2] *= scale
            K_scaled[1, 2] *= scale

            # TRT 接受未归一化的 [1,3,H,W] float32 RGB（0~255）
            # 因为 ONNX 导出时把归一化层 (x-mean)/std 放进了图里
            arr = np.ascontiguousarray(
                info['img_pad'].transpose(2, 0, 1)[None], dtype=np.float32
            )
            pred_depth = runner.infer(arr)        # numpy [1,1,H,W]
            pred_depth = torch.from_numpy(pred_depth)  # 转成 torch 以便复用 recover_depth

            pred_depth_np = recover_depth(
                pred_depth, K_scaled, fake_cfg,
                ori_h, ori_w,
                info['pad_h_half'], info['pad_w_half'],
                info['new_h'], info['new_w'],
                max_depth
            )
            return pred_depth_np, img_rgb, info

        print(f"[Backend] TensorRT 就绪（FP16 engine）")
        return {'cfg': fake_cfg, 'infer': trt_infer}

    else:
        # === PyTorch 后端（原版逻辑）===
        cfg, model = load_model(args.config, args.ckpt)
        print(f"[Backend] PyTorch 模型就绪")

        def torch_infer(img_path, K, input_size, max_depth):
            """PyTorch 推理：输入归一化后的 tensor，输出深度图。"""
            info = preprocess_image(img_path, input_size)
            img_rgb = info['img_rgb']
            ori_h, ori_w = info['ori_h'], info['ori_w']
            scale = info['scale']

            K_scaled = K.copy()
            K_scaled[0, 0] *= scale
            K_scaled[1, 1] *= scale
            K_scaled[0, 2] *= scale
            K_scaled[1, 2] *= scale

            pred_depth, _, _ = model.inference({'input': info['img_tensor']})
            pred_depth_np = recover_depth(
                pred_depth, K_scaled, cfg,
                ori_h, ori_w,
                info['pad_h_half'], info['pad_w_half'],
                info['new_h'], info['new_w'],
                max_depth
            )
            return pred_depth_np, img_rgb, info

        return {'cfg': cfg, 'infer': torch_infer}


# =====================================================================
# Step 1：4 视角各自的 mask → 去畸变 → 推理 → 深度图 + 相机坐标点云
# =====================================================================
# 逻辑与 infer_undistorted.process_batch_from_fisheye 完全一致，只是抽出来。
# =====================================================================
def step1_infer_per_view(args):
    """对每个视角做完整的推理流程，输出深度图和相机坐标系下的点云。"""
    origin_root = args.origin_root
    mask_root = args.mask_root
    undistort_root = args.undistort_root
    newK_root = args.newK_root
    cloud_root = args.cloud_root
    depth_root = args.depth_root
    views = args.views

    # 创建所有输出目录
    for root in [undistort_root, newK_root, cloud_root, depth_root]:
        os.makedirs(root, exist_ok=True)

    # 加载推理后端（PyTorch 或 TRT，根据 Config.backend）
    backend = load_backend(args)

    # 外层循环：视角（和 infer_undistorted.py 保持一致）
    for view in views:
        origin_view_dir = os.path.join(origin_root, view)
        if not os.path.isdir(origin_view_dir):
            print(f"[Warning] 跳过不存在的目录: {origin_view_dir}")
            continue

        # 静态掩码
        mask_path = os.path.join(mask_root, f'cam_hy_n5_avm_{view}.png')
        if not os.path.isfile(mask_path):
            print(f"[Warning] 找不到掩码: {mask_path}，跳过视角 {view}")
            continue

        # 鱼眼 K/D
        K_fisheye, D_fisheye, calib_size = load_fisheye_calib(args.calib, view=view)

        # 枚举该视角所有时间戳
        img_files = sorted([f for f in os.listdir(origin_view_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"\n[{view}] 发现 {len(img_files)} 张鱼眼图")
        if not img_files:
            continue

        # 用首张图估计尺寸、缩放 K、计算新针孔 K 和 remap 映射
        img0 = cv2.imread(os.path.join(origin_view_dir, img_files[0]))
        if img0 is None:
            print(f"[Warning] 无法读取首张图像，跳过视角 {view}")
            continue
        H, W = img0.shape[:2]
        K_fisheye_actual = scale_K_to_image(K_fisheye, calib_size, (H, W))
        mask = load_mask(mask_path, (H, W))

        # 计算去畸变映射和新针孔内参（一次，复用）
        _, new_K, map1, map2 = undistort_fisheye_image(
            img0, K_fisheye_actual, D_fisheye, balance=args.balance
        )

        # 保存新针孔内参
        newK_view_dir = os.path.join(newK_root, view)
        os.makedirs(newK_view_dir, exist_ok=True)
        np.save(os.path.join(newK_view_dir, 'new_K.npy'), new_K.astype(np.float32))
        print(f"[{view}] 新针孔内参 K:\n{new_K}")

        # 创建该视角的输出目录
        undistort_view_dir = os.path.join(undistort_root, view)
        cloud_view_dir = os.path.join(cloud_root, view)
        depth_view_dir = os.path.join(depth_root, view)
        for d in [undistort_view_dir, cloud_view_dir, depth_view_dir]:
            os.makedirs(d, exist_ok=True)

        # 内层循环：时间戳
        for img_file in img_files:
            img_path = os.path.join(origin_view_dir, img_file)
            base_name = os.path.splitext(img_file)[0]

            fisheye_img = cv2.imread(img_path)
            if fisheye_img is None:
                print(f"[Warning] 无法读取图像: {img_path}，跳过")
                continue

            # 尺寸不一致时重新计算映射（边界情况）
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

            # 推理（后端可能是 PyTorch 或 TRT，统一接口）
            pred_depth_np, img_rgb, info = backend['infer'](
                undist_path, cur_K, args.input_size, args.max_depth
            )
            ori_h, ori_w = info['ori_h'], info['ori_w']

            # 保存深度图（不加 mask，保留全局信息）
            save_depth_outputs(pred_depth_np, img_rgb, depth_view_dir, base_name, args.max_depth)

            # 构建 extra_mask：warped mask + 可选的上下比例裁切
            extra_mask = warped_mask.copy()
            if args.ignore_top_ratio > 0:
                top_rows = int(ori_h * args.ignore_top_ratio)
                extra_mask[:top_rows, :] = False
            if args.ignore_bottom_ratio > 0:
                bottom_rows = int(ori_h * args.ignore_bottom_ratio)
                extra_mask[-bottom_rows:, :] = False

            # 保存相机坐标系点云（用 warped mask 过滤无效像素）
            save_pinhole_pcd(pred_depth_np, cur_K, img_rgb, cloud_view_dir, base_name,
                             args.max_depth, extra_mask=extra_mask)

            print(f"[{view}] 完成: {img_file}")


# =====================================================================
# Step 2：把相机坐标点云变换到 ego 坐标系
# =====================================================================
# 直接调用 visualize_two_clouds_in_ego.cam_to_ego。
# 每视角外参 [tx,ty,tz,qx,qy,qz,qw] 从 calibration.yml 加载。
# =====================================================================
def step2_transform_to_ego(cloud_root, cloud_ego_root, calib_path, views,
                           max_range=200.0):
    """读取每视角的相机坐标系点云，变换到 ego 并保存。

    Returns:
        ego_clouds: dict[view] = (pts_ego, cols, extrinsic)
                    供 step3 使用，避免重复读盘。
    """
    ego_clouds = {}
    for view in views:
        cam_dir = Path(cloud_root) / view
        ego_dir = Path(cloud_ego_root) / view
        ego_dir.mkdir(parents=True, exist_ok=True)

        if not cam_dir.exists():
            print(f"[Warning] 找不到相机点云目录: {cam_dir}")
            continue

        # 加载该视角外参
        extrinsic = load_extrinsic_from_calibration(view, calib_path)
        print(f"[{view}] 外参: {extrinsic}")

        # 处理该视角每个时间戳
        for cam_file in sorted(cam_dir.glob('*_pointcloud.ply')):
            pcd = o3d.io.read_point_cloud(str(cam_file))
            if pcd.is_empty():
                continue
            pts_cam = np.asarray(pcd.points, dtype=np.float64)
            cols = np.asarray(pcd.colors) if pcd.has_colors() else None

            # 相机系 -> ego 系
            pts_ego = cam_to_ego(pts_cam, extrinsic)

            # 过滤极端离群点（与 visualize_two_clouds_in_ego 一致）
            if max_range and max_range > 0:
                valid = np.all(np.abs(pts_ego) <= max_range, axis=1)
                pts_ego = pts_ego[valid]
                if cols is not None:
                    cols = cols[valid]

            # 保存 ego 坐标系点云
            ego_pcd = o3d.geometry.PointCloud()
            ego_pcd.points = o3d.utility.Vector3dVector(pts_ego)
            if cols is not None and len(cols) == len(pts_ego):
                ego_pcd.colors = o3d.utility.Vector3dVector(cols)
            ego_path = ego_dir / cam_file.name
            o3d.io.write_point_cloud(str(ego_path), ego_pcd)

            # 记录到字典（按时间戳覆盖；外部按时间戳分组时再用）
            ego_clouds.setdefault(view, []).append(
                (cam_file.stem.replace('_pointcloud', ''), pts_ego, cols, extrinsic)
            )
            print(f"[{view}] ego 变换完成: {cam_file.name} ({len(pts_ego)} 点)")

    return ego_clouds


# =====================================================================
# Step 3：按时间戳做重叠过滤 + 4 路融合
# =====================================================================
# 逻辑完全照搬 visualize_two_clouds_in_ego.py 的过滤段：
#   1. 用 front/back 自己的点投影到自己的图像，建立"占用像素 KDTree"
#   2. 把 left/right 的点投影到 front/back 图像，若落在 tolerance 内则移除
#   3. 把过滤后的四路点云合并保存
# =====================================================================
def step3_filter_and_fuse_one_timestamp(ts, views, pts_per_view, cols_per_view,
                                        extrinsic_per_view, calib_path,
                                        out_path, pixel_tolerance=5.0,
                                        voxel_size=0.0,
                                        fov_overlap_filter=True):
    """对单个时间戳的 4 路点云做重叠过滤 + 合并保存。

    Args:
        ts: 时间戳字符串
        views: 视角列表，如 ['front','back','left','right']
        pts_per_view: dict[view] = pts_ego (np.ndarray)
        cols_per_view: dict[view] = cols (np.ndarray | None)
        extrinsic_per_view: dict[view] = extrinsic list[7]
        calib_path: calibration.yml
        out_path: 输出 .ply 路径
        pixel_tolerance: 投影覆盖判定的像素容差
        voxel_size: 合并时体素下采样分辨率（0 = 不下采样）
        fov_overlap_filter: 是否启用图像投影重叠过滤
    """
    # 组装 cam_data：和 visualize_two_clouds_in_ego 的格式完全一致
    # 元素格式: (pts_in_ego, cols, path, suffix, extrinsic, fov)
    cam_data = []
    for view in views:
        if view not in pts_per_view:
            continue
        pts = pts_per_view[view]
        cols = cols_per_view.get(view)
        ext = extrinsic_per_view[view]
        cam_data.append([pts, cols, None, view, ext, None])

    # === 重叠过滤（照搬 visualize_two_clouds_in_ego.py 的逻辑）===
    if fov_overlap_filter:
        # 1. 用 front/back 自己的点投影到自己的图像，建立占用像素 KDTree
        ref_configs = []  # list of (tree, extrinsic, intrinsics)
        for pts, cols, _path, suffix, extrinsic, _fov in cam_data:
            if suffix not in ('front', 'back'):
                continue
            intrinsics = load_intrinsics_from_calibration(suffix, calib_path)
            pixels, valid = project_to_camera_image(pts, extrinsic, *intrinsics)
            valid_pixels = pixels[valid]
            tree = cKDTree(valid_pixels) if len(valid_pixels) > 0 else None
            ref_configs.append((tree, extrinsic, intrinsics))
            print(f"  [图像投影] {suffix}/{ts}: 建立 {len(valid_pixels)} 像素占用图")

        # 2. 把 left/right 的点投影到 front/back 图像，被覆盖的移除
        if ref_configs:
            for i, (pts, cols, _path, suffix, extrinsic, _fov) in enumerate(cam_data):
                if suffix not in ('left', 'right'):
                    continue
                in_front_or_back = np.zeros(len(pts), dtype=bool)
                for tree, ref_ext, intrinsics in ref_configs:
                    covered = is_covered_by_image_projection(
                        pts, tree, ref_ext, intrinsics, pixel_tolerance
                    )
                    in_front_or_back |= covered

                keep = ~in_front_or_back
                n_removed = len(pts) - int(keep.sum())
                if n_removed > 0:
                    print(f"  [重叠过滤] {suffix}/{ts}: 移除 {n_removed} 点（与前后重叠），"
                          f"保留 {int(keep.sum())} 点")
                cam_data[i][0] = pts[keep]
                cam_data[i][1] = cols[keep] if cols is not None else None

    # === 合并保存（直接调用 visualize_two_clouds_in_ego.save_merged_point_cloud）===
    # save_merged_point_cloud 期望 cam_data 是 list of tuple，这里转一下
    cam_data_tuples = [tuple(d) for d in cam_data]
    save_merged_point_cloud(cam_data_tuples, Path(out_path), voxel_size)
    print(f"[融合] {ts}: 已保存到 {out_path}")


def step3_filter_and_fuse_all(ego_clouds, views, calib_path, dataset_root,
                              pixel_tolerance=5.0, voxel_size=0.0,
                              fov_overlap_filter=True,
                              cloud_root=None, cloud_ego_root=None,
                              cleanup_intermediate=False):
    """对所有时间戳执行 step3。

    Args:
        ego_clouds: step2 的返回值，dict[view] = list of (ts, pts, cols, extrinsic)
        views: 视角顺序
        calib_path, dataset_root: 标定 + 输出根目录
        pixel_tolerance, voxel_size, fov_overlap_filter: 同 step3
        cloud_root, cloud_ego_root: 中间结果目录，cleanup_intermediate=True 时会删
        cleanup_intermediate: 是否在融合后删除该时间戳的 cloud/cloud_ego 文件
    """
    # 把 ego_clouds 重新组织成按时间戳分组
    timestamps = sorted({ts for view in views if view in ego_clouds
                         for (ts, _, _, _) in ego_clouds[view]})
    if not timestamps:
        print("[Warning] 没有可融合的时间戳")
        return

    print(f"\n=== 开始融合 {len(timestamps)} 个时间戳 ===")
    for ts in timestamps:
        # 收集该时间戳下 4 视角的 ego 点云
        pts_per_view = {}
        cols_per_view = {}
        extrinsic_per_view = {}
        for view in views:
            if view not in ego_clouds:
                continue
            for (t, pts, cols, ext) in ego_clouds[view]:
                if t == ts:
                    pts_per_view[view] = pts
                    cols_per_view[view] = cols
                    extrinsic_per_view[view] = ext
                    break

        if len(pts_per_view) < 2:
            print(f"[Warning] 时间戳 {ts} 只有 {len(pts_per_view)} 个视角，跳过")
            continue

        # 输出路径：Datas01/{ts}_fused.ply
        out_path = Path(dataset_root) / f'{ts}_fused.ply'
        step3_filter_and_fuse_one_timestamp(
            ts, views, pts_per_view, cols_per_view, extrinsic_per_view,
            calib_path, out_path,
            pixel_tolerance=pixel_tolerance,
            voxel_size=voxel_size,
            fov_overlap_filter=fov_overlap_filter,
        )

        # === 融合成功后清理该时间戳的中间点云（可选）===
        # 删除 cloud/{view}/{ts}_pointcloud.ply 和 cloud_ego/{view}/{ts}_pointcloud.ply
        # depth/{view}/、undistort/{view}/、new_K/{view}/ 都保留（深度图有用）
        if cleanup_intermediate:
            n_deleted = 0
            for view in views:
                for root in [cloud_root, cloud_ego_root]:
                    if root is None:
                        continue
                    f = Path(root) / view / f'{ts}_pointcloud.ply'
                    if f.exists():
                        f.unlink()
                        n_deleted += 1
            if n_deleted > 0:
                print(f"[清理] 删除 {ts} 的中间点云 {n_deleted} 个文件"
                      f"（cloud/ + cloud_ego/）")


# =====================================================================
# 主入口
# =====================================================================
# =====================================================================
# 配置区：在这里修改所有路径和参数
# =====================================================================
class Config:
    # --- 数据集 ---
    dataset_root = '/root/ly/Map/Datas02'
    # calib 自动推导为 dataset_root + 'calibration.yml'
    views        = ['front', 'back', 'left', 'right']

    # --- Metric3D 模型 ---
    config      = 'training/mono/configs/RAFTDecoder/vit.raft5.large.py'
    ckpt        = '/root/ly/Map/metric_depth_vit_large_800k.pth'
    input_size  = '616,1064'    # H,W，必须是 14 的倍数
    max_depth   = 200.0         # 深度上限（米）

    # --- 推理后端 ---
    # 'pytorch' = 用原始 PyTorch 模型（~2000 ms/张）
    # 'trt'     = 用 TensorRT FP16 engine（~300 ms/张，7× 加速）
    backend        = 'trt'
    trt_engine     = '/root/ly/Map/Metric3D/onnx/metric3d_vit_large_fp16.engine'
    canonical_focal = 1000.0    # Metric3D 默认值，用于把相对深度还原成米

    # --- 去畸变 ---
    balance             = 0.0   # cv2.fisheye estimateNewCameraMatrixForUndistortRectify 的 balance
    ignore_top_ratio    = 0.3   # 点云忽略图像上方比例（去天空），0~1
    ignore_bottom_ratio = 0.0   # 点云忽略图像下方比例（去车体），0~1

    # --- ego 变换 + 重叠过滤 + 合并 ---
    max_range           = 200.0    # 过滤 |坐标| 超过此值的离群点（米）
    pixel_tolerance     = 5.0      # 重叠过滤的像素容差
    voxel_size          = 0.0      # 合并时体素下采样分辨率（米），0 = 不下采样
    fov_overlap_filter  = True     # 是否启用图像投影重叠过滤（左右 vs 前后）

    # --- 流程控制（调试用）---
    skip_infer = False   # True = 跳过 Step1，直接用已有 cloud/
    skip_ego   = False   # True = 跳过 Step2，直接用已有 cloud_ego/
    only_fuse  = False   # True = 只跑 Step3（cloud_ego/ 必须已存在）

    # --- 清理中间结果 ---
    # True = 每个时间戳融合完，删除 cloud/{view}/{ts}_pointcloud.ply 和
    #        cloud_ego/{view}/{ts}_pointcloud.ply（深度图、去畸变图、new_K 保留）
    cleanup_intermediate = False
# =====================================================================


def main():
    args = Config()

    # 推导所有子目录路径
    args.calib          = os.path.join(args.dataset_root, 'calibration.yml')
    args.origin_root    = os.path.join(args.dataset_root, 'origin')
    args.mask_root      = os.path.join(args.dataset_root, 'mask')
    args.undistort_root = os.path.join(args.dataset_root, 'undistort')
    args.newK_root      = os.path.join(args.dataset_root, 'new_K')
    args.cloud_root     = os.path.join(args.dataset_root, 'cloud')
    args.depth_root     = os.path.join(args.dataset_root, 'depth')
    cloud_ego_root      = os.path.join(args.dataset_root, 'cloud_ego')

    print("=" * 60)
    print("数据集:", args.dataset_root)
    print("标定文件:", args.calib)
    print("视角:", args.views)
    print("=" * 60)

    # ============= Step 1：4 视角推理 =============
    if not args.skip_infer and not args.only_fuse:
        print("\n" + "=" * 60)
        print("Step 1: 4 视角推理（mask → 去畸变 → Metric3D → 深度图 + 相机点云）")
        print("=" * 60)
        step1_infer_per_view(args)

    # ============= Step 2：ego 坐标变换 =============
    ego_clouds = {}
    if not args.skip_ego and not args.only_fuse:
        print("\n" + "=" * 60)
        print("Step 2: 相机坐标 → ego 坐标变换")
        print("=" * 60)
        ego_clouds = step2_transform_to_ego(
            args.cloud_root, cloud_ego_root, args.calib, args.views,
            max_range=args.max_range
        )
    else:
        # 跳过变换时，直接从 cloud_ego/ 读盘重建 ego_clouds
        print("\n[Step 2] 跳过，从已有 cloud_ego/ 读取")
        for view in args.views:
            ego_dir = Path(cloud_ego_root) / view
            if not ego_dir.exists():
                continue
            extrinsic = load_extrinsic_from_calibration(view, args.calib)
            for f in sorted(ego_dir.glob('*_pointcloud.ply')):
                pcd = o3d.io.read_point_cloud(str(f))
                if pcd.is_empty():
                    continue
                ts = f.stem.replace('_pointcloud', '')
                ego_clouds.setdefault(view, []).append(
                    (ts, np.asarray(pcd.points), np.asarray(pcd.colors), extrinsic)
                )

    # ============= Step 3：重叠过滤 + 4 路融合 =============
    print("\n" + "=" * 60)
    print("Step 3: 重叠过滤 + 4 路融合")
    print("=" * 60)
    step3_filter_and_fuse_all(
        ego_clouds, args.views, args.calib, args.dataset_root,
        pixel_tolerance=args.pixel_tolerance,
        voxel_size=args.voxel_size,
        fov_overlap_filter=args.fov_overlap_filter,
        cloud_root=args.cloud_root,
        cloud_ego_root=cloud_ego_root,
        cleanup_intermediate=args.cleanup_intermediate,
    )

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"融合点云: {args.dataset_root}/{{timestamp}}_fused.ply")
    print("=" * 60)


if __name__ == '__main__':
    main()
