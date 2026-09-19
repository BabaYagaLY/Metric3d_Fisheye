"""
可视化合并后的相机点云与雷达(LiDAR)真值点云的叠加效果。

合并点云（visualize_two_clouds_in_ego.py --save_merged 生成，已在 ego 坐标系）
与雷达点云（.pcd / .las / .laz）统一到 ego 坐标系后叠加显示，
支持鼠标旋转/缩放/平移和 WASD+QE 自由移动视角。

用法示例:
  python visualize_merged_vs_lidar.py ^
      --merged E:/lidar-depth/cloud02/1767850446919_merged.ply ^
      --lidar  E:/lidar-depth/cloud02/1767850446919.pcd

  python visualize_merged_vs_lidar.py ^
      --merged E:/lidar-depth/cloud02/1767850446919_merged.ply ^
      --lidar  E:/lidar-depth/cloud02/1767850446919.las ^
      --lidar_color 0.0 0.8 1.0 --point_size 2.0
"""

from __future__ import annotations

import argparse

import numpy as np
import open3d as o3d

# 注意：visualize_two_clouds_in_ego 在导入时已把 stdout 包装为 UTF-8，
# 这里不要重复包装（否则旧 wrapper 被回收会关闭底层缓冲）。
from visualize_two_clouds_in_ego import (
    align_pcd_to_ego,
    build_key_callbacks,
    create_pcd,
    load_point_cloud,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="叠加可视化合并相机点云与雷达真值点云，支持 WASD 移动视角"
    )
    parser.add_argument(
        "--merged",
        required=True,
        help="合并后的相机点云文件（.ply/.pcd，已在 ego 坐标系）",
    )
    parser.add_argument(
        "--lidar",
        required=True,
        help="雷达真值点云文件（.pcd/.las/.laz，原始 PCD 坐标系会自动对齐到 ego）",
    )
    parser.add_argument(
        "--align_pcd",
        action="store_true",
        default=True,
        help="将雷达点云从 X右/Y前/Z上 对齐到标定的 X前/Y左/Z上（默认开启）",
    )
    parser.add_argument(
        "--no_align_pcd",
        action="store_false",
        dest="align_pcd",
        help="禁用雷达点云坐标系对齐（雷达文件已是 ego 坐标系时使用）",
    )
    parser.add_argument(
        "--lidar_color",
        nargs=3,
        type=float,
        default=[0.0, 0.8, 1.0],
        help="雷达点云显示颜色 (R G B)，默认亮青；若文件自带颜色则优先使用自带颜色",
    )
    parser.add_argument(
        "--merged_color",
        nargs=3,
        type=float,
        default=None,
        help="强制合并点云使用统一颜色 (R G B)；默认保留合并文件自带颜色",
    )
    parser.add_argument(
        "--frame_size",
        type=float,
        default=2.0,
        help="ego 坐标轴长度（米），默认 2",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.5,
        help="点大小，默认 1.5",
    )
    parser.add_argument(
        "--bg_color",
        nargs=3,
        type=float,
        default=[0.9, 0.9, 0.9],
        help="背景颜色 (R G B)，默认浅灰",
    )
    parser.add_argument(
        "--move_step",
        type=float,
        default=0.3,
        help="WASD 每次移动的距离（米），默认 0.3",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载合并后的相机点云（已在 ego 坐标系）
    merged_pts, merged_cols = load_point_cloud(args.merged)
    if args.merged_color is not None:
        pcd_merged = create_pcd(merged_pts, uniform_color=args.merged_color)
        print(f"合并点云: {args.merged}\n  点数 {len(merged_pts)}，使用统一颜色 {args.merged_color}")
    else:
        pcd_merged = create_pcd(merged_pts, colors=merged_cols)
        has_cols = merged_cols is not None and len(merged_cols) == len(merged_pts)
        print(f"合并点云: {args.merged}\n  点数 {len(merged_pts)}，{'使用自带颜色' if has_cols else '无颜色'}")

    # 加载雷达点云并对齐到 ego 坐标系
    lidar_pts, lidar_cols = load_point_cloud(args.lidar)
    if args.align_pcd:
        lidar_pts = align_pcd_to_ego(lidar_pts)
        print("已将雷达点云从 PCD 坐标系对齐到 ego 坐标系")
    has_lidar_cols = lidar_cols is not None and len(lidar_cols) == len(lidar_pts)
    pcd_lidar = create_pcd(
        lidar_pts,
        colors=lidar_cols if has_lidar_cols else None,
        uniform_color=None if has_lidar_cols else args.lidar_color,
    )
    print(f"雷达点云: {args.lidar}\n  点数 {len(lidar_pts)}，{'使用自带颜色' if has_lidar_cols else f'使用颜色 {args.lidar_color}'}")

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=args.frame_size, origin=[0, 0, 0]
    )

    print("\n交互操作:")
    print("  左键拖动 : 旋转视角")
    print("  滚轮     : 缩放")
    print("  Shift+左键拖动 : 平移")
    print("  W/S/A/D : 前后左右移动视角")
    print("  Q/E     : 上升/下降视角")
    print("  1       : 显示/隐藏合并相机点云")
    print("  2       : 显示/隐藏雷达点云")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Merged camera cloud vs LiDAR cloud (EGO)",
        width=1280,
        height=720,
    )
    vis.add_geometry(pcd_merged)
    vis.add_geometry(pcd_lidar)
    vis.add_geometry(frame)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray(args.bg_color, dtype=np.float64)
    opt.light_on = True

    vis.poll_events()
    vis.update_renderer()
    build_key_callbacks(vis, args.move_step)

    # 1/2 切换两路点云显隐
    state = {"merged": True, "lidar": True}

    def toggle_merged(v):
        if state["merged"]:
            v.remove_geometry(pcd_merged)
        else:
            v.add_geometry(pcd_merged)
        state["merged"] = not state["merged"]
        v.update_renderer()
        print(f"[显示] {'显示' if state['merged'] else '隐藏'} 合并点云")
        return True

    def toggle_lidar(v):
        if state["lidar"]:
            v.remove_geometry(pcd_lidar)
        else:
            v.add_geometry(pcd_lidar)
        state["lidar"] = not state["lidar"]
        v.update_renderer()
        print(f"[显示] {'显示' if state['lidar'] else '隐藏'} 雷达点云")
        return True

    vis.register_key_callback(ord("1"), toggle_merged)
    vis.register_key_callback(ord("2"), toggle_lidar)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
