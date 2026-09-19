"""
将相机坐标系下的点云与 ego 坐标系下的雷达点云统一变换到 ego 坐标系，
并使用 Open3D 交互式可视化（鼠标拖动旋转、滚轮缩放、Shift+左键平移、WASD 自由移动视角）。

用法示例:
  python visualize_two_clouds_in_ego.py \\
      --cam_pcd front001_pointcloud.ply \\
      --ego_pcd 1767850432718.pcd \\
      --point_size 1.0 --bg_color 0.9 0.9 0.9

只输入 cloud 路径即可自动显示（目录下有唯一时间戳时）:
  python visualize_two_clouds_in_ego.py --cloud_dir E:/lidar-depth/cloud

同时显示四个相机视角（按时间戳自动加载）:
  python visualize_two_clouds_in_ego.py \\
      --cloud_dir E:/lidar-depth/cloud \\
      --timestamp 1767850446919 \\
      --point_size 1.0 --bg_color 0.9 0.9 0.9

移除左右视角与前后视角的重叠区域（前后全部保留，左右只保留非重叠部分，基于图像投影判断）:
  python visualize_two_clouds_in_ego.py \\
      --cloud_dir E:/lidar-depth/cloud

禁用图像投影非重叠过滤:
  python visualize_two_clouds_in_ego.py \\
      --cloud_dir E:/lidar-depth/cloud \\
      --no_fov_overlap_filter

调整图像投影容差（像素，默认 5）:
  python visualize_two_clouds_in_ego.py \\
      --cloud_dir E:/lidar-depth/cloud \\
      --pixel_tolerance 10

重叠过滤后把四路点云合并保存为一个文件（默认 PLY，也可用 .pcd）:
  python visualize_two_clouds_in_ego.py \\
      --cloud_dir E:/lidar-depth/cloud \\
      --save_merged E:/lidar-depth/cloud/merged_pointcloud.ply

  不带路径时自动保存到 <cloud_dir>/<时间戳>_merged.ply:
  python visualize_two_clouds_in_ego.py \\
      --cloud_dir E:/lidar-depth/cloud \\
      --save_merged

  合并时做体素下采样平滑接缝（单位米，默认 0 表示不下采样）:
  python visualize_two_clouds_in_ego.py \\
      --cloud_dir E:/lidar-depth/cloud \\
      --save_merged --merged_voxel_size 0.05

--extrinsic 会根据文件名中的 front/back/left/right 自动从 calibration.yml 加载；
也可以显式传入 --extrinsic tx ty tz qx qy qz qw 进行覆盖（单相机时可用，多相机时建议依赖自动加载）。
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


DEFAULT_CALIBRATION = r"E:\lidar-depth\calibration.yml"
CAMERA_SENSOR_IDS = {
    "front": "cam_hy_n5_avm_front",
    "back": "cam_hy_n5_avm_back",
    "left": "cam_hy_n5_avm_left",
    "right": "cam_hy_n5_avm_right",
}
DEFAULT_CAMERA_COLORS = {
    "front": [1.0, 0.2, 0.0],   # 红
    "back":  [0.0, 1.0, 0.2],   # 绿
    "left":  [0.0, 0.4, 1.0],   # 蓝
    "right": [1.0, 1.0, 0.0],   # 黄
}


def cam_to_ego(points_cam: np.ndarray, extrinsic: list | np.ndarray) -> np.ndarray:
    """
    把相机坐标系点云转换到 ego 坐标系。

    extrinsic = [tx, ty, tz, qx, qy, qz, qw]
    R: camera -> ego 的旋转
    t: 相机光心在 ego 中的位置
    P_ego = R * P_cam + t
    """
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    t = extrinsic[:3]
    R = Rotation.from_quat(extrinsic[3:]).as_matrix()  # camera -> ego

    pts = np.asarray(points_cam, dtype=np.float64)
    return (R @ pts.T).T + t


def align_pcd_to_ego(points: np.ndarray) -> np.ndarray:
    """
    把 PCD 坐标系对齐到标定使用的 ego 坐标系。

    标定 ego: X 前, Y 左, Z 上；PCD 实测: X 右, Y 前, Z 上。
    变换: P_ego = (pcd_y, -pcd_x, pcd_z)。
    """
    pts = np.asarray(points, dtype=np.float64)
    return np.stack([pts[:, 1], -pts[:, 0], pts[:, 2]], axis=1)


def load_point_cloud(path: str | Path):
    """加载点云，返回 (points, colors)。支持 .ply / .pcd / .las / .laz 等格式。"""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError as e:
            raise RuntimeError(
                f"读取 .las/.laz 文件需要安装 laspy: pip install laspy"
            ) from e
        las = laspy.read(str(path))
        pts = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
        cols = None
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            r = np.asarray(las.red)
            g = np.asarray(las.green)
            b = np.asarray(las.blue)
            max_val = max(r.max(), g.max(), b.max(), 1)
            cols = np.stack([r, g, b], axis=1).astype(np.float64) / max_val
        elif hasattr(las, "intensity"):
            intensity = np.asarray(las.intensity)
            max_val = max(intensity.max(), 1)
            c = intensity / max_val
            cols = np.stack([c, c, c], axis=1)
        return pts, cols

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError(f"点云为空或读取失败: {path}")
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    return pts, cols


def filter_outliers(pts: np.ndarray, cols: np.ndarray | None, max_range: float):
    """移除坐标绝对值超过 max_range 的离群点，返回过滤后的 (pts, cols)。"""
    if max_range is None or max_range <= 0:
        return pts, cols
    valid_mask = np.all(np.abs(pts) <= max_range, axis=1)
    n_removed = len(pts) - int(valid_mask.sum())
    if n_removed > 0:
        print(f"[过滤] 移除 {n_removed} 个离群点（|coord| > {max_range} m），保留 {int(valid_mask.sum())} 点")
    return pts[valid_mask], cols[valid_mask] if cols is not None else None


def load_fov_from_calibration(cam_suffix: str, yaml_path: str | Path) -> float:
    """从 calibration.yml 加载指定相机的 fov_fit（度）。"""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到标定文件: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sensor_id = CAMERA_SENSOR_IDS[cam_suffix]
    fov = cfg["rig"][sensor_id].get("fov_fit")
    if fov is None:
        raise ValueError(f"{sensor_id} 没有 fov_fit 参数")
    return float(fov)


def load_intrinsics_from_calibration(
    cam_suffix: str, yaml_path: str | Path
) -> tuple[float, float, float, float, float, float, float, float, int, int]:
    """从 calibration.yml 加载指定相机的内参。

    返回 (fx, fy, cx, cy, k1, k2, k3, k4, width, height)。
    inv_poly 作为 OpenCV fisheye 前向畸变系数 [k1, k2, k3, k4] 使用。
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到标定文件: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sensor_id = CAMERA_SENSOR_IDS[cam_suffix]
    cam = cfg["rig"][sensor_id]
    fx, fy = cam["focal"]
    cx, cy = cam["pp"]
    width, height = cam["image_size"]
    k = cam.get("inv_poly", [0.0, 0.0, 0.0, 0.0])
    if len(k) < 4:
        k = list(k) + [0.0] * (4 - len(k))
    return (
        float(fx), float(fy), float(cx), float(cy),
        float(k[0]), float(k[1]), float(k[2]), float(k[3]),
        int(width), int(height),
    )


def project_to_camera_image(
    pts_in_ego: np.ndarray,
    extrinsic: list | np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    把 ego 坐标系下的点投影到相机图像平面（OpenCV fisheye 模型）。

    投影公式：
        theta = atan2(sqrt(x^2 + y^2), z)
        r     = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
        u     = r * x / sqrt(x^2 + y^2)
        v     = r * y / sqrt(x^2 + y^2)
        px    = fx * u + cx
        py    = fy * v + cy

    返回 (pixel_coords, valid_mask)，其中 valid_mask 表示点在相机前方（z>0）
    且投影落在图像边界内。
    """
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    t = extrinsic[:3]
    R = Rotation.from_quat(extrinsic[3:]).as_matrix()

    pts = np.asarray(pts_in_ego, dtype=np.float64)
    pts_cam = (R.T @ (pts - t).T).T

    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    z = pts_cam[:, 2]
    r_xy = np.sqrt(x * x + y * y)
    theta = np.arctan2(r_xy, z)

    # OpenCV fisheye 前向畸变多项式
    r = theta * (1.0 + k1 * theta ** 2 + k2 * theta ** 4 + k3 * theta ** 6 + k4 * theta ** 8)

    phi = np.arctan2(y, x)
    u = np.zeros_like(r)
    v = np.zeros_like(r)
    on_axis = r_xy <= 1e-9
    u[~on_axis] = r[~on_axis] * x[~on_axis] / r_xy[~on_axis]
    v[~on_axis] = r[~on_axis] * y[~on_axis] / r_xy[~on_axis]

    px = fx * u + cx
    py = fy * v + cy

    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    valid = (z > 0) & in_bounds
    pixels = np.stack([px, py], axis=1)
    return pixels, valid


def is_covered_by_image_projection(
    pts_in_ego: np.ndarray,
    ref_pixel_tree: cKDTree | None,
    ref_extrinsic: list | np.ndarray,
    intrinsics: tuple,
    tolerance: float,
) -> np.ndarray:
    """
    判断 ego 点是否被参考相机的图像占用区域覆盖。

    把点投影到参考相机图像，查询最近参考像素，若在 tolerance 像素内则认为被覆盖。
    """
    if ref_pixel_tree is None or len(pts_in_ego) == 0:
        return np.zeros(len(pts_in_ego), dtype=bool)

    pixels, valid = project_to_camera_image(pts_in_ego, ref_extrinsic, *intrinsics)
    covered = np.zeros(len(pts_in_ego), dtype=bool)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return covered

    dist, _ = ref_pixel_tree.query(pixels[valid], k=1)
    covered[valid_indices] = dist <= tolerance
    return covered


def create_pcd(pts: np.ndarray, colors=None, uniform_color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None and len(colors) == len(pts):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif uniform_color is not None:
        pcd.paint_uniform_color(uniform_color)
    return pcd


def merge_cam_point_clouds(cam_data: list) -> tuple[np.ndarray, np.ndarray]:
    """把（重叠过滤后的）多路相机点云合并为一路。

    cam_data 中每项为 (pts_in_ego, cols, path, suffix, extrinsic, fov)。
    无颜色的相机用 DEFAULT_CAMERA_COLORS 中的默认色补齐，保证合并文件颜色完整。
    返回 (points, colors)。
    """
    all_pts = []
    all_cols = []
    for pts, cols, _path, suffix, _extrinsic, _fov in cam_data:
        if len(pts) == 0:
            continue
        all_pts.append(pts)
        if cols is not None and len(cols) == len(pts):
            all_cols.append(cols)
        else:
            color = np.asarray(
                DEFAULT_CAMERA_COLORS.get(suffix, [1.0, 0.2, 0.0]), dtype=np.float64
            )
            all_cols.append(np.tile(color, (len(pts), 1)))
    if not all_pts:
        raise ValueError("没有可合并的相机点云（全部为空）")
    return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0)


def save_merged_point_cloud(
    cam_data: list, out_path: Path, voxel_size: float = 0.0
) -> None:
    """合并四路相机点云并保存到文件（.ply / .pcd 由扩展名决定）。"""
    merged_pts, merged_cols = merge_cam_point_clouds(cam_data)
    merged_pcd = create_pcd(merged_pts, colors=merged_cols)
    if voxel_size and voxel_size > 0:
        before = len(merged_pts)
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
        print(
            f"[合并保存] 体素下采样 {voxel_size} m: {before} -> "
            f"{len(merged_pcd.points)} 点"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(out_path), merged_pcd)
    if not ok:
        raise RuntimeError(f"保存合并点云失败: {out_path}")
    print(f"[合并保存] 共 {len(merged_pcd.points)} 点，已保存到 {out_path}")


def infer_camera_suffix(cam_pcd_path: str | Path) -> str | None:
    """从点云文件名或其所在目录推断相机名（front/back/left/right），忽略大小写。"""
    path = Path(cam_pcd_path)
    # 优先从文件名判断
    name = path.stem.lower()
    for suffix in CAMERA_SENSOR_IDS:
        if suffix in name:
            return suffix
    # 文件名中没有时，向上检查目录名
    for parent in path.parents:
        parent_name = parent.name.lower()
        for suffix in CAMERA_SENSOR_IDS:
            if suffix in parent_name:
                return suffix
    return None


def load_extrinsic_from_calibration(cam_suffix: str, yaml_path: str | Path) -> list[float]:
    """从 calibration.yml 加载指定相机的外参 [tx ty tz qx qy qz qw]。"""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到标定文件: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sensor_id = CAMERA_SENSOR_IDS[cam_suffix]
    extrinsic = cfg["rig"][sensor_id]["extrinsic"]
    if len(extrinsic) != 7:
        raise ValueError(f"{sensor_id} 的 extrinsic 长度不是 7: {extrinsic}")
    return [float(v) for v in extrinsic]


def resolve_calibration_path(args) -> Path:
    """确定标定文件路径。

    优先级：
      1. 用户显式传入的 --calibration
      2. --cloud_dir/calibration.yml（若存在）
      3. 默认的 DEFAULT_CALIBRATION
    """
    if args.calibration is not None:
        return Path(args.calibration)
    cloud_calib = Path(args.cloud_dir) / "calibration.yml"
    if cloud_calib.exists():
        return cloud_calib
    return Path(DEFAULT_CALIBRATION)


def discover_timestamps(cloud_dir: Path) -> list[str]:
    """扫描 cloud_dir 下四个子目录，返回所有找到的时间戳列表。支持 .ply / .las / .laz。"""
    if not cloud_dir.exists():
        raise FileNotFoundError(f"找不到 cloud 目录: {cloud_dir}")
    timestamps: set[str] = set()
    for subdir in cloud_dir.iterdir():
        if not subdir.is_dir():
            continue
        for ext in ("*.ply", "*.las", "*.laz"):
            for f in subdir.glob(ext):
                ts = f.stem.split("_")[0]
                timestamps.add(ts)
    return sorted(timestamps)


def find_cam_pcd_paths(cloud_dir: Path, timestamp: str) -> list[str]:
    """在 cloud_dir 下查找该时间戳的四个相机点云，支持 .ply / .las / .laz 等命名。"""
    paths = []
    for suffix in CAMERA_SENSOR_IDS:
        cam_dir = cloud_dir / suffix
        if not cam_dir.exists():
            print(f"[警告] 缺少相机目录: {cam_dir}")
            continue
        candidates = []
        for ext in (".ply", ".las", ".laz"):
            candidates.extend(cam_dir.glob(f"{timestamp}*{ext}"))
        if len(candidates) == 1:
            paths.append(str(candidates[0]))
        elif len(candidates) > 1:
            raise ValueError(
                f"时间戳 {timestamp} 在 {cam_dir} 下匹配到多个文件: {candidates}"
            )
        else:
            print(f"[警告] 找不到 {suffix} 相机的时间戳 {timestamp} 点云")
    if not paths:
        raise FileNotFoundError(
            f"在 {cloud_dir} 下找不到时间戳 {timestamp} 的任何相机点云"
        )
    return paths


def find_ego_pcd(cloud_dir: Path, timestamp: str) -> Path | None:
    """查找对应时间戳的 ego/LiDAR 点云，优先 cloud_dir，其次父目录。支持 .pcd / .las / .laz。"""
    for base in [cloud_dir, cloud_dir.parent]:
        for ext in (".pcd", ".las", ".laz"):
            candidates = list(base.glob(f"{timestamp}*{ext}"))
            if candidates:
                return candidates[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="在 ego 坐标系下对齐可视化两个点云，支持 WASD 移动视角")
    parser.add_argument("--cam_pcd", action="append", help="相机坐标系下的点云文件路径（可多次传入，同时显示多个相机）")
    parser.add_argument("--ego_pcd", default=None, help="ego 坐标系下的点云文件路径；若未指定且使用 --cloud_dir，会自动查找同名 .pcd")
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="时间戳；若省略且指定了 --cloud_dir，会自动扫描目录下的时间戳",
    )
    parser.add_argument(
        "--cloud_dir",
        default=r"E:\lidar-depth\cloud",
        help="四个相机点云所在的根目录（默认 E:\\lidar-depth\\cloud）",
    )
    parser.add_argument(
        "--extrinsic",
        nargs=7,
        type=float,
        default=None,
        help="相机外参 [tx ty tz qx qy qz qw]；如不指定，根据 --cam_pcd 文件名自动从 --calibration 加载",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="标定文件路径；默认优先使用 --cloud_dir/calibration.yml，其次 E:\\lidar-depth\\calibration.yml",
    )
    parser.add_argument(
        "--cam_color",
        nargs=3,
        type=float,
        default=[1.0, 0.2, 0.0],
        help="相机点云在 ego 下的显示颜色 (R G B)，默认亮红",
    )
    parser.add_argument(
        "--ego_color",
        nargs=3,
        type=float,
        default=[0.0, 0.8, 1.0],
        help="ego 雷达点云的显示颜色 (R G B)，默认亮青",
    )
    parser.add_argument(
        "--frame_size",
        type=float,
        default=2.0,
        help="ego 坐标轴长度（米），默认 2",
    )
    parser.add_argument(
        "--align_pcd",
        action="store_true",
        default=True,
        help="将输入的 ego PCD 从 X右/Y前/Z上 对齐到标定的 X前/Y左/Z上（默认开启）",
    )
    parser.add_argument(
        "--no_align_pcd",
        action="store_false",
        dest="align_pcd",
        help="禁用 PCD 坐标系对齐",
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
        "--max_range",
        type=float,
        default=200.0,
        help="过滤相机点云中坐标绝对值超过该阈值（米）的离群点，默认 200；设为 0 则禁用",
    )
    parser.add_argument(
        "--fov_overlap_filter",
        action="store_true",
        default=True,
        help="基于图像投影过滤：把左右视角点投影到前后相机图像，移除像素位置被前后点云覆盖的部分（默认开启）",
    )
    parser.add_argument(
        "--no_fov_overlap_filter",
        action="store_false",
        dest="fov_overlap_filter",
        help="禁用基于图像投影的左右视角非重叠过滤",
    )
    parser.add_argument(
        "--pixel_tolerance",
        type=float,
        default=5.0,
        help="判断左右视角点是否被前后点云覆盖时允许的像素误差（默认 5 像素）",
    )
    parser.add_argument(
        "--save_merged",
        nargs="?",
        const="",
        default=None,
        help="重叠过滤后把四路相机点云合并保存为一个文件；"
        "可指定输出路径（.ply/.pcd），不带路径时自动保存到 <cloud_dir>/<时间戳>_merged.ply",
    )
    parser.add_argument(
        "--merged_voxel_size",
        type=float,
        default=0.0,
        help="合并保存时的体素下采样分辨率（米），默认 0 表示不下采样",
    )
    parser.add_argument(
        "--move_step",
        type=float,
        default=0.3,
        help="WASD 每次移动的距离（米），默认 0.3",
    )
    return parser.parse_args()


def get_camera_pose(vis):
    """从 ViewControl 读取当前相机位姿。"""
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.asarray(param.extrinsic, dtype=np.float64)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    pos = -R.T @ t
    front = -R[2, :]
    up = R[1, :]
    return pos, front / np.linalg.norm(front), up / np.linalg.norm(up)


def set_camera_pose(vis, pos, front, up):
    """直接把新的相机位姿写回 ViewControl。"""
    front = front / (np.linalg.norm(front) + 1e-9)
    up = up / (np.linalg.norm(up) + 1e-9)
    right = np.cross(front, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    # 重新正交化 up
    up = np.cross(right, front)
    up = up / (np.linalg.norm(up) + 1e-9)

    # world -> camera 旋转：相机坐标轴在世界系中的表示作为行
    R = np.vstack([right, up, -front])
    t = -R @ pos

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    param.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_renderer()


def build_key_callbacks(vis, step: float, pcd_cam_list: list | None = None, cam_pcd_paths: list | None = None):
    """注册 WASD 键盘回调（FPS 式移动）和 1/2/3/4 相机显隐切换。"""
    pos, front, up = get_camera_pose(vis)
    state = {"pos": pos, "front": front, "up": up}

    def right_vec():
        r = np.cross(state["front"], state["up"])
        return r / (np.linalg.norm(r) + 1e-9)

    def refresh():
        set_camera_pose(vis, state["pos"], state["front"], state["up"])

    def move_forward(vis):
        state["pos"] += state["front"] * step
        refresh()
        return True

    def move_back(vis):
        state["pos"] -= state["front"] * step
        refresh()
        return True

    def move_left(vis):
        state["pos"] -= right_vec() * step
        refresh()
        return True

    def move_right(vis):
        state["pos"] += right_vec() * step
        refresh()
        return True

    def move_up(vis):
        state["pos"] += state["up"] * step
        refresh()
        return True

    def move_down(vis):
        state["pos"] -= state["up"] * step
        refresh()
        return True

    vis.register_key_callback(ord("W"), move_forward)
    vis.register_key_callback(ord("w"), move_forward)
    vis.register_key_callback(ord("S"), move_back)
    vis.register_key_callback(ord("s"), move_back)
    vis.register_key_callback(ord("A"), move_left)
    vis.register_key_callback(ord("a"), move_left)
    vis.register_key_callback(ord("D"), move_right)
    vis.register_key_callback(ord("d"), move_right)
    vis.register_key_callback(ord("Q"), move_up)
    vis.register_key_callback(ord("q"), move_up)
    vis.register_key_callback(ord("E"), move_down)
    vis.register_key_callback(ord("e"), move_down)

    # 也支持方向键
    try:
        vis.register_key_callback(265, move_forward)  # GLFW_KEY_UP
        vis.register_key_callback(264, move_back)     # GLFW_KEY_DOWN
        vis.register_key_callback(263, move_left)     # GLFW_KEY_LEFT
        vis.register_key_callback(262, move_right)    # GLFW_KEY_RIGHT
    except Exception:
        pass

    # 1/2/3/4 切换前后左右相机点云显示
    if pcd_cam_list is not None and cam_pcd_paths is not None:
        cam_index = {
            infer_camera_suffix(path): idx
            for idx, path in enumerate(cam_pcd_paths)
            if infer_camera_suffix(path) is not None
        }
        visible = [True] * len(pcd_cam_list)

        def make_toggle(suffix: str):
            def toggle(vis):
                idx = cam_index.get(suffix)
                if idx is None:
                    return True
                if visible[idx]:
                    vis.remove_geometry(pcd_cam_list[idx])
                    visible[idx] = False
                    print(f"[显示] 隐藏 {suffix}")
                else:
                    vis.add_geometry(pcd_cam_list[idx])
                    visible[idx] = True
                    print(f"[显示] 显示 {suffix}")
                vis.update_renderer()
                return True

            return toggle

        vis.register_key_callback(ord("1"), make_toggle("front"))
        vis.register_key_callback(ord("2"), make_toggle("back"))
        vis.register_key_callback(ord("3"), make_toggle("left"))
        vis.register_key_callback(ord("4"), make_toggle("right"))


def main():
    args = parse_args()

    # 确定相机点云路径
    if args.cam_pcd is not None:
        cam_pcd_paths = args.cam_pcd
        timestamp = None
    else:
        cloud_dir = Path(args.cloud_dir)
        if args.timestamp is not None:
            timestamp = args.timestamp
            cam_pcd_paths = find_cam_pcd_paths(cloud_dir, timestamp)
        else:
            timestamps = discover_timestamps(cloud_dir)
            if not timestamps:
                raise ValueError(f"在 {cloud_dir} 下没有找到任何时间戳的点云")
            if len(timestamps) == 1:
                timestamp = timestamps[0]
                print(f"自动发现唯一时间戳: {timestamp}")
            else:
                print(f"在 {cloud_dir} 下发现多个时间戳:")
                for ts in timestamps:
                    print(f"  {ts}")
                raise ValueError("请用 --timestamp 指定要显示的时间戳")
            cam_pcd_paths = find_cam_pcd_paths(cloud_dir, timestamp)

    # 确定标定文件路径
    calibration_path = resolve_calibration_path(args)
    print(f"使用标定文件: {calibration_path}")

    n_cams = len(cam_pcd_paths)
    if n_cams > 1 and args.extrinsic is not None:
        raise ValueError(
            "同时显示多个相机时，请从文件名自动推断外参，不要显式传入 --extrinsic。"
        )

    # 确定 ego 点云路径
    ego_pcd_path = args.ego_pcd
    if ego_pcd_path is None and timestamp is not None:
        ego_candidate = find_ego_pcd(Path(args.cloud_dir), timestamp)
        if ego_candidate is not None:
            ego_pcd_path = str(ego_candidate)
            print(f"自动发现 ego 点云: {ego_pcd_path}")
        else:
            print("[提示] 未找到对应时间戳的 ego 点云，将只显示相机点云")

    # 加载 ego 点云（可选）
    pcd_ego = None
    if ego_pcd_path is not None:
        ego_pts, ego_cols = load_point_cloud(ego_pcd_path)
        # PCD 坐标系对齐到标定 ego
        if args.align_pcd:
            ego_pts = align_pcd_to_ego(ego_pts)
            print("已对齐 PCD 坐标系到标定 ego 坐标系")
        pcd_ego = create_pcd(ego_pts, colors=ego_cols, uniform_color=args.ego_color)
        print(f"\nego 点云点数: {len(ego_pts)}")

    pcd_cam_list = []
    print(f"\n共加载 {n_cams} 个相机点云")
    cam_data = []  # (pts_in_ego, cols, cam_path, cam_suffix, extrinsic, fov)
    for i, cam_path in enumerate(cam_pcd_paths, start=1):
        cam_pts, cam_cols = load_point_cloud(cam_path)

        # 过滤相机点云中的极端离群点
        if args.max_range > 0:
            cam_pts, cam_cols = filter_outliers(cam_pts, cam_cols, args.max_range)

        # 推断相机名
        cam_suffix = infer_camera_suffix(cam_path)
        if cam_suffix is None:
            raise ValueError(
                f"无法从文件名推断相机名: {cam_path}\n"
                "请在文件名中包含 front/back/left/right 之一，或显式传入 --extrinsic。"
            )

        # 确定外参与 FOV
        if args.extrinsic is not None:
            extrinsic = args.extrinsic
            fov = None
            print(f"[{i}/{n_cams}] {cam_path} -> {cam_suffix} 相机，使用命令行传入的外参")
        else:
            extrinsic = load_extrinsic_from_calibration(cam_suffix, calibration_path)
            fov = load_fov_from_calibration(cam_suffix, calibration_path)
            print(f"[{i}/{n_cams}] {cam_path} -> {cam_suffix} 相机，外参 {extrinsic}，FOV {fov}°")

        # 相机点云 -> ego
        cam_pts_in_ego = cam_to_ego(cam_pts, extrinsic)
        cam_data.append((cam_pts_in_ego, cam_cols, cam_path, cam_suffix, extrinsic, fov))

    # 基于图像投影的左右视角非重叠区域过滤：
    # 把 front/back 相机自己的点投影到它自己的图像平面，建立“占用像素 KDTree”；
    # 对 left/right 点，分别投影到 front/back 图像，若落在已被占用的像素附近（tolerance 内），
    # 则认为该 left/right 点被前后视角覆盖，予以移除。
    if args.fov_overlap_filter:
        ref_configs = []
        for pts, cols, path, suffix, extrinsic, fov in cam_data:
            if suffix in ("front", "back"):
                if extrinsic is None:
                    ref_configs = []
                    break
                intrinsics = load_intrinsics_from_calibration(suffix, calibration_path)
                pixels, valid = project_to_camera_image(pts, extrinsic, *intrinsics)
                valid_pixels = pixels[valid]
                tree = cKDTree(valid_pixels) if len(valid_pixels) > 0 else None
                ref_configs.append((tree, extrinsic, intrinsics))
                print(f"[图像投影] {suffix}: 建立 {len(valid_pixels)} 像素的占用图")

        if ref_configs:
            print("\n[图像投影非重叠过滤] 使用前后视角图像占用区域，移除左右视角中重合的部分")
            for i, (pts, cols, path, suffix, extrinsic, fov) in enumerate(cam_data):
                if suffix not in ("left", "right"):
                    continue
                if extrinsic is None:
                    print(f"[图像投影非重叠过滤] 跳过 {suffix}：缺少外参")
                    continue

                in_front_or_back = np.zeros(len(pts), dtype=bool)
                for tree, ref_ext, intrinsics in ref_configs:
                    covered = is_covered_by_image_projection(
                        pts, tree, ref_ext, intrinsics, args.pixel_tolerance
                    )
                    in_front_or_back |= covered

                keep = ~in_front_or_back
                n_removed = len(pts) - int(keep.sum())
                if n_removed > 0:
                    print(
                        f"[图像投影非重叠过滤] {suffix}: 移除与前后视角重合的 {n_removed} 点，"
                        f"保留 {int(keep.sum())} 点"
                    )
                cam_data[i] = (
                    pts[keep],
                    cols[keep] if cols is not None else None,
                    path,
                    suffix,
                    extrinsic,
                    fov,
                )
        else:
            print(
                "\n[图像投影非重叠过滤] 未找到完整的前后视角参数（可能使用了 --extrinsic），跳过过滤"
            )

    # 重叠过滤后，把四路相机点云合并保存为一个完整文件
    if args.save_merged is not None:
        if args.save_merged:
            merged_out = Path(args.save_merged)
        else:
            ts_part = timestamp if timestamp is not None else "merged"
            merged_out = Path(args.cloud_dir) / f"{ts_part}_merged.ply"
        save_merged_point_cloud(cam_data, merged_out, args.merged_voxel_size)

    for i, (cam_pts_in_ego, cam_cols, cam_path, cam_suffix, _, _) in enumerate(
        cam_data, start=1
    ):
        # 颜色：优先使用点云自带颜色；没有颜色时再根据单/多相机选择默认色
        has_colors = cam_cols is not None and len(cam_cols) == len(cam_pts_in_ego)
        if has_colors:
            uniform_color = None
            print(f"[{i}/{n_cams}] 使用点云自带颜色")
        else:
            if n_cams > 1:
                uniform_color = DEFAULT_CAMERA_COLORS.get(cam_suffix, args.cam_color)
                print(f"[{i}/{n_cams}] 点云无颜色，使用默认 {cam_suffix} 颜色")
            else:
                uniform_color = args.cam_color
                print(f"[{i}/{n_cams}] 点云无颜色，使用 --cam_color")

        pcd_cam = create_pcd(cam_pts_in_ego, colors=cam_cols, uniform_color=uniform_color)
        pcd_cam_list.append(pcd_cam)

    # ego 坐标系参考轴
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=args.frame_size, origin=[0, 0, 0]
    )

    print("\n交互操作:")
    print("  左键拖动 : 旋转视角")
    print("  滚轮     : 缩放")
    print("  Shift+左键拖动 : 平移")
    print("  W/S/A/D : 前后左右移动视角")
    print("  Q/E     : 上升/下降视角")
    print("  1/2/3/4 : 显示/隐藏 front/back/left/right 相机点云")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Camera + EGO point clouds aligned in EGO",
        width=1280,
        height=720,
    )
    for pcd_cam in pcd_cam_list:
        vis.add_geometry(pcd_cam)
    if pcd_ego is not None:
        vis.add_geometry(pcd_ego)
    vis.add_geometry(frame)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray(args.bg_color, dtype=np.float64)
    opt.light_on = True

    # 让窗口先完成一次渲染，确保能读到正确的初始相机位姿
    vis.poll_events()
    vis.update_renderer()
    build_key_callbacks(vis, args.move_step, pcd_cam_list, cam_pcd_paths)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
