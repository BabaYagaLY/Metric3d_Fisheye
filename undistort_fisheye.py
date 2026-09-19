"""
鱼眼去畸变工具：
  1) 单图模式：--img + --calib
  2) 批量模式：--origin-root + --calib，自动处理 front/back/left/right 四个视角的同步时间戳

输出：
  - 去畸变图：--undistort-root/{view}/{timestamp}.jpg
  - 新内参：  --newK-root/{view}/new_K.npy
"""
import os
import sys
import argparse
import re

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


VIEW_TO_CAM_KEY = {
    'front': 'cam_hy_n5_avm_front',
    'back': 'cam_hy_n5_avm_back',
    'left': 'cam_hy_n5_avm_left',
    'right': 'cam_hy_n5_avm_right',
}


def extract_timestamp(filename):
    """从文件名中提取时间戳，例如 1234567890.jpg 或 1234567890.123.jpg"""
    base = os.path.splitext(filename)[0]
    m = re.search(r'(\d+(?:\.\d+)?)$', base)
    if m:
        return m.group(1)
    return None


def load_fisheye_calib(calib_path, cam_key):
    """从 calibration.yml 读取指定相机的鱼眼内参与畸变系数。"""
    if yaml is None:
        raise ImportError("请安装 PyYAML: pip install pyyaml")

    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)

    if 'rig' not in calib:
        raise ValueError("calibration.yml 顶层缺少 'rig' 字段")

    rig = calib['rig']
    if cam_key not in rig:
        available = [k for k in rig.keys() if 'avm' in k or 'fisheye' in k.lower()]
        raise ValueError(f"calibration.yml 中找不到 {cam_key}，可用相机: {available}")

    cam = rig[cam_key]
    focal = cam['focal']
    pp = cam['pp']
    inv_poly = cam.get('inv_poly', [0.0, 0.0, 0.0, 0.0])

    K = np.array([
        [focal[0], 0.0, pp[0]],
        [0.0, focal[1], pp[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    D = np.array(inv_poly, dtype=np.float32)
    return K, D


def undistort_image(img, K, D, balance=0.0, new_size=None):
    """对单张鱼眼图去畸变，返回去畸变图 + 新内参。"""
    h, w = img.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    out_size = (w, h) if new_size is None else new_size
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, out_size, cv2.CV_32FC1
    )
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return undist, new_K


def process_single(args):
    """单图模式。"""
    os.makedirs(args.out_dir, exist_ok=True)

    cam_key = args.cam if args.cam else VIEW_TO_CAM_KEY.get(
        os.path.basename(os.path.dirname(args.img)).lower(), None
    )
    if cam_key is None:
        raise ValueError(f"无法推断相机 key，请显式指定 --cam")

    K, D = load_fisheye_calib(args.calib, cam_key)
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {args.img}")

    new_size = None
    if args.new_size:
        parts = args.new_size.split(',')
        new_size = (int(parts[0]), int(parts[1]))

    undist, new_K = undistort_image(img, K, D, balance=args.balance, new_size=new_size)

    base_name = os.path.splitext(os.path.basename(args.img))[0]
    out_img_path = os.path.join(args.out_dir, f'{base_name}_undistort.jpg')
    out_K_path = os.path.join(args.out_dir, f'{base_name}_undistort_K.npy')

    cv2.imwrite(out_img_path, undist)
    np.save(out_K_path, new_K.astype(np.float32))

    print(f"\n去畸变完成:")
    print(f"  输出图像: {out_img_path}")
    print(f"  新内参 K:\n{new_K}")
    print(f"  新内参已保存: {out_K_path}")


def process_batch(args):
    """批量模式：处理 front/back/left/right 四个视角的同步时间戳。"""
    if yaml is None:
        raise ImportError("请安装 PyYAML: pip install pyyaml")

    views = args.views
    origin_root = args.origin_root
    undistort_root = args.undistort_root
    newK_root = args.newK_root

    os.makedirs(undistort_root, exist_ok=True)
    os.makedirs(newK_root, exist_ok=True)

    # 收集每个视角的时间戳
    view_timestamps = {}
    view_ext = {}
    for view in views:
        view_dir = os.path.join(origin_root, view)
        if not os.path.isdir(view_dir):
            print(f"[Warning] 跳过不存在的视角目录: {view_dir}")
            continue

        timestamps = {}
        for f in sorted(os.listdir(view_dir)):
            ts = extract_timestamp(f)
            if ts is None:
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            timestamps[ts] = f
            view_ext[view] = ext
        view_timestamps[view] = timestamps

    # 取所有视角的公共时间戳
    common_ts = sorted(set.intersection(*[set(v.keys()) for v in view_timestamps.values()]))
    if not common_ts:
        raise ValueError("未找到四个视角的公共时间戳，请检查文件名")

    print(f"发现 {len(common_ts)} 个公共时间戳")
    print(f"视角: {list(view_timestamps.keys())}")

    # 缓存每个视角的 K/D 和新 K
    view_KD = {}
    view_newK = {}

    for ts in common_ts:
        for view in views:
            view_dir = os.path.join(origin_root, view)
            src_path = os.path.join(view_dir, view_timestamps[view][ts])

            # 加载/缓存该视角的标定参数
            if view not in view_KD:
                cam_key = VIEW_TO_CAM_KEY[view]
                K, D = load_fisheye_calib(args.calib, cam_key)
                view_KD[view] = (K, D)

                # 用第一张图估计新内参（假设同视角所有图尺寸一致）
                img0 = cv2.imread(src_path)
                if img0 is None:
                    raise FileNotFoundError(f"无法读取图像: {src_path}")
                h, w = img0.shape[:2]
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, (w, h), np.eye(3), balance=args.balance
                )
                view_newK[view] = new_K.astype(np.float32)

                # 保存新内参
                view_newK_dir = os.path.join(newK_root, view)
                os.makedirs(view_newK_dir, exist_ok=True)
                np.save(os.path.join(view_newK_dir, 'new_K.npy'), view_newK[view])
                print(f"[{view}] 新内参:\n{view_newK[view]}")

            K, D = view_KD[view]
            new_K = view_newK[view]

            img = cv2.imread(src_path)
            if img is None:
                raise FileNotFoundError(f"无法读取图像: {src_path}")
            h, w = img.shape[:2]

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
            )
            undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

            out_view_dir = os.path.join(undistort_root, view)
            os.makedirs(out_view_dir, exist_ok=True)
            ext = view_ext.get(view, '.jpg')
            out_path = os.path.join(out_view_dir, f'{ts}{ext}')
            cv2.imwrite(out_path, undist)

        print(f"已处理: {ts}")

    print(f"\n批量去畸变完成。输出:")
    print(f"  去畸变图: {undistort_root}/{{view}}/{ts}{ext}")
    print(f"  新内参:   {newK_root}/{{view}}/new_K.npy")


def main():
    parser = argparse.ArgumentParser(description='鱼眼图像去畸变')
    # 单图模式参数
    parser.add_argument('--img', default=None, help='单图模式：输入鱼眼图像路径')
    parser.add_argument('--out-dir', default='./undistort_output', help='单图模式：输出目录')

    # 批量模式参数
    parser.add_argument('--origin-root', default='/root/ly/Map/Datas/origin',
                        help='批量模式：原始鱼眼图根目录')
    parser.add_argument('--undistort-root', default='/root/ly/Map/Datas/undistort',
                        help='批量模式：去畸变图输出根目录')
    parser.add_argument('--newK-root', default='/root/ly/Map/Datas/new_K',
                        help='批量模式：新内参输出根目录')
    parser.add_argument('--views', nargs='+', default=['front', 'back', 'left', 'right'],
                        help='要处理的视角列表')

    # 公共参数
    parser.add_argument('--calib', default='/root/ly/Map/Datas/calibration.yml',
                        help='calibration.yml 路径')
    parser.add_argument('--cam', default=None,
                        help='单图模式下显式指定相机 key；批量模式下根据目录名自动推断')
    parser.add_argument('--balance', type=float, default=0.0,
                        help='去畸变 balance 参数')
    parser.add_argument('--new-size', default=None,
                        help='输出图像尺寸 (W,H)，例如 "1920,1280"')
    args = parser.parse_args()

    if args.img:
        process_single(args)
    else:
        process_batch(args)


if __name__ == '__main__':
    main()
