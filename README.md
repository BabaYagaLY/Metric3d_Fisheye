# 基于 Metric3D 的鱼眼环视深度估计与点云重建

面向车载环视（AVM）的单目深度估计与多视角点云重建项目：输入 4 路 1920×1280 鱼眼原始图像，经「mask + 去畸变 → Metric3D 单目深度推理 → 相机到 ego 坐标变换 → 多视角重叠剔除」，输出车辆周围一圈融合后的三维点云，为感知评测与 3D 标注提供统一空间参考。

## 特性

- **鱼眼适配**：OpenCV 鱼眼模型去畸变 + 新针孔内参估计（balance=0.0 保留完整视野），按时间戳对齐四路同步帧。
- **绝对尺度深度**：基于 Metric3D 的 Canonical Camera 机制，用焦距比把 canonical 深度还原为真实米制深度。
- **自研鱼眼反投影**：针对 OpenCV `fisheye.undistortPoints` 在大入射角下反解退化的问题，实现基于 Newton 迭代的 Kannala-Brandt 逆投影求解器（有效域判定 + 解钳制 + active-set 逐点收敛提前退出），round-trip 误差 < 1e-12。
- **多视角重叠剔除**：基于图像投影占用的 KDTree 判定（5 像素容差），保留像素级遮挡一致性、消除多相机重影。
- **推理后端可插拔**：PyTorch / TensorRT 热切换，FP16 部署（单帧约 2s → 300ms、激活显存 1.9GB → 480MB，RTX 3050）。
- **Docker 部署**：Dockerfile + docker-compose，GPU 直通，大权重走 volume 挂载。

## 数据流

```
四路鱼眼原图 1920×1280
  → mask + 去畸变（cv2.fisheye.initUndistortRectifyMap）
  → 新针孔图 + 新内参
  → Metric3D 推理（PyTorch / TensorRT FP16）
  → recover_depth（去 padding / 插值 / 焦距比恢复真实尺度）
  → 针孔反投影 → 相机系点云
  → cam_to_ego 坐标变换 → ego 系点云
  → 前后视角自投影建 KDTree 占用，左右视角投影查询剔除重叠
  → 融合点云（单帧约 620 万点）
```

## 目录结构

```
.
├─ infer_and_fuse.py                  # 主入口：去畸变→推理→融合 流水线（Config / step1/2/3）
├─ infer_common.py                    # 模型加载、预处理、recover_depth、KB 逆投影求解
├─ undistort_fisheye.py               # 鱼眼去畸变 + 新针孔内参估计
├─ infer_undistorted.py               # 去畸变图推理（针孔路径）
├─ infer_fisheye.py                   # 鱼眼原图直接推理（实验路径）
├─ infer_undistorted_to_fisheye.py    # 针孔深度图重映射回鱼眼坐标
├─ visualize_two_clouds_in_ego.py     # 双点云 ego 系可视化 + 重叠剔除
├─ visualize_merged_vs_lidar.py       # 融合点云与 LiDAR 真值叠加验证
├─ onnx/                              # ONNX 导出 / TensorRT 部署
├─ training/                          # 训练与微调（可选）
├─ mono/                              # Metric3D 模型源码
├─ Dockerfile / docker-compose.yml    # Docker 部署
├─ DOCKER_GUIDE.md
├─ requirements.txt
└─ 导学-metric3d.md                   # 项目导学 / 面试准备
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据：四路鱼眼原图 + 标定 + mask（参考 data/ 目录）

# 3. 去畸变（生成针孔图与新内参）
python undistort_fisheye.py

# 4. 深度推理 + 点云融合
python infer_and_fuse.py
```

> 具体路径与开关以脚本顶部 Config 为准；各步骤可单独重跑、支持断点续跑。

## 效果（自测口径）

- 单帧时延：约 2s → 300ms（TensorRT FP16，RTX 3050）
- 激活显存：1.9GB → 480MB
- 自研逆投影求解器 round-trip 误差 < 1e-12，大入射角（θ > 1.4 rad）精度优于 OpenCV
- 单帧融合约 620 万点，与 LiDAR 真值在约 10 米静态场景肉眼对齐（定性）

## 说明

- 点云与 LiDAR 的对齐目前为肉眼/定性验证，未做定量误差评测。
- 训练侧（training/）为可选微调，默认使用预训练 Metric3D 权重。

## 致谢

深度估计模型基于 [Metric3D](https://github.com/YvanYin/Metric3D)（DINOv2 ViT-Large + RAFT Decoder）。

## License

见仓库 `LICENSE` 文件。
