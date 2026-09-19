# Docker 部署指南

## 前置条件

### 1. 安装 Docker
```bash
# Ubuntu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 注销重登，使 docker 组生效
```

### 2. 安装 NVIDIA Container Toolkit（让容器能用 GPU）
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

验证：
```bash
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
# 看到 GPU 表格 = OK
```

---

## 项目目录结构

```
Metric3D/
├── Dockerfile              ← 构建镜像的配方
├── docker-compose.yml      ← 运行容器的配置
├── requirements.txt        ← Python 依赖清单
├── .dockerignore           ← 排除大文件，避免镜像爆炸
├── DOCKER_GUIDE.md         ← 本文件
│
├── infer_and_fuse.py       ← 主入口
├── infer_common.py
├── infer_undistorted.py
├── visualize_two_clouds_in_ego.py
├── training/               ← 模型配置
├── mono/                   ← 模型代码
├── onnx/                   ← TRT 引擎 + 推理脚本
│   ├── metric3d_vit_large_fp16.engine
│   ├── metric3d_vit_large.onnx
│   └── infer_trt.py
│
│   ↓ 以下通过 volume 挂载，不进镜像 ↓
│
├── ../metric_depth_vit_large_800k.pth     ← 1.6 GB 权重
└── ../Datas01/                            ← 数据集
```

---

## 构建镜像

```bash
cd /root/ly/Map/Metric3D

# 第一次构建（~30 分钟，要下 PyTorch 等）
docker compose build

# 查看镜像
docker images metric3d-trt
# REPOSITORY      TAG       SIZE
# metric3d-trt    latest    ~12 GB
```

⚠️ 镜像很大（PyTorch + CUDA + TensorRT 加起来 10+ GB），**这是正常的**。

---

## 运行容器

### 方式 1：跑默认命令（推理）

```bash
docker compose run --rm metric3d
```

会执行 `python infer_and_fuse.py`，使用默认配置。

### 方式 2：进入容器调试

```bash
docker compose run --rm metric3d bash
# 在容器里：
cd /workspace/Metric3D
python infer_and_fuse.py
```

### 方式 3：单条命令

```bash
docker compose run --rm metric3d \
    python -c "import torch; print(torch.cuda.is_available())"
```

---

## 挂载说明（重要）

容器里的路径映射：

| 宿主机 | 容器内 | 说明 |
|---|---|---|
| `/root/ly/Map/Metric3D/` | `/workspace/Metric3D/` | 项目代码 |
| `/root/ly/Map/metric_depth_vit_large_800k.pth` | `/models/metric_depth_vit_large_800k.pth` | 模型权重 |
| `/root/ly/Map/Metric3D/onnx/` | `/trt_engines/` | TRT engines |
| `/root/ly/Map/` | `/data/` | 数据集根目录 |

**因此 Config 类里路径要改成容器内路径**：

```python
class Config:
    dataset_root = '/data/Datas01'                  # 不是 /root/ly/Map/Datas01
    config      = 'training/mono/configs/RAFTDecoder/vit.raft5.large.py'
    ckpt        = '/models/metric_depth_vit_large_800k.pth'   # 改这里
    trt_engine  = '/trt_engines/metric3d_vit_large_fp16.engine'  # 改这里
```

---

## 跨机器部署

### 1. 在原机器上导出镜像

```bash
docker save metric3d-trt:latest | gzip > metric3d-trt.tar.gz
# ~5 GB（gzip 后）
```

### 2. 传到新机器

```bash
scp metric3d-trt.tar.gz user@new-host:~/
```

### 3. 在新机器上导入

```bash
docker load < metric3d-trt.tar.gz
```

### 4. ⚠ TRT engine 不能跨 GPU

如果新机器 GPU 架构不同（如 3050 → 4090），**必须重新编译 engine**：

```bash
docker compose run --rm metric3d \
    python onnx/infer_trt.py \
        --onnx /trt_engines/metric3d_vit_large.onnx \
        --engine /trt_engines/metric3d_vit_large_fp16_new.engine \
        --fp16 --build-only
```

---

## 常见问题

### Q1: 容器里看不到 GPU
**症状**：`torch.cuda.is_available()` 返回 False  
**原因**：NVIDIA Container Toolkit 没装好  
**解决**：
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Q2: 镜像太大想瘦身
使用 multi-stage build（多阶段构建）：
```dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 as final
# 复制编译好的 Python 包，不带 devel 工具
```
能省 1~2 GB。

### Q3: 编译 mmcv 失败
mmcv 必须用**预编译 wheel**，不能用 pip 默认源：
```dockerfile
RUN pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
```

### Q4: pip install 太慢
在 Dockerfile 里换源：
```dockerfile
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 构建优化建议

1. **利用层缓存**：先 COPY requirements.txt 装 pip 包，再 COPY 代码。改代码时不用重装 pip。

2. **使用 BuildKit**（更快）：
   ```bash
   DOCKER_BUILDKIT=1 docker compose build
   ```

3. **多阶段构建**：
   - 第一阶段：装编译工具，编译 pycuda
   - 第二阶段：runtime 镜像，只复制产物
   - 能减少 ~2 GB

---

## 一键部署清单

部署到新机器需要：

- [x] 安装 Docker
- [x] 安装 NVIDIA Container Toolkit
- [x] 加载镜像 `docker load < metric3d-trt.tar.gz`
- [x] 准备 `docker-compose.yml`
- [x] 准备数据（Datas01 等）
- [x] 准备模型权重（如果架构不同，重新编译 engine）
