# =====================================================================
# Metric3D + TensorRT 推理镜像
# =====================================================================
# 基础镜像：CUDA 11.7 + cuDNN 8 + 开发工具
# 选择 devel 版本因为 pycuda/tensorrt 编译时需要 nvcc
# =====================================================================
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# 避免交互式询问（tzdata 等）
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# =====================================================================
# 1. 系统依赖
# =====================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        wget \
        curl \
        ca-certificates \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python -m pip install --upgrade pip

# =====================================================================
# 2. Python 依赖（单独一层，利用 Docker 缓存）
# =====================================================================
WORKDIR /workspace/Metric3D
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir mako pytools siphash24   # pycuda 编译依赖

# =====================================================================
# 3. 项目代码（再一层，代码变化时不重装依赖）
# =====================================================================
COPY . /workspace/Metric3D

# =====================================================================
# 4. 环境变量
# =====================================================================
ENV PYTHONPATH=/workspace/Metric3D:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# =====================================================================
# 5. 默认入口
# =====================================================================
# 注意：模型权重、TRT engine、数据集都通过 -v 挂载，不烤进镜像
# 详见 docker-compose.yml
WORKDIR /workspace/Metric3D
CMD ["python", "infer_and_fuse.py"]
