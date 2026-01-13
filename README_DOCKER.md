# Docker Compose 使用说明

本文档说明如何使用 Docker Compose 一键运行画线执行任务脚本。

只需设置好docker代理和DISPLAY隧道即可。


## 前置要求

1. **Docker** 和 **Docker Compose** 已安装
2. **NVIDIA Docker** 支持（用于 GPU 加速）
3. **X11 显示服务器**（用于显示图像窗口）

## 一键运行

### 方式1: 直接使用 docker-compose（推荐）

```bash
# 设置 X11 权限（首次使用）
xhost +local:docker

# 一键运行
docker-compose up --build
```

### 方式2: 使用启动脚本

```bash
chmod +x docker-compose-run.sh
./docker-compose-run.sh
```

## 配置说明

### 环境变量

可以通过环境变量或 `.env` 文件配置参数：

```bash
# 方式1: 使用环境变量
export DISPLAY=:0
export ENV_NAME=google_robot_pick_coke_can
export SCENE_NAME=google_scene
docker-compose up

# 方式2: 使用 .env 文件
cp env.example .env
# 编辑 .env 文件
nano .env
docker-compose up
```

### 可配置参数

- `ENV_NAME`: 环境名称（默认: `google_robot_pick_coke_can`）
- `SCENE_NAME`: 场景名称（默认: `google_scene`）
- `ROBOT`: 机器人名称（默认: `google_robot`）
- `ROBOT_INIT_X`: 机器人初始 x 坐标（默认: `0.0`）
- `ROBOT_INIT_Y`: 机器人初始 y 坐标（默认: `0.0`）
- `OBS_CAMERA_NAME`: 观察相机名称（默认: `3rd_view_camera`）
- `LOGGING_DIR`: 日志目录（默认: `/workspace/results`）
- `DISPLAY`: X11 显示（默认: `:0`）

## 使用说明

1. **启动容器后**，会显示当前场景的图像窗口
2. **在图像上画线**：
   - 按住鼠标左键拖拽画一条线
   - 按 `空格键` 确认并开始执行
   - 按 `r` 键清除重新画
   - 按 `q` 键退出
3. **执行完成后**，视频会保存到 `./results` 目录

## Docker Compose 特性

- ✅ **一键运行**: `docker-compose up` 即可启动
- ✅ **GPU 支持**: 自动使用所有可用的 NVIDIA GPU
- ✅ **X11 显示**: 支持图形界面显示（用于画线交互）
- ✅ **网络模式**: 使用 `host` 模式，便于网络通信
- ✅ **IPC/PID 共享**: 支持共享内存和进程间通信
- ✅ **特权模式**: 使用 `privileged: true` 以获得完整权限
- ✅ **目录挂载**: 整个项目目录挂载到容器，方便实时修改

## 常见问题

### 1. 无法显示窗口

**问题**: 窗口无法显示

**解决方案**:
```bash
# 检查 DISPLAY 环境变量
echo $DISPLAY

# 设置 DISPLAY（如果未设置）
export DISPLAY=:0

# 允许 X11 连接
xhost +local:docker
```

### 2. GPU 不可用

**问题**: 容器内无法使用 GPU

**解决方案**:
```bash
# 检查 NVIDIA Docker 是否安装
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# 如果上述命令失败，需要安装 nvidia-docker2
# Ubuntu/Debian:
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. 权限问题

**问题**: X11 权限错误

**解决方案**:
```bash
# 允许 X11 连接
xhost +local:docker

# 或者使用启动脚本自动设置
./docker-compose-run.sh
```

### 4. 代理设置

如果需要使用代理，设置环境变量：

```bash
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
docker-compose up
```

## 停止容器

```bash
# 停止并删除容器
docker-compose down

# 停止并删除容器和卷
docker-compose down -v
```

## 查看日志

```bash
# 查看实时日志
docker-compose logs -f

# 查看最近日志
docker-compose logs --tail=100
```

## 进入容器调试

```bash
# 进入运行中的容器
docker-compose exec simpler-env bash

# 或者启动新的交互式容器
docker-compose run --rm simpler-env bash
```

## 结果文件

执行完成后，结果文件会保存在 `./results` 目录：
- `draw_line_execute.mp4`: 执行过程的视频

## 注意事项

1. **首次运行**需要构建镜像，可能需要较长时间
2. **GPU 内存**：确保有足够的 GPU 内存（建议至少 8GB）
3. **X11 显示**：需要 X11 服务器运行，远程连接可能需要 X11 转发
4. **网络**：如果需要下载模型或数据，确保网络连接正常
5. **目录挂载**：整个项目目录挂载到容器，修改代码后无需重建镜像

## 快速命令参考

```bash
# 一键运行
docker-compose up --build

# 后台运行
docker-compose up -d --build

# 停止
docker-compose down

# 查看日志
docker-compose logs -f

# 进入容器
docker-compose exec simpler-env bash
```
