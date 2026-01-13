#!/bin/bash
# SimplerEnv 画线执行任务启动脚本 - 使用 widowx 机器人

set -e

echo "=========================================="
echo "SimplerEnv 画线执行任务 (WidowX)"
echo "=========================================="

# 检查 X11 是否可用
if [ -z "$DISPLAY" ]; then
    echo "警告: DISPLAY 环境变量未设置，尝试使用默认值 :0"
    export DISPLAY=:0
fi

# 检查 X11 socket（仅对本地显示进行检查）
# 远程显示（如 host:display）不需要本地 socket
if [[ "$DISPLAY" =~ ^: ]]; then
    # 本地显示，检查 socket
    display_num=${DISPLAY#*:}
    if [ ! -S /tmp/.X11-unix/X${display_num} ]; then
        echo "警告: X11 socket 不存在 (/tmp/.X11-unix/X${display_num})，可能无法显示窗口"
    else
        echo "X11 socket 检查通过: /tmp/.X11-unix/X${display_num}"
    fi
else
    # 远程显示，不需要本地 socket
    echo "使用远程 X11 显示: $DISPLAY"
    echo "提示: 确保远程 X11 服务器允许连接"
    
    # 测试远程 X11 连接
    echo "测试 X11 连接..."
    x11_test_passed=false
    
    # 方法1: 使用 xdpyinfo（如果可用）
    if command -v xdpyinfo &> /dev/null; then
        if xdpyinfo -display "$DISPLAY" &> /dev/null 2>&1; then
            echo "✓ X11 连接测试成功 (使用 xdpyinfo)"
            x11_test_passed=true
        else
            echo "✗ xdpyinfo 测试失败"
        fi
    else
        echo "  xdpyinfo 不可用，尝试其他方法..."
    fi
    
    # 方法2: 使用 Python 测试 X11 连接（备用方法）
    if [ "$x11_test_passed" = false ]; then
        if python3 -c "
import os
import sys
os.environ['DISPLAY'] = '$DISPLAY'
try:
    import tkinter
    root = tkinter.Tk()
    root.withdraw()  # 不显示窗口
    root.destroy()
    print('✓ X11 连接测试成功 (使用 Python tkinter)')
    sys.exit(0)
except Exception as e:
    print('✗ Python tkinter 测试失败:', str(e))
    sys.exit(1)
" 2>&1; then
            x11_test_passed=true
        fi
    fi
    
    # 如果所有测试都失败，显示诊断信息
    if [ "$x11_test_passed" = false ]; then
        echo "✗ X11 连接测试失败，可能无法显示窗口"
        display_num=${DISPLAY#*:}
        x11_port=$((6000 + display_num))
        echo "  请确保:"
        echo "  1. 远程 X11 服务器 (${DISPLAY%:*}) 正在运行"
        echo "  2. 网络连接正常"
        echo "  3. 防火墙允许 X11 连接（端口 ${x11_port}）"
        echo "  4. 远程 X11 服务器允许来自容器的连接（可能需要运行: xhost +）"
    fi
fi

# 设置 XDG_RUNTIME_DIR（Open3D/GLFW 等需要）
if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR=/tmp/runtime-$(id -u)
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
    echo "✓ 设置 XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"
fi

# 设置 OpenCV 显示后端环境变量（如果需要）
export OPENCV_IO_ENABLE_OPENEXR=1

# 检查 X11 是否真的可用（用于判断是否需要禁用 Open3D 窗口）
x11_available=false
if [ -n "$DISPLAY" ]; then
    # 检查是本地还是远程显示
    if [[ "$DISPLAY" =~ ^: ]]; then
        # 本地显示，检查 socket
        display_num=${DISPLAY#*:}
        if [ -S "/tmp/.X11-unix/X${display_num}" ]; then
            x11_available=true
        fi
    else
        # 远程显示，尝试测试连接
        if command -v xdpyinfo &> /dev/null; then
            if timeout 2 xdpyinfo -display "$DISPLAY" &> /dev/null 2>&1; then
                x11_available=true
            fi
        else
            # 如果没有 xdpyinfo，假设可用（让应用自己处理失败）
            x11_available=true
        fi
    fi
fi

if [ "$x11_available" = false ]; then
    echo "⚠️  检测到 X11 不可用，设置环境变量以避免 Open3D GLX 错误"
    # 即使没有 X11，也设置一个假的 DISPLAY 避免某些库报错
    # 但 Open3D 在尝试创建窗口时会失败，应该被捕获
    # 设置 MESA 相关变量可能有助于避免某些 GLX 错误
    export MESA_GL_VERSION_OVERRIDE=3.3
    export LIBGL_ALWAYS_SOFTWARE=1
fi

# 设置 Vulkan 环境变量
# 如果 USE_SOFTWARE_VULKAN=1，强制使用软件渲染（CPU，不需要 GPU）
if [ "${USE_SOFTWARE_VULKAN:-0}" = "1" ]; then
    echo "使用软件 Vulkan 渲染（CPU，不需要 GPU）"
    export LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe
    export MESA_GL_VERSION_OVERRIDE=4.5
    export MESA_GLSL_VERSION_OVERRIDE=450
    # 禁用 Vulkan 验证层
    export VK_INSTANCE_LAYERS=""
    export VK_LOADER_DEBUG=warn
    # 尝试使用 Mesa 软件渲染
    mesa_icd="/usr/share/vulkan/icd.d/intel_icd.x86_64.json"
    if [ -f "$mesa_icd" ]; then
        export VK_ICD_FILENAMES="$mesa_icd"
        echo "使用 Mesa Vulkan ICD: $mesa_icd"
    else
        # 查找任何可用的 Mesa ICD
        available_icd=$(find /usr/share/vulkan/icd.d -name "*mesa*.json" -o -name "*intel*.json" -o -name "*lvp*.json" 2>/dev/null | head -1)
        if [ -n "$available_icd" ]; then
            export VK_ICD_FILENAMES="$available_icd"
            echo "使用找到的 Vulkan ICD: $available_icd"
        else
            echo "警告: 未找到 Mesa Vulkan ICD 文件"
            # 列出所有可用的 ICD 文件用于调试
            echo "可用的 Vulkan ICD 文件:"
            ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || echo "  /usr/share/vulkan/icd.d/ 目录不存在"
        fi
    fi
else
    # 尝试使用硬件加速（NVIDIA GPU）
    # NVIDIA Container Toolkit 应该挂载 Vulkan ICD 文件到 /usr/share/vulkan/icd.d/
    # 检查多个可能的路径
    nvidia_icd_paths=(
        "/usr/share/vulkan/icd.d/nvidia_icd.json"
        "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"
    )
    
    nvidia_icd_found=""
    for icd_path in "${nvidia_icd_paths[@]}"; do
        if [ -f "$icd_path" ]; then
            export VK_ICD_FILENAMES="$icd_path"
            echo "✓ 使用 NVIDIA Vulkan 驱动（硬件加速）: $icd_path"
            nvidia_icd_found="yes"
            break
        fi
    done
    
    if [ -z "$nvidia_icd_found" ]; then
        # 列出所有可用的 ICD 文件用于调试
        echo "⚠️  NVIDIA Vulkan ICD 文件未找到"
        echo "检查可用的 Vulkan ICD 文件:"
        if [ -d /usr/share/vulkan/icd.d/ ]; then
            ls -la /usr/share/vulkan/icd.d/ 2>/dev/null || true
            # 尝试使用找到的任何 ICD 文件
            available_icd=$(find /usr/share/vulkan/icd.d -name "*.json" 2>/dev/null | head -1)
            if [ -n "$available_icd" ]; then
                export VK_ICD_FILENAMES="$available_icd"
                echo "使用找到的 Vulkan ICD: $available_icd"
            else
                echo "❌ 未找到任何 Vulkan ICD 文件"
                echo "提示: 确保 NVIDIA Container Toolkit 正确安装并配置"
            fi
        else
            echo "  /usr/share/vulkan/icd.d/ 目录不存在"
            echo "提示: NVIDIA Container Toolkit 可能未正确挂载 Vulkan ICD 文件"
        fi
    fi
fi

export VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d

# 诊断信息：显示 Vulkan 配置
echo ""
echo "Vulkan 配置诊断:"
echo "  - VK_ICD_FILENAMES: ${VK_ICD_FILENAMES:-未设置}"
echo "  - VK_LAYER_PATH: ${VK_LAYER_PATH}"
if [ -n "$VK_ICD_FILENAMES" ] && [ -f "$VK_ICD_FILENAMES" ]; then
    echo "  - ICD 文件存在: ✓"
    echo "  - ICD 文件内容:"
    head -5 "$VK_ICD_FILENAMES" 2>/dev/null | sed 's/^/    /' || true
else
    echo "  - ICD 文件存在: ✗"
fi
echo ""

# 创建结果目录
mkdir -p "${LOGGING_DIR:-/workspace/results}"

# 显示配置信息
echo "配置信息:"
echo "  - 环境名称: ${ENV_NAME:-PutCarrotOnPlateInScene-v0}"
echo "  - 场景名称: ${SCENE_NAME:-bridge_table_1_v1}"
echo "  - 机器人: ${ROBOT:-widowx}"
echo "  - 初始位置: (${ROBOT_INIT_X:-0.147}, ${ROBOT_INIT_Y:-0.028})"
echo "  - 物体 Episode ID: ${OBJ_EPISODE_ID:-0}"
echo "  - 相机: ${OBS_CAMERA_NAME:-3rd_view_camera}"
echo "  - 最大步数: ${MAX_EPISODE_STEPS:-200}"
echo "  - 日志目录: ${LOGGING_DIR:-/workspace/results}"
echo "  - DISPLAY: ${DISPLAY}"
echo ""

# 运行画线执行脚本
echo "启动画线执行任务..."
echo "=========================================="
echo ""
# "${ENV_NAME:-PutCarrotOnPlateInScene-v0}" \
# PutSpoonOnTableClothInScene-v0
# PutEggplantInBasketScene-v0
# StackGreenCubeOnYellowCubeBakedTexInScene-v0
python draw_line_execute.py \
    --env_name "StackGreenCubeOnYellowCubeBakedTexInScene-v0" \
    --scene_name "${SCENE_NAME:-bridge_table_1_v1}" \
    --robot "${ROBOT:-widowx}" \
    --robot_init_x "${ROBOT_INIT_X:-0.147}" \
    --robot_init_y "${ROBOT_INIT_Y:-0.028}" \
    --obj_episode_id "${OBJ_EPISODE_ID:-0}" \
    --obs_camera_name "${OBS_CAMERA_NAME:-3rd_view_camera}" \
    --max_episode_steps "${MAX_EPISODE_STEPS:-200}" \
    --logging_dir "${LOGGING_DIR:-/workspace/results}"

echo ""
echo "=========================================="
echo "任务完成"
echo "=========================================="
