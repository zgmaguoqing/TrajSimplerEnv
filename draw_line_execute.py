"""
画线执行任务脚本
允许用户在图像上画一条线，然后机器人会沿着这条线执行任务
现在直接复用 fsd_execution 的轨迹执行逻辑
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# 设置 Vulkan 环境变量（在导入 sapien 之前）
# 优先尝试使用软件渲染（SwiftShader），避免 GPU 依赖
# 如果 GPU Vulkan 可用，也可以使用硬件加速

# 检查是否强制使用软件渲染
USE_SOFTWARE_RENDERING = os.environ.get("USE_SOFTWARE_VULKAN", "0") == "1"

if USE_SOFTWARE_RENDERING:
    print("使用软件 Vulkan 渲染（CPU，不需要 GPU）")
    # SwiftShader 路径（如果已安装）
    # 注意：需要先安装 SwiftShader
    swiftshader_icd = "/usr/share/vulkan/icd.d/vk_swiftshader_icd.json"
    if os.path.exists(swiftshader_icd):
        os.environ["VK_ICD_FILENAMES"] = swiftshader_icd
    else:
        print("警告: SwiftShader 未找到，尝试使用 Mesa 软件渲染")
        # 使用 Mesa 软件渲染 - 设置所有必要的环境变量
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["GALLIUM_DRIVER"] = "llvmpipe"
        os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.5"
        os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "450"
        # 尝试使用 Mesa Vulkan 软件渲染
        import glob
        mesa_icd_files = glob.glob("/usr/share/vulkan/icd.d/*mesa*.json") + \
                        glob.glob("/usr/share/vulkan/icd.d/*intel*.json") + \
                        glob.glob("/usr/share/vulkan/icd.d/*lvp*.json")
        if mesa_icd_files:
            os.environ["VK_ICD_FILENAMES"] = mesa_icd_files[0]
            print(f"使用 Mesa Vulkan ICD: {mesa_icd_files[0]}")
        else:
            # 如果找不到 Mesa ICD，尝试任何可用的 ICD
            all_icd_files = glob.glob("/usr/share/vulkan/icd.d/*.json")
            if all_icd_files:
                os.environ["VK_ICD_FILENAMES"] = all_icd_files[0]
                print(f"使用找到的 Vulkan ICD: {all_icd_files[0]}")
else:
    # 尝试使用硬件加速（NVIDIA GPU）
    if not os.environ.get("VK_ICD_FILENAMES"):
        # 检查多个可能的 NVIDIA ICD 路径
        nvidia_icd_paths = [
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json",
        ]
        
        nvidia_icd_found = False
        for icd_path in nvidia_icd_paths:
            if os.path.exists(icd_path):
                os.environ["VK_ICD_FILENAMES"] = icd_path
                print(f"✓ 使用 NVIDIA Vulkan 驱动（硬件加速）: {icd_path}")
                nvidia_icd_found = True
                break
        
        if not nvidia_icd_found:
            # 尝试查找任何可用的 ICD
            import glob
            icd_files = glob.glob("/usr/share/vulkan/icd.d/*.json")
            if icd_files:
                os.environ["VK_ICD_FILENAMES"] = icd_files[0]
                print(f"使用找到的 Vulkan ICD: {icd_files[0]}")
            else:
                print("⚠️  警告: 未找到可用的 Vulkan ICD 文件")
                print("提示: 确保 NVIDIA Container Toolkit 正确安装并配置")
                # 不设置软件渲染，让 Vulkan 尝试使用默认配置

if not os.environ.get("VK_LAYER_PATH"):
    os.environ["VK_LAYER_PATH"] = "/usr/share/vulkan/explicit_layer.d"

# 禁用 Vulkan 验证层以避免初始化问题（软件渲染时）
if USE_SOFTWARE_RENDERING:
    os.environ["VK_LOADER_DEBUG"] = "warn"  # 减少调试输出
    # 尝试禁用验证层
    if "VK_INSTANCE_LAYERS" not in os.environ:
        os.environ["VK_INSTANCE_LAYERS"] = ""

import cv2
import numpy as np

from simpler_env.utils.env.env_builder import build_maniskill2_env
from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
    get_depth_from_maniskill2_obs_dict,
)
from simpler_env.utils.visualization import write_video

# 直接复用 fsd_execution 函数
from simpler_env.evaluation.maniskill2_evaluator_fsd_widowx import fsd_execution

# 导入点云过滤所需的模块
import torch
import open3d as o3d
from plan.src.utils.robot_model import RobotModel
from plan.src.utils.utils import to_torch
from plan.src.utils.constants import ARM_URDF_FULL_WIDOWX, ROBOT_JOINTS_WIDOWX
import plan.src.utils.config as config


def filter_pc(robot_urdf, sce_pts_base, obs, robot_joints):
    """
    过滤掉场景点云中与机器人自身重叠的点，避免碰撞检测时误判。
    参考 maniskill2_evaluator_sofar_widowx.py 中的实现。
    """
    rm = RobotModel(robot_urdf)
    init_qpos = to_torch(obs['agent']['qpos'][None]).float()
    init_qpos = {k: init_qpos[:, i] for i, k in enumerate(robot_joints)}
    robot_pc, link_trans, link_rot, link_pc = rm.sample_surface_points_full(init_qpos, n_points_each_link=2**11, with_fk=True)
    robot_pc = robot_pc[0]
    # 确保 robot_pc 是 numpy 数组（open3d 需要）
    if isinstance(robot_pc, torch.Tensor):
        robot_pc = robot_pc.cpu().numpy()
    state_pc = o3d.geometry.PointCloud()
    state_pc.points = o3d.utility.Vector3dVector(sce_pts_base)
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_pc)
    kd_tree = o3d.geometry.KDTreeFlann(state_pc)
    indices_to_remove = []
    for point in robot_pcd.points:
        [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius=0.05)
        indices_to_remove.extend(idx)
    state_pc = state_pc.select_by_index(indices_to_remove, invert=True)
    scene_pc_filter = torch.tensor(np.asarray(state_pc.points))

    return scene_pc_filter


def fsd_execution_with_pc_filter(images, env, obs, obs_camera_name, task_description, 
                                  additional_env_build_kwargs, env_reset_options,
                                  pick_goal_uvd=None, place_goal_uvd=None):
    """
    包装 fsd_execution 函数，在规划前过滤点云以避免自身碰撞误判。
    通过 monkey patching config.DotDict 来拦截点云配置的创建。
    """
    # 保存原始的 DotDict
    original_DotDict = config.DotDict
    
    # 创建一个继承自 DotDict 的类，在创建配置时过滤点云
    class FilteredDotDict(original_DotDict):
        def __init__(self, *args, **kwargs):
            # 如果配置中包含 'pc' 参数，且它是点云数据，则过滤它
            if 'pc' in kwargs and isinstance(kwargs['pc'], torch.Tensor):
                pc_data = kwargs['pc']
                # 检查是否是点云数据（形状应该是 [N, 3]）
                if len(pc_data.shape) == 2 and pc_data.shape[1] == 3:
                    # 过滤点云
                    pc_numpy = pc_data.cpu().numpy() if isinstance(pc_data, torch.Tensor) else pc_data
                    filtered_pc = filter_pc(ARM_URDF_FULL_WIDOWX, pc_numpy, obs, ROBOT_JOINTS_WIDOWX)
                    kwargs['pc'] = filtered_pc
                    print(f"✓ 已过滤点云: {len(pc_numpy)} -> {len(filtered_pc)} 个点")
            
            # 调用父类的 __init__
            super().__init__(*args, **kwargs)
    
    # Monkey patch
    config.DotDict = FilteredDotDict
    
    try:
        # 调用原始的 fsd_execution 函数
        result = fsd_execution(images, env, obs, obs_camera_name, task_description,
                              additional_env_build_kwargs, env_reset_options,
                              pick_goal_uvd=pick_goal_uvd, place_goal_uvd=place_goal_uvd)
        return result
    finally:
        # 恢复原始的 DotDict
        config.DotDict = original_DotDict


def select_pick_and_place_points(image, depth, task_description="Select pick and place points"):
    """
    让用户通过画线选择 pick 和 place 点（替代 VLM 自动选择）
    用户可以画两条线：
    - 第一条线：用于选择 pick 点（线的起点或终点）
    - 第二条线：用于选择 place 点（线的起点或终点）
    返回: (pick_goal_uvd, place_goal_uvd) 或 None（如果用户取消）
    """
    image_cv = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    display_image = image_cv.copy()
    
    pick_point = None  # (u, v, d)
    place_point = None  # (u, v, d)
    current_line = []  # 当前正在画的线
    drawing = False
    current_mode = "pick"  # "pick" 或 "place"
    
    def on_mouse(event, u, v, flags, param):
        nonlocal display_image, pick_point, place_point, current_line, drawing, current_mode
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_line = [(u, v)]
            display_image = image_cv.copy()
            # 重新绘制已选择的点
            if pick_point:
                cv2.circle(display_image, (pick_point[0], pick_point[1]), 10, (0, 0, 255), -1)
                cv2.circle(display_image, (pick_point[0], pick_point[1]), 15, (0, 0, 255), 2)
                cv2.putText(display_image, "PICK", (pick_point[0]+15, pick_point[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if place_point:
                cv2.circle(display_image, (place_point[0], place_point[1]), 10, (255, 0, 0), -1)
                cv2.circle(display_image, (place_point[0], place_point[1]), 15, (255, 0, 0), 2)
                cv2.putText(display_image, "PLACE", (place_point[0]+15, place_point[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                current_line.append((u, v))
                # 绘制当前线
                if len(current_line) > 1:
                    color = (0, 0, 255) if current_mode == "pick" else (255, 0, 0)
                    for i in range(len(current_line) - 1):
                        cv2.line(display_image, current_line[i], current_line[i+1], color, 2)
                cv2.circle(display_image, (u, v), 3, (0, 255, 0), -1)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(current_line) > 0:
                # 使用线的起点作为目标点（也可以改为终点或中点）
                u, v = current_line[0]  # 使用起点
                # 或者使用终点: u, v = current_line[-1]
                # 或者使用中点: u, v = current_line[len(current_line)//2]
                
                if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
                    d = depth[v, u].item()
                    if d > 0:  # 有效的深度点
                        if current_mode == "pick":
                            pick_point = (u, v, d)
                            print(f"[OK] Selected Pick point: ({u}, {v}, {d:.4f})")
                            # 绘制 pick 点
                            cv2.circle(display_image, (u, v), 10, (0, 0, 255), -1)
                            cv2.circle(display_image, (u, v), 15, (0, 0, 255), 2)
                            cv2.putText(display_image, "PICK", (u+15, v-15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # 切换到 place 模式
                            if place_point is None:
                                current_mode = "place"
                                print("Now please draw a line to select Place point (blue)")
                        else:  # place mode
                            place_point = (u, v, d)
                            print(f"[OK] Selected Place point: ({u}, {v}, {d:.4f})")
                            # 绘制 place 点
                            cv2.circle(display_image, (u, v), 10, (255, 0, 0), -1)
                            cv2.circle(display_image, (u, v), 15, (255, 0, 0), 2)
                            cv2.putText(display_image, "PLACE", (u+15, v-15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            # 如果两个点都选好了，保持 place 模式（可以重新选择）
                    else:
                        print(f"Warning: Invalid depth at point ({u}, {v}) (d={d})")
                current_line = []
    
    cv2.namedWindow(task_description, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(task_description, on_mouse)
    
    print("\n" + "="*60)
    print("Please draw lines to select Pick and Place points (replacing VLM)")
    print("="*60)
    print("Instructions:")
    print("  1. Hold left mouse button and draw a line to select Pick point (grasp position) - red line")
    print("     - The start point of the line will be used as Pick point")
    print("  2. Hold left mouse button and draw a line to select Place point (place position) - blue line")
    print("     - The start point of the line will be used as Place point")
    print("  3. Press SPACE to confirm and start execution")
    print("  4. Press 'r' to reset selection")
    print("  5. Press 'p' to switch selection mode (Pick/Place)")
    print("  6. Press 'q' to quit")
    print("="*60 + "\n")
    print(f"Current mode: {current_mode.upper()} (please draw a line)")
    
    while True:
        # 显示当前选择状态
        status_text = f"Mode: {current_mode.upper()} | "
        if pick_point:
            status_text += "Pick: OK "
        else:
            status_text += "Pick: X "
        if place_point:
            status_text += "Place: OK"
        else:
            status_text += "Place: X"
        
        # 在图像上显示状态
        display_with_status = display_image.copy()
        # 添加半透明背景
        overlay = display_with_status.copy()
        cv2.rectangle(overlay, (5, 5), (500, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_with_status, 0.4, 0, display_with_status)
        cv2.putText(display_with_status, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # 显示提示信息
        hint_text = f"Draw line to select {current_mode.upper()} point (Space: confirm, r: reset, p: switch, q: quit)"
        cv2.putText(display_with_status, hint_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(task_description, display_with_status)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # 空格键确认
            if pick_point and place_point:
                print(f"\n[OK] Pick point: ({pick_point[0]}, {pick_point[1]}, {pick_point[2]:.4f})")
                print(f"[OK] Place point: ({place_point[0]}, {place_point[1]}, {place_point[2]:.4f})")
                print("Starting task execution...\n")
                break
            else:
                missing = []
                if not pick_point:
                    missing.append("Pick point")
                if not place_point:
                    missing.append("Place point")
                print(f"Please select first: {', '.join(missing)}")
        elif key == ord('r'):  # r键重新选择
            pick_point = None
            place_point = None
            current_mode = "pick"
            display_image = image_cv.copy()
            print("Cleared, please reselect (current mode: PICK)")
        elif key == ord('p'):  # p键切换选择模式
            if current_mode == "pick":
                current_mode = "place"
                print("Switched to Place mode (blue)")
            else:
                current_mode = "pick"
                print("Switched to Pick mode (red)")
        elif key == ord('q'):  # q键退出
            print("User cancelled")
            cv2.destroyAllWindows()
            return None
    
    # 用户确认，关闭窗口并返回选择的点
    cv2.destroyAllWindows()
    return pick_point, place_point
    #             print("切换到 Place 点选择模式")
    #         else:
    #             current_selection = "pick"
    #             print("切换到 Pick 点选择模式")
    
    # cv2.destroyAllWindows()
    # return pick_point, place_point


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="画线执行任务")
    # 默认使用 widowx 机器人和对应的环境
    parser.add_argument("--env_name", type=str, default="PutCarrotOnPlateInScene-v0", help="环境名称")
    parser.add_argument("--scene_name", type=str, default="bridge_table_1_v1", help="场景名称")
    parser.add_argument("--robot", type=str, default="widowx", help="机器人名称")
    parser.add_argument("--robot_init_x", type=float, default=0.147, help="机器人初始x坐标")
    parser.add_argument("--robot_init_y", type=float, default=0.028, help="机器人初始y坐标")
    parser.add_argument("--obj_episode_id", type=int, default=0, help="物体 episode ID")
    parser.add_argument("--obs_camera_name", type=str, default="3rd_view_camera", help="观察相机名称")
    parser.add_argument("--logging_dir", type=str, default="./results", help="日志目录")
    parser.add_argument("--max_episode_steps", type=int, default=200, help="最大步数")
    
    args = parser.parse_args()
    
    # 创建环境 - 使用 widowx 机器人
    control_mode = "arm_pd_joint_pos_gripper_pd_joint_pos"
    # 配置渲染器：使用 offscreen 模式，避免 Vulkan 显示问题
    additional_env_build_kwargs = {
        "renderer_kwargs": {
            "offscreen_only": True,  # 告诉渲染器不需要显示到屏幕
        }
    }
    
    # 尝试创建环境，如果 Vulkan 失败则提供有用的错误信息
    try:
        env = build_maniskill2_env(
        args.env_name,
        obs_mode="rgbd",
        robot=args.robot,
        sim_freq=513,
        control_mode=control_mode,
        control_freq=30,
        max_episode_steps=args.max_episode_steps,
        scene_name=args.scene_name,
        camera_cfgs={"add_segmentation": True},
        prepackaged_config=True,  # 启用预打包配置，确保物体正确加载
        # render_mode="human",
        **additional_env_build_kwargs,
    )
    except RuntimeError as e:
        if "Vulkan" in str(e) or "vk::" in str(e):
            print("\n" + "="*70)
            print("❌ Vulkan 渲染器初始化失败！")
            print("="*70)
            print(f"错误信息: {e}")
            print("\n🔧 解决方案:")
            print("\n【方案 1】修复主机的 Vulkan 支持（推荐）:")
            print("  1. 在主机上检查 Vulkan 支持:")
            print("     vulkaninfo --summary")
            print("  2. 如果失败，安装/更新 NVIDIA 驱动:")
            print("     - 确保驱动版本 >= 470.x（支持 Vulkan）")
            print("     - 重启系统")
            print("  3. 确保 NVIDIA Container Toolkit 正确安装:")
            print("     docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi")
            print("\n【方案 2】使用软件渲染（较慢，但不需要 GPU）:")
            print("  设置环境变量: USE_SOFTWARE_VULKAN=1")
            print("  然后重新运行容器")
            print("\n【方案 3】检查容器内的 Vulkan ICD 文件:")
            print("  docker-compose exec simpler-env bash")
            print("  ls -la /usr/share/vulkan/icd.d/")
            print("  vulkaninfo --summary")
            print("="*70 + "\n")
            
            # 如果设置了软件渲染但仍然失败，提供额外建议
            if USE_SOFTWARE_RENDERING:
                print("⚠️  注意: 即使设置了软件渲染，Vulkan 仍然无法初始化。")
                print("   这可能是因为:")
                print("   - Mesa Vulkan 驱动在容器中无法正常工作")
                print("   - 需要安装 SwiftShader（软件 Vulkan 实现）")
                print("   - 或者需要修复主机的 Vulkan 支持")
                print()
        raise
    
    # 初始化环境
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([args.robot_init_x, args.robot_init_y]),
            "init_rot_quat": np.array([1, 0, 0, 0]),
        },
        "obj_init_options": {
            "episode_id": args.obj_episode_id,
        }
    }
    obs, _ = env.reset(options=env_reset_options)
    # 执行初始动作（与 fsd_execution 保持一致）
    obs, reward, done, truncated, info = env.step(np.array([-0.01840777, -0.398835, -0.52242722, -0.00460194, 1.365243, 0.00153398, 0.037, 0.037]))
    
    # 获取任务描述
    task_description = env.get_language_instruction()
    print(f"Task: {task_description}")
    
    # 获取图像（用于可视化）
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
    depth = get_depth_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
    
    # 让用户通过可视化界面选择 pick 和 place 点（替代 VLM）
    print("\n" + "="*50)
    print("Please select Pick and Place points through the visualization interface")
    print("="*50 + "\n")
    
    result = select_pick_and_place_points(image, depth, task_description)
    
    if result is None:
        print("User cancelled, exiting program")
        return
    
    pick_goal_uvd, place_goal_uvd = result
    
    print("\n" + "="*50)
    print("Executing task with user-specified points...")
    print("="*50 + "\n")
    
    # 使用用户指定的 pick 和 place 点执行轨迹
    images = []
    print(f"Starting task execution, initial image count: {len(images)}")
    
    for attempt in range(1):
        images_before = len(images)
        # 使用带点云过滤的包装函数
        images, env, obs, done, info = fsd_execution_with_pc_filter(
            images, env, obs, args.obs_camera_name, task_description,
            additional_env_build_kwargs, env_reset_options,
            pick_goal_uvd=pick_goal_uvd,  # 传递用户指定的 pick 点
            place_goal_uvd=place_goal_uvd  # 传递用户指定的 place 点
        )
        images_after = len(images)
        print(f"Execution completed, saved {images_after - images_before} images (total: {images_after} images)")
        
        if done:
            print(f"Attempt {attempt + 1} succeeded")
            break
        else:
            print(f"Attempt {attempt + 1} not completed, continuing...")
    
    # 最后再获取一张图像
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
    images.append(image)
    print(f"Added final image, total image count: {len(images)}")
    
    # 保存视频
    if len(images) > 0:
        os.makedirs(args.logging_dir, exist_ok=True)
        video_path = os.path.join(args.logging_dir, "draw_line_execute.mp4")
        print(f"\nSaving video, total {len(images)} frames...")
        write_video(video_path, images, fps=5)
        print(f"✓ Video saved to: {video_path} (total {len(images)} frames)")
    else:
        print("⚠️  Warning: No images to save!")
    
    success = "success" if done else "failure"
    print(f"\nTask completed, result: {success}")


if __name__ == "__main__":
    main()
