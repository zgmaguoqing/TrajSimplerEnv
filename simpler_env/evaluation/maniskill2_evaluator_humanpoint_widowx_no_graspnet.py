"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import plan.src.utils.config as config
import torch
import numpy as np
from transforms3d.euler import quat2euler
import open3d as o3d

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict, get_depth_from_maniskill2_obs_dict, \
                                                    get_camera_extrinsics_from_maniskill2_obs_dict, get_pointcloud_in_camera, get_base_pose
from simpler_env.utils.visualization import write_video

from PIL import Image
from copy import deepcopy
from transforms3d.euler import euler2axangle
from transforms3d.quaternions import axangle2quat, quat2axangle, mat2quat
import json
from scipy.spatial.transform import Rotation as R
from SoFar.depth.utils import transform_point_cloud_nohw, inverse_transform_point_cloud

from plan.src.plan import pb_ompl
from plan.src.utils.vis_plotly import Vis
from plan.src.utils.config import DotDict
from plan.src.utils.utils import to_list, to_torch
from plan.src.utils.robot_model import RobotModel
from plan.src.utils.ik import IK
from plan.src.utils.vis_plotly import Vis
from plan.src.utils.constants import ARM_URDF_FULL_GOOGLE_ROBOT, ARM_URDF_FULL_WIDOWX, ROBOT_JOINTS_WIDOWX, ROBOT_JOINTS_GOOGLEROBOT, FRANKA_COLLISION_FILE, FRANKA_CUROBO_FILE

import cv2
import numpy as np
import math
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import json
import os
from dataclasses import dataclass

class Planner:
    def __init__(self, config, fix_joints=[], planner="RRTConnect"):
        self.config = config

        # load robot
        robot = RobotModel(config.urdf)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, config, fix_joints=fix_joints)
        self.pb_ompl_interface.set_planner(planner)

    def clear_obstacles(self):
        raise NotImplementedError
        self.obstacles = []

    def plan(self, start=None, goal=None, interpolate_num=None, fix_joints_value=dict(), time=None, first=None, only_test_start_end=False):
        if start is None:
            start = [0,0,0,-1,0,1.5,0, 0.02, 0.02]
        if goal is None:
            goal = [1,0,0,-1,0,1.5,0, 0.02, 0.02]

        self.pb_ompl_interface.fix_joints_value = fix_joints_value
        start, goal = to_list(start), to_list(goal)
        for name, pose in [('start', start), ('goal', goal)]:
            if not self.pb_ompl_interface.is_state_valid(pose):
                print(f'unreachable {name}')
                return False, None
        if only_test_start_end:
            return True, None

        res, path = self.pb_ompl_interface.plan(start, goal, interpolate_num=interpolate_num, fix_joints_value=fix_joints_value, allowed_time=time, first=first)
        if res:
            path = np.array(path)
        return res, path

    def close(self):
        pass

def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=30,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    control_mode = "arm_pd_joint_pos_gripper_pd_joint_pos"

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
        render_mode="human",
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    obs, reward, done, truncated, info = env.step(np.array([-0.01840777,  -0.398835 ,  -0.52242722, -0.00460194,  1.365243  , 0.00153398, 0.037, 0.037]))
    print("render"); env.render()
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask()
    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)


    # Initialize model
    timestep = 0
    success = "failure"

    print('Task Start')
    images = []
    for _ in range(1):
        images, env, obs, done, info = humanpoint_execution(images, env, obs, obs_camera_name, task_description, additional_env_build_kwargs, env_reset_options)
        if done:
            break
        else:
            print("this time is not done")
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images.append(image)
    success = "success" if done else "failure"
    new_task_description = env.get_language_instruction()
    if new_task_description != task_description:
        task_description = new_task_description
        print(task_description)

    is_final_subtask = env.is_final_subtask()
    timestep += 1

    _, _, _, _, info = env.step(np.zeros(8))
    print("render"); env.render()
    episode_stats = info.get("episode_stats", {})
    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    ckpt_path_basename = 'motion_planning'
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)
    print('=============video save================')
    print(video_path)
    print('=============video save================')
    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    return success == "success"

def maniskill2_evaluator_humanpoint_widowx(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []
    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr

def get_uvd(image, depth, task_description):
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow(task_description, image_cv)
    cv2.waitKey(1)
    
    print("select grasp point on image, ", task_description)

    grasp_goal_uvd = None
    
    def on_mouse(event, u, v, flags, param):
        nonlocal image_cv, grasp_goal_uvd
        
        # if clicked, goto the position
        if event == cv2.EVENT_LBUTTONDOWN:
            image_cv = cv2.circle(image_cv, (u, v), 5, (0, 0, 255), -1)
            d = depth[v, u].item()
            grasp_goal_uvd = (u, v, d)
            
    cv2.setMouseCallback(task_description, on_mouse)
    
    while grasp_goal_uvd is None:
        cv2.waitKey(1)
    
    cv2.imshow(task_description, image_cv)
    cv2.waitKey(1)
    
    print("select place point on image, ", task_description)

    place_goal_uvd = None
    
    def on_mouse(event, u, v, flags, param):
        nonlocal image_cv, place_goal_uvd
        
        # if clicked, goto the position
        if event == cv2.EVENT_LBUTTONDOWN:
            image_cv = cv2.circle(image_cv, (u, v), 5, (255, 0, 0), -1)
            d = depth[v, u].item()
            place_goal_uvd = (u, v, d)
            
    cv2.setMouseCallback(task_description, on_mouse)
    
    while place_goal_uvd is None:
        cv2.waitKey(1)
    
    cv2.imshow(task_description, image_cv)
    cv2.waitKey(1)
    
    return grasp_goal_uvd, place_goal_uvd

def get_xyz_from_uvd(u, v, d, intrinsic_matrix, depth_scale=1):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def get_xyz_in_arm_frame(uvd, intrinsic, extrinsics):
    x, y, z = get_xyz_from_uvd(*uvd, intrinsic_matrix=intrinsic)
    point2camera = np.array([[x, y, z, 1]]).T
    point2arm = extrinsics @ point2camera
    return point2arm[:3, 0]

def humanpoint_execution(images, env, obs, obs_camera_name, task_description, additional_env_build_kwargs, env_reset_options):
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images.append(image)

    depth = get_depth_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    intrinsic, extrinsics = get_camera_extrinsics_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)

    base_pose = get_base_pose(obs)
    extrinsics = np.linalg.inv(np.array(extrinsics) @ np.array(base_pose)) # inv(world2camera @ base2world = base2camera) = camera2base

    grasp_goal_uvd, place_goal_uvd = get_uvd(image, depth, task_description)

    grasp_point2arm = get_xyz_in_arm_frame(grasp_goal_uvd, intrinsic, extrinsics)
    place_point2arm = get_xyz_in_arm_frame(place_goal_uvd, intrinsic, extrinsics)
    
    @dataclass
    class Grasp:
        translation: np.ndarray
        rotation_matrix: np.ndarray
        
    gg = Grasp(translation=grasp_point2arm, rotation_matrix=pr.matrix_from_euler((0, math.pi/2, 0), 0, 1, 2, True))
    gg_goal = Grasp(translation=place_point2arm, rotation_matrix=pr.matrix_from_euler((0, math.pi/2, 0), 0, 1, 2, True))

    robot_urdf = ARM_URDF_FULL_WIDOWX
    cfg = config.DotDict(
        urdf=robot_urdf,
        pc=torch.tensor(np.zeros((0, 3))),
        curobo_file="./plan/robot_models/widowx/curobo/widowx.yml",
        robot_type='widowx',
    )

    print("Star Planning First Phase")

    robot_state = np.array(obs['agent']['qpos'])
    init = np.array(obs['agent']['qpos'][:6])

    ik = IK(robot='widowx')
    goal = ik.ik(gg.translation, gg.rotation_matrix, joints=ik.robot_to_arm_joints(init))
    for _ in range(3):
        goal = ik.ik(gg.translation, gg.rotation_matrix, joints=ik.robot_to_arm_joints(init))
        if goal is not None:
            break
    if goal is None:
        print("Grasp Path IK No Solution")
        return images, env, obs, None, None
    print("Grasp Path IK Solved")

    print('Mid IK Starting')
    mid1_init_qpos = goal
    for _ in range(3):
        mid1_point = ik.ik(gg.translation+[0, 0, 0.15], gg.rotation_matrix, joints=ik.robot_to_arm_joints(init))
        if mid1_point is not None:
            break
    if mid1_point is None:
        print("Mid Path IK No Solution")
        return images, env, obs, None, None

    for _ in range(3):
        mid2_point = ik.ik(gg_goal.translation+[0, 0, 0.15], gg_goal.rotation_matrix, joints=ik.robot_to_arm_joints(init))
        if mid2_point is not None:
            break
    if mid2_point is None:
        print("Mid Path IK No Solution")
        return images, env, obs, None, None

    print('Place IK Starting')
    place_init_qpos = goal
    for _ in range(3):
        place_goal_qpos = ik.ik(gg_goal.translation+[0, 0, 0.04], gg_goal.rotation_matrix, joints=ik.robot_to_arm_joints(init))
        if place_goal_qpos is not None:
            break
    if place_goal_qpos is None:
        print("Place Path IK No Solution")
        return images, env, obs, None, None
    print("Place Path IK Solved")

    for _ in range(2):
        planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
        res_grasp, grasp_path = planner.plan(robot_state[:6], goal, interpolate_num=30,
                                            fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037})
        if res_grasp:
            print('Grasp Path Completed')
            break
    if grasp_path is None or res_grasp == False:
        return images, env, obs, None, None

    for _ in range(2):
        planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
        res_mid1, mid_path1 = planner.plan(mid1_init_qpos[:6], mid1_point[:6], interpolate_num=30, fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037})
        if res_mid1:
            print('Mid1 Path Completed')
            break
    if mid_path1 is None or res_mid1==False:
        return images, env, obs, None, None

    for _ in range(2):
        planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
        res_mid2, mid_path2 = planner.plan(mid1_point[:6], mid2_point[:6], interpolate_num=30, fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037})
        if res_mid2:
            print('Mid2 Path Completed')
            break
    if mid_path2 is None or res_mid2==False: #or not isinstance(place_path, np.ndarray)
        return images, env, obs, None, None

    for _ in range(2):
        planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
        res_place, place_path = planner.plan(mid2_point[:6], place_goal_qpos[:6], interpolate_num=30, fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037})
        if res_place:
            print('Place Path Completed')
            break
    if place_path is None or res_place == False: # or not isinstance(place_path, np.ndarray)
        return images, env, obs, None, None
    
    import IPython; IPython.embed()

    try:
        if isinstance(grasp_path, np.ndarray):
            num_copies = 5
            repeated_elements = np.tile(grasp_path[-1], (num_copies, 1))
            for index in range(num_copies):
                repeated_elements[index, -2:] = [0.037-index/num_copies*0.025, 0.037-index/num_copies*0.025]
            grasp_path = np.vstack([grasp_path, repeated_elements])

            for index in range(len(grasp_path)):
                obs, reward, done, truncated, info = env.step(grasp_path[index])
                print("render grasp"); env.render()
                img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
                images.append(img)

            # import IPython; IPython.embed()

            mid_path1[:, -2:] = [0.010, 0.010]
            for index in range(len(mid_path1)):
                obs, reward, done, truncated, info = env.step(mid_path1[index])
                print("render path1"); env.render()
                img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
                images.append(img)

            # import IPython; IPython.embed()

            mid_path2[:, -2:] = [0.010, 0.010]
            for index in range(len(mid_path2)):
                obs, reward, done, truncated, info = env.step(mid_path2[index])
                print("render path2"); env.render()
                img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
                images.append(img)

            # import IPython; IPython.embed()
        else:
            return images, env, obs, None, None
    except:
        return images, env, obs, None, None

    try:
        if isinstance(place_path, np.ndarray):
            place_path[:, -2:] = [0.010, 0.010]
            num_copies = 5
            repeated_elements = np.tile(place_path[-1], (num_copies, 1))
            for index in range(num_copies):
                repeated_elements[index, -2:] = [0.015+index/num_copies*0.022, 0.015+index/num_copies*0.022]
            place_path = np.vstack([place_path, repeated_elements])

            for index in range(len(place_path)):
                obs, reward, done, truncated, info = env.step(place_path[index])
                print("render place"); env.render()
                img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
                images.append(img)
                
            # import IPython; IPython.embed()
        else:
            return images, env, obs, None, None
    except:
        return images, env, obs, None, None

    return images, env, obs, done, info
