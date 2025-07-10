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

    def plan(self, start=None, goal=None, interpolate_num=None, fix_joints_value=dict(), time=None, first=None, only_test_start_end=False, save_pc=None):
        if start is None:
            start = [0,0,0,-1,0,1.5,0, 0.02, 0.02]
        if goal is None:
            goal = [1,0,0,-1,0,1.5,0, 0.02, 0.02]

        self.pb_ompl_interface.fix_joints_value = fix_joints_value
        start, goal = to_list(start), to_list(goal)
        for name, pose in [('start', start), ('goal', goal)]:
            if not self.pb_ompl_interface.is_state_valid(pose, save_pc=save_pc if name == "goal" else None):
                print(f'unreachable {name}')
                return False, None
        if only_test_start_end:
            return True, None

        # print("ok")
        # exit()

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
        # render_mode="human",
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
    # print("render"); env.render()
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
    for _ in range(3):
        images, env, obs, done, info = fsd_execution(images, env, obs, obs_camera_name, task_description, additional_env_build_kwargs, env_reset_options)
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
    # print("render"); env.render()
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

def maniskill2_evaluator_fsd_widowx(model, args):
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

from waypoint_fsd import get_waypoints
import time

def get_uvd(image, depth, task_description):
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    ts_str = time.strftime('%Y%m%d_%H%M%S')
    
    os.makedirs("cv_results", exist_ok=True)
    cv2.imwrite(f"cv_results/{ts_str} {task_description}.png", image_cv)

    cv2.imshow(task_description, image_cv)
    cv2.waitKey(1)
    
    (u, v), points = get_waypoints(image, "Select grasp point of: " + task_description + ".")

    for point in points:
        image_cv = cv2.circle(image_cv, (int(point[0]), int(point[1])), 5, (0, 0, 127), -1)
    image_cv = cv2.circle(image_cv, (int(u), int(v)), 5, (0, 0, 255), -1)
    d = depth[int(v), int(u)].item()
    pick_goal_uvd = (u, v, d)
    
    cv2.imshow(task_description, image_cv)
    cv2.waitKey(1)
    
    (u, v), points = get_waypoints(image, "Select place point of: " + task_description + ".")
    
    for point in points:
        image_cv = cv2.circle(image_cv, (int(point[0]), int(point[1])), 5, (127, 0, 0), -1)
    image_cv = cv2.circle(image_cv, (int(u), int(v)), 5, (255, 0, 0), -1)
    d = depth[int(v), int(u)].item()
    place_goal_uvd = (u, v, d)
    
    cv2.imwrite(f"cv_results/{ts_str} {task_description} result.png", image_cv)
    
    cv2.imshow(task_description, image_cv)
    cv2.waitKey(1)
    
    return pick_goal_uvd, place_goal_uvd

def get_xyz_from_uvd(u, v, d, intrinsic_matrix, depth_scale=1):
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

from GSNet.gsnet_simpler import GSNet
from SoFar.depth.metric_3d_v2.utils.unproj_pcd import reconstruct_pcd
from SoFar.depth.utils import transform_obj_pts
import traceback
import pickle as pkl

@dataclass
class Grasp:
    translation: np.ndarray
    rotation_matrix: np.ndarray

def fsd_execution(images, env, obs, obs_camera_name, task_description, additional_env_build_kwargs, env_reset_options):
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images.append(image)

    depth = get_depth_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    intrinsic, extrinsics = get_camera_extrinsics_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)

    base_pose = get_base_pose(obs)
    extrinsics = np.linalg.inv(np.array(extrinsics) @ np.array(base_pose)) # inv(world2camera @ base2world = base2camera) = camera2base
    
    # with open("extrinsics.pkl", "wb") as f:
    #     pkl.dump(extrinsics, f)

    pick_goal_uvd, place_goal_uvd = get_uvd(image, depth, task_description)
    pick_goal_xyz = get_xyz_from_uvd(*pick_goal_uvd, intrinsic_matrix=intrinsic)
    place_goal_xyz = get_xyz_from_uvd(*place_goal_uvd, intrinsic_matrix=intrinsic)
    
    # depth to pcd
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    pcd_camera = reconstruct_pcd(depth, fx, fy, cx, cy) # pcd_camera.shape == (512, 640, 3)
    pcd_camera = pcd_camera.reshape(-1, 3)
    
    pcd_base = transform_obj_pts(pcd_camera, extrinsics)

    gsnet = GSNet()
    gg = gsnet.inference(pcd_camera)
    # gsnet.visualize(pcd_camera[::1000,:], gg, save_image="grasp.png")

    # with open("pcd_camera.pkl", "wb") as f:
    #     pkl.dump(pcd_camera, f)
    # with open("gg.pkl", "wb") as f:
    #     pkl.dump(gg, f)

    gs = []
    for g in gg:
        # 限制抓取角度
        mat = extrinsics[:3, :3] @ g.rotation_matrix
        z_dir = -mat[:, 0]
        up = np.array([0, 0, 1])
        angle = np.arccos(np.clip(np.dot(z_dir, up), -1.0, 1.0)) 
        
        angle_z = 60
        if not angle <= np.deg2rad(angle_z):
            continue
        
        dist = ((g.translation - pick_goal_xyz) ** 2).sum() ** 0.5
        if dist > 0.1:
            continue
        
        gs.append((
            dist, # select nearest grasp
            g
        ))
    sorted_gs = list(sorted(gs, key=lambda x: x[0]))[:20]
    
    planned = False
    for idx, (_, pick_grasp) in enumerate(sorted_gs):
        try:
            print(f"try grasp {idx}: ", pick_grasp)
            # gsnet.visualize(pcd_camera[::1000,:], g=pick_grasp, save_image=f"grasp_{idx}.png")
            
            # with open("g.pkl", "wb") as f:
            #     pkl.dump(pick_grasp, f)

            point2camera = np.eye(4)
            point2camera[:3, 3] = pick_grasp.translation
            point2camera[:3, :3] = pick_grasp.rotation_matrix
            pick_goal_T = extrinsics @ point2camera
            pick_goal_T = pick_goal_T @ [
                [1, 0, 0, -0.05],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
                        
            point2camera = np.eye(4)
            point2camera[:3, 3] = place_goal_xyz
            place_goal_T = extrinsics @ point2camera
            place_goal_T[:3, :3] = pr.matrix_from_euler((0, math.pi/2, 0), 0, 1, 2, True)
            place_goal_T = place_goal_T @ [
                [1, 0, 0, -0.05],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
            
            # with open("goal_T.pkl", "wb") as f:
            #     pkl.dump((pick_goal_T, place_goal_T), f)

            print("Star Planning First Phase")
            
            ik = IK(robot='widowx')
            cfg = config.DotDict(
                urdf=ARM_URDF_FULL_WIDOWX,
                # pc=torch.tensor(np.zeros((0, 3))),
                pc=torch.tensor(pcd_base),
                curobo_file="./plan/robot_models/widowx/curobo/widowx.yml",
                robot_type='widowx',
            )
            
            init_qpos = np.array(obs['agent']['qpos'][:6])

            grasp_point = None
            # for i in range(5):
            #     planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
            #     if not planner.pb_ompl_interface.is_state_valid(grasp_point[:6]):
            for try_from_qpos in [
                init_qpos,
                init_qpos,
                init_qpos
            ]:
                try:
                    grasp_point = ik.ik(pick_goal_T[:3, 3], pick_goal_T[:3, :3], joints=ik.robot_to_arm_joints(try_from_qpos))
                    if grasp_point is not None:
                        break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("Failed IK")
            assert grasp_point is not None # TODO Fails
            print("Grasp Path IK Solved", grasp_point)

            mid1_point = None
            for try_from_qpos in [
                init_qpos,
                init_qpos,
                init_qpos
            ]:
                try:
                    mid1_point = ik.ik(pick_goal_T[:3, 3] + [0, 0, 0.15], pick_goal_T[:3, :3], joints=ik.robot_to_arm_joints(try_from_qpos))
                    if mid1_point is not None:
                        break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("Failed IK")
            assert mid1_point is not None
            print("Mid1 Path IK Solved", mid1_point)
            
            mid2_point = None
            for try_from_qpos in [
                init_qpos,
                init_qpos,
                init_qpos
            ]:
                try:
                    mid2_point = ik.ik(place_goal_T[:3, 3] + [0, 0, 0.15], place_goal_T[:3, :3], joints=ik.robot_to_arm_joints(try_from_qpos))
                    if mid2_point is not None:
                        break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("Failed IK")
            assert mid2_point is not None  # TODO Fails
            print("Mid2 Path IK Solved", mid2_point)

            place_point = None
            for try_from_qpos in [
                init_qpos,
                init_qpos,
                init_qpos
            ]:
                try:
                    place_point = ik.ik(place_goal_T[:3, 3] + [0, 0, 0.04], place_goal_T[:3, :3], joints=ik.robot_to_arm_joints(try_from_qpos))
                    if place_point is not None:
                        break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("Failed IK")
            assert place_point is not None
            print("Place Path IK Solved", place_point)

            for _ in range(2):
                planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
                res, grasp_path = planner.plan(init_qpos, grasp_point[:6], interpolate_num=100, fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037}, 
                                            #    save_pc="pcd.pkl"
                                               )
                if res:
                    break
            # import IPython; IPython.embed()
            assert res and grasp_path is not None and isinstance(grasp_path, np.ndarray)
            print('Grasp Path Completed')

            for _ in range(2):
                planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
                res, mid_path1 = planner.plan(grasp_point[:6], mid1_point[:6], interpolate_num=100, fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037})
                if res:
                    break
            assert res and mid_path1 is not None and isinstance(mid_path1, np.ndarray)
            print('Mid1 Path Completed')

            for _ in range(2):
                planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
                res, mid_path2 = planner.plan(mid1_point[:6], mid2_point[:6], interpolate_num=100, fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037})
                if res:
                    break
            assert res and mid_path2 is not None and isinstance(mid_path2, np.ndarray)
            print('Mid2 Path Completed')

            for _ in range(2):
                planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
                res, place_path = planner.plan(mid2_point[:6], place_point[:6], interpolate_num=100, fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037})
                if res:
                    break
            assert res and place_path is not None and isinstance(place_path, np.ndarray)
            print('Place Path Completed')

            num_copies = 5
            repeated_elements = np.tile(grasp_path[-1], (num_copies, 1))
            for index in range(num_copies):
                repeated_elements[index, -2:] = [0.037-index/num_copies*0.025, 0.037-index/num_copies*0.025]
            grasp_path = np.vstack([grasp_path, repeated_elements])
            
            mid_path1[:, -2:] = [0.010, 0.010]
            
            mid_path2[:, -2:] = [0.010, 0.010]

            place_path[:, -2:] = [0.010, 0.010]

            num_copies = 5
            repeated_elements = np.tile(place_path[-1], (num_copies, 1))
            for index in range(num_copies):
                repeated_elements[index, -2:] = [0.015+index/num_copies*0.022, 0.015+index/num_copies*0.022]
            place_path = np.vstack([place_path, repeated_elements])
            
            print("Path Planning Done")
            
            planned = True
            break
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("Failed try")
    
    # assert planned
    if not planned: 
        print("Failed to plan")
        return images, env, obs, None, None

    # import IPython; IPython.embed()

    for index in range(len(grasp_path)):
        obs, reward, done, truncated, info = env.step(grasp_path[index])
        # print("render grasp"); env.render()
        img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(img)
        
    # import IPython; IPython.embed()

    for index in range(len(mid_path1)):
        obs, reward, done, truncated, info = env.step(mid_path1[index])
        # print("render path1"); env.render()
        img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(img)
        
    # import IPython; IPython.embed()

    for index in range(len(mid_path2)):
        obs, reward, done, truncated, info = env.step(mid_path2[index])
        # print("render path2"); env.render()
        img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(img)

    # import IPython; IPython.embed()

    for index in range(len(place_path)):
        obs, reward, done, truncated, info = env.step(place_path[index])
        # print("render place"); env.render()
        img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(img)

    # import IPython; IPython.embed()

    return images, env, obs, done, info
