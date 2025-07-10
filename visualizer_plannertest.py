from simpler_env.evaluation.maniskill2_evaluator_fsd_widowx import *

if __name__ == "__main__":
    # pcd_camera = o3d.io.read_point_cloud("pcd.ply")
    # pcd_camera = np.asarray(pcd_camera.points)
    
    import pickle as pkl
    pcd_camera = pkl.load(open("pcd_camera.pkl", "rb"))
    
    robot_urdf = ARM_URDF_FULL_WIDOWX
    cfg = config.DotDict(
        urdf=robot_urdf,
        # pc=torch.tensor(np.zeros((0, 3))),
        pc=torch.tensor(pcd_camera),
        curobo_file="./plan/robot_models/widowx/curobo/widowx.yml",
        robot_type='widowx',
    )
    
    # import pickle as pkl
    # with open("args.pkl", "wb") as f:
    #     pkl.dump((init_qpos, grasp_point[:6]), f)
    
    import pickle as pkl
    with open("args.pkl", "rb") as f:
        init_qpos, grasp_point = pkl.load(f)
        
    # import IPython; IPython.embed()
    
    planner = Planner(cfg, planner='AITstar', fix_joints=['left_finger', 'right_finger'])
    res, grasp_path = planner.plan(init_qpos, grasp_point, interpolate_num=100, fix_joints_value={'left_finger': 0.037, 'right_finger': 0.037})
    
    import IPython; IPython.embed()