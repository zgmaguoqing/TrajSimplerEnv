import open3d as o3d
import numpy as np
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import pickle as pkl

def create_coordinate_frame(translation, rotation, scale=0.1):
    # Create coordinate frame mesh
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    
    # Transform frame to desired pose
    frame.translate(translation)
    frame.rotate(rotation, center=translation)
    
    return frame

from SoFar.depth.utils import transform_obj_pts

# Example usage
if __name__ == "__main__":
    with open("pcd.pkl", "rb") as f:
        pcd_points, robot_pcd_points = pkl.load(f)
        
    with open("extrinsics.pkl", "rb") as f:
        extrinsics = pkl.load(f)
        
    pcd_points = transform_obj_pts(pcd_points, np.linalg.inv(extrinsics))
    robot_pcd_points = transform_obj_pts(robot_pcd_points, np.linalg.inv(extrinsics))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_pcd_points)

    # Example EEF poses (translation + euler angles)
    eef_poses_euler = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # x,y,z, rx,ry,rz
        # np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),  # x,y,z, rx,ry,rz
        # np.array([0.0, 0.5, 0.0, np.pi/2, 0.0, 0.0])
    ]
    
    eef_poses = []
    for pose in eef_poses_euler:
        translation = pose[:3]
        rotation = pose[3:]  # Assuming last 3 values are euler angles
        rotation = pr.matrix_from_euler(rotation, 0, 1, 2, True)
        eef_poses.append((translation, rotation))
        
    with open("g.pkl", "rb") as f:
        g = pkl.load(f)
    geoms = [g.to_open3d_geometry()]
    
    with open("goal_T.pkl", "rb") as f:
        pick_goal_T, place_goal_T = pkl.load(f)
    pick_goal_T = np.linalg.inv(extrinsics) @ pick_goal_T
    place_goal_T = np.linalg.inv(extrinsics) @ place_goal_T
        
    eef_poses.append((pick_goal_T[:3, 3], pick_goal_T[:3, :3]))
    eef_poses.append((place_goal_T[:3, 3], place_goal_T[:3, :3]))
    
    # with open("gg.pkl", "rb") as f:
    #     gg = pkl.load(f)
    # geoms = gg.to_open3d_geometry_list()

    combined_mesh = o3d.geometry.TriangleMesh()
    for geom in geoms:
        assert isinstance(geom, o3d.geometry.TriangleMesh)
        combined_mesh += geom
    
    # Load point cloud from file    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add point cloud to visualizer
    vis.add_geometry(pcd)
    vis.add_geometry(robot_pcd)
    vis.add_geometry(combined_mesh)
    
    # Add coordinate frames for each EEF pose if provided
    for translation, rotation in eef_poses:
        frame = create_coordinate_frame(translation, rotation)
        vis.add_geometry(frame)
    
    # Set default view
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

    # To save a point cloud as a PLY file:
    # 1. Create an Open3D point cloud object
    # 2. Use o3d.io.write_point_cloud("pcd.ply", pcd)
    # Example:
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.io.write_point_cloud("pcd.ply", pcd)

    # import IPython; IPython.embed()
