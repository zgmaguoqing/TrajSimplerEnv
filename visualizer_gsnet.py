import open3d as o3d
import numpy as np
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import pickle as pkl
from GSNet.gsnet_simpler import GSNet

def create_coordinate_frame(translation, rotation, scale=0.1):
    # Create coordinate frame mesh
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    
    # Transform frame to desired pose
    frame.translate(translation)
    frame.rotate(rotation, center=translation)
    
    return frame

# Example usage
if __name__ == "__main__":
    with open("pcd_camera.pkl", "rb") as f:
        pcd_points = pkl.load(f)
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)

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
        
    # pick_translation, pick_rotation = pkl.load(open("pick.pkl", "rb"))
    # eef_poses.append((pick_translation, pick_rotation))
        
    # place_translation, place_rotation = pkl.load(open("place.pkl", "rb"))
    # eef_poses.append((place_translation, place_rotation))
    
    gsnet = GSNet()
    gg = gsnet.inference(pcd_points)
    
    geoms = gg.to_open3d_geometry_list()
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

    import IPython; IPython.embed()
