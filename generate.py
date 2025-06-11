import cv2
import numpy as np
import torch
import open3d as o3d
import pyvista as pv
from exif_extractor import extract_camera_intrinsics
from pre_processing import refine_point_cloud
from mesh_generatation import apply_surface_reconstruction
from depthmap import depth_genrate



def generate_model(image_bgr: np.ndarray):
    """
    Generate a 3D mesh and point cloud from an input image using depth estimation.

    Parameters:
    -----------
    image_bgr : np.ndarray
        Input RGB image in BGR format (OpenCV standard), dtype=uint8.

    Returns:
    --------
    mesh : o3d.geometry.TriangleMesh
        Reconstructed 3D surface mesh.
    point_cloud : pv.PolyData
        Colored point cloud in PyVista format.
    depth_map : np.ndarray
        Estimated depth map (float32) aligned with the input image.
    """
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input must be a 3-channel (color) image.")

    # Step 1: Depth map estimation
    depth_map = depth_genrate(image_bgr)
    if depth_map.dtype != np.float32:
        depth_map = depth_map.astype(np.float32)

    height, width = depth_map.shape

    # Step 2: RGB normalization
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Step 3: Tensor conversion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_tensor = torch.from_numpy(depth_map).to(device)
    rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).to(device)  # Shape: [3, H, W]

    # Step 4: Intrinsics extraction
    intrinsics = extract_camera_intrinsics(rgb_image)
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    # Step 5: Pixel grid creation
    u = torch.arange(0, width, device=device).float()
    v = torch.arange(0, height, device=device).float()
    grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # Shape: [H, W]

    # Step 6: Backproject to 3D space
    z = depth_tensor
    x = (grid_u - cx) * z / fx
    y = (grid_v - cy) * z / fy

    # Step 7: Flatten and filter
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    rgb_flat = rgb_tensor.view(3, -1).permute(1, 0)  # Shape: [N, 3]

    valid = (z_flat > 0) & (~torch.isnan(z_flat))
    xyz = torch.stack((x_flat[valid], -y_flat[valid], z_flat[valid]), dim=1).cpu().numpy()
    colors = rgb_flat[valid].cpu().numpy()

    # Step 8: Construct and refine Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = refine_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0, voxel_size=0.01)

    # Step 9: Convert to PyVista format
    pcd_np = np.asarray(pcd.points)
    rgb_np = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    point_cloud = pv.PolyData(pcd_np)
    point_cloud["RGB"] = rgb_np

    # Step 10: Surface reconstruction
    mesh = apply_surface_reconstruction(pcd)

    return mesh, point_cloud, depth_map
