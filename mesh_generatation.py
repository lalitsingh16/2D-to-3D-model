import open3d as o3d
import pyvista as pv
import numpy as np
import trimesh
import cv2
import base64


def apply_surface_reconstruction(point_cloud: o3d.geometry.PointCloud, depth=9, width=0, scale=1.1, linear_fit=True):
    """
    Perform Poisson surface reconstruction on the input point cloud.

    Parameters:
    -----------
    point_cloud : o3d.geometry.PointCloud
        Input point cloud with points and normals.
    depth : int
        Octree depth for reconstruction detail (default=9).
    width : int
        Specifies the width of the finest level octree cells (default=0).
    scale : float
        Scale factor to extend the reconstruction volume (default=1.1).
    linear_fit : bool
        Whether to use linear interpolation in reconstruction (default=True).

    Returns:
    --------
    mesh : o3d.geometry.TriangleMesh
        Reconstructed triangle mesh after filtering low-density vertices.

    Notes:
    ------
    - Normals are estimated if absent.
    - Low-density vertices are removed by thresholding at 10th percentile density.
    """
    if not point_cloud.has_normals():
        print("[INFO] Estimating normals...")
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30)
        )

    print("[INFO] Applying Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )

    densities = np.asarray(densities)
    density_threshold = np.percentile(densities, 10)
    vertices_to_keep = densities > density_threshold
    mesh.remove_vertices_by_index(np.where(~vertices_to_keep)[0])

    return mesh


def visualize_with_pyvista(mesh: o3d.geometry.TriangleMesh):
    """
    Visualize a mesh using PyVista.

    Parameters:
    -----------
    mesh : o3d.geometry.TriangleMesh
        Input mesh to visualize.

    Process:
    --------
    - Converts Open3D mesh vertices and faces to PyVista format.
    - Transfers vertex colors if available.
    - Displays interactive PyVista window with mesh.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Prepend number of points per face (3 for triangles)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    faces_pv = faces_pv.flatten()

    pv_mesh = pv.PolyData(vertices, faces_pv)

    if len(mesh.vertex_colors) > 0:
        colors = np.asarray(mesh.vertex_colors)
        pv_mesh.point_data["RGB"] = colors

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="RGB", rgb=True, point_size=3.0)
    plotter.show()


def o3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """
    Convert an Open3D mesh to a Trimesh object.

    Parameters:
    -----------
    o3d_mesh : o3d.geometry.TriangleMesh
        Input mesh.

    Returns:
    --------
    trimesh.Trimesh
        Mesh in Trimesh format including vertex colors.
    """
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    vertex_colors = np.asarray(o3d_mesh.vertex_colors)

    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)


def depthmap_to_png_base64(depthmap: np.ndarray) -> str:
    """
    Convert a depth map to a PNG image encoded as a base64 string.

    Parameters:
    -----------
    depthmap : np.ndarray
        Input depth map as a 2D float array.

    Returns:
    --------
    str
        Base64 encoded PNG image with data URI prefix.
    """
    depth_normalized = cv2.normalize(depthmap, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)

    success, buffer = cv2.imencode('.png', depth_uint8)
    if not success:
        raise RuntimeError("Failed to encode depth map as PNG")

    png_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{png_base64}"
