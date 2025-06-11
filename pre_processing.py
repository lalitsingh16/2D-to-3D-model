import open3d as o3d

def refine_point_cloud(pcd, nb_neighbors=20, std_ratio=2.0, voxel_size=None,
                       estimate_normals=True, normal_radius=0.7, max_nn=30, orient_normals=True):
    """
    Refines an Open3D point cloud by removing outliers, optionally downsampling, and estimating normals.

    Parameters:
    -----------
    pcd : open3d.geometry.PointCloud
        Input point cloud to refine.
    nb_neighbors : int
        Number of neighbors to analyze for outlier removal.
    std_ratio : float
        Standard deviation multiplier for statistical outlier removal.
    voxel_size : float or None
        If provided, downsample the point cloud with this voxel size.
    estimate_normals : bool
        Whether to estimate normals after cleaning.
    normal_radius : float
        Search radius for normal estimation.
    max_nn : int
        Maximum number of neighbors to use for normal estimation.
    orient_normals : bool
        Whether to orient the normals after estimation.

    Returns:
    --------
    refined_pcd : open3d.geometry.PointCloud
        The refined (cleaned, optionally downsampled, normal-estimated) point cloud.
    """

    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise TypeError("Input must be an open3d.geometry.PointCloud object.")
    if len(pcd.points) == 0:
        raise ValueError("Input point cloud contains no points.")



    # Clone to avoid in-place modification
    pcd_copy = o3d.geometry.PointCloud(pcd)

    # Outlier removal
    _, ind = pcd_copy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    refined_pcd = pcd_copy.select_by_index(ind)

    # Downsampling
    if voxel_size is not None and voxel_size > 0:
        refined_pcd = refined_pcd.voxel_down_sample(voxel_size=voxel_size)
    # Normal estimation
    if estimate_normals and len(refined_pcd.points) > 0:
        refined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=max_nn))

    return refined_pcd
