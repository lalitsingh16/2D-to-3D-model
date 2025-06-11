import argparse
import cv2
import generate
from mesh_generatation import visualize_with_pyvista
import open3d as o3d
import pyvista

def main():
    """
    Command-line tool to generate a 3D mesh from a given RGB image file,
    save the resulting mesh to disk, and visualize it using PyVista.

    Workflow:
    1. Parse input arguments for RGB image path and output mesh filename.
    2. Read the input RGB image using OpenCV.
    3. Generate 3D mesh, point cloud, and depth map from the image via generate_model().
    4. Save the mesh to the specified output file if it is an Open3D TriangleMesh.
    5. Visualize the mesh using PyVista-based visualization utility.
    """

    parser = argparse.ArgumentParser(description="Generate 3D mesh from image and depth map.")
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--out', default='output_mesh.ply', help='Output mesh filename (.ply or .obj)')

    args = parser.parse_args()

    # Read input RGB image as BGR NumPy array
    img = cv2.imread(args.rgb)

    # Generate 3D mesh, point cloud, and depth map from the image
    output, point, depth = generate.generate_model(img)

    output_ply_path = args.out

    # Verify mesh is Open3D TriangleMesh before saving
    if isinstance(output, o3d.geometry.TriangleMesh):
        print(f"[INFO] Saving mesh to: {output_ply_path}")
        o3d.io.write_triangle_mesh(output_ply_path, output)
    else:
        raise TypeError("The mesh object is not an Open3D TriangleMesh.")

    # Visualize the generated mesh with PyVista
    visualize_with_pyvista(output)
    cv2.imwrite("depth_map.png",depth)
    pc = pyvista.PolyData(point)
    pc.plot(point_size=4,rgb=True)




if __name__ == "__main__":
    main()
