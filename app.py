from flask import Flask, request, render_template, jsonify
import base64
from generate import generate_model
import numpy as np
import cv2
from mesh_generatation import o3d_to_trimesh
import tempfile
import open3d as o3d

app = Flask(__name__)

@app.route('/')
def index():
    """
    Route: GET /
    Serves the main HTML page (index.html).
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Route: POST /upload
    Handles image file upload, processes the image to generate a 3D model,
    encodes the model to both PLY and GLB formats in base64,
    and returns JSON containing base64 data URLs for client-side usage.

    Workflow:
    1. Retrieve the uploaded image file from the request.
    2. Convert the uploaded image bytes into an OpenCV image.
    3. Call generate_model(img) to produce Open3D mesh, point cloud, and depth map.
    4. Convert the Open3D mesh to PLY format and encode it as a base64 string.
    5. Convert the Open3D mesh to trimesh, export as GLB bytes, then encode to base64.
    6. Return a JSON response containing base64 encoded model URLs.
    """

    # Retrieve uploaded image from the client
    image = request.files['image']
    print(f"[INFO] Received image: {image.filename}")

    # Convert image file bytes into numpy array for OpenCV processing
    file_bytes = np.frombuffer(image.read(), np.uint8)

    # Decode bytes into an OpenCV BGR image matrix
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Generate 3D mesh, point cloud, and depth map from the input image
    mesh_org, point, depth = generate_model(img)

    # Function to convert Open3D mesh to PLY format base64 string
    def mesh_to_ply_base64(mesh_org):
        """
        Write the Open3D mesh to a temporary PLY file in ASCII format,
        read the bytes, and encode as a base64 string for data URL embedding.
        """
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as tmp:
            o3d.io.write_triangle_mesh(
                tmp.name,
                mesh_org,
                write_ascii=True,
                write_vertex_colors=True,
                write_vertex_normals=True,
                write_triangle_uvs=False,  # UVs not required here
                compressed=False,
                print_progress=False
            )
            tmp.seek(0)
            ply_bytes = tmp.read()
        return base64.b64encode(ply_bytes).decode('utf-8')

    # Convert mesh to PLY base64 string and prepend data URL scheme
    ply_base64 = mesh_to_ply_base64(mesh_org)
    ply_data_url = f"data:model/ply;base64,{ply_base64}"

    # Convert Open3D mesh to trimesh format for GLB export
    mesh = o3d_to_trimesh(mesh_org)

    # Export trimesh object to GLB binary bytes
    glb_bytes = mesh.export(file_type='glb')

    # Encode GLB bytes to base64 and prepend data URL scheme
    glb_base64 = base64.b64encode(glb_bytes).decode('utf-8')
    glb_data_url = f"data:model/gltf-binary;base64,{glb_base64}"

    # Return JSON response with status message and base64-encoded model data URLs
    return jsonify({
        'message': f"Processed: {image.filename}",
        'model_data_url': glb_data_url,
        "model_ply_url": ply_data_url
    })


if __name__ == '__main__':
    # Run Flask application on port 5001 with debug mode enabled
    app.run(debug=True, port=5001)