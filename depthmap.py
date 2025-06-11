from transformers import pipeline
from PIL import Image
import numpy as np


def depth_genrate(url):
    """
    Generate a depth map from an input image array using a pretrained depth estimation model.

    Parameters:
    ----------
    url : numpy.ndarray
        Input image as a NumPy array (e.g., read via OpenCV or similar).

    Returns:
    -------
    depth_map : numpy.ndarray (float32)
        Depth map output as a 2D float32 NumPy array representing per-pixel depth values.

    Process:
    1. Initialize the Hugging Face transformers pipeline for depth estimation
       using the "Intel/dpt-large" model.
    2. Convert the input NumPy array image to a PIL Image.
    3. Pass the PIL Image to the depth estimation pipeline.
    4. Extract and convert the resulting depth output to a float32 NumPy array.
    """

    # Initialize the depth estimation pipeline with pretrained model
    pipe = pipeline(task="depth-estimation", model="Intel/dpt-large")

    # Convert input NumPy image array to PIL Image format for pipeline input
    image = Image.fromarray(url)

    # Execute the depth estimation pipeline to obtain depth data
    result = pipe(image)

    # Extract the depth map from the pipeline output and convert to float32 NumPy array
    depth_map = np.array(result["depth"]).astype(np.float32)

    return depth_map
