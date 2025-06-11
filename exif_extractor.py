from PIL import Image
from PIL.ExifTags import TAGS
import warnings

def extract_camera_intrinsics(image_path, default_fx=1000, default_fy=1000):
    """
    Extract camera intrinsic parameters (fx, fy, cx, cy) from image EXIF metadata.
    If metadata is missing or incomplete, return default intrinsic values.

    Parameters:
    -----------
    image_path : str
        File path to the input image.
    default_fx : float
        Default focal length in pixels on x-axis (fallback).
    default_fy : float
        Default focal length in pixels on y-axis (fallback).

    Returns:
    --------
    dict
        Dictionary containing intrinsic parameters:
        {
            'fx': focal length x (pixels),
            'fy': focal length y (pixels),
            'cx': principal point x (pixels),
            'cy': principal point y (pixels)
        }

    Procedure:
    ----------
    1. Open the image using PIL and extract EXIF metadata.
    2. Parse focal length from EXIF, converting rational to float if needed.
    3. Assume a sensor width of 36mm (full-frame) unless overridden.
    4. Calculate fx and fy by scaling focal length (mm) relative to sensor size and image dimensions.
    5. Principal point (cx, cy) is the image center.
    6. If metadata unavailable or parsing fails, use default focal lengths and fallback dimensions.
    """

    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if exif_data is None:
            warnings.warn("No EXIF metadata found, using default intrinsics.")
            raise ValueError("No EXIF data")

        # Map EXIF tags to human-readable keys
        exif = {TAGS.get(tag, tag): val for tag, val in exif_data.items()}

        focal_length = exif.get('FocalLength')
        if isinstance(focal_length, tuple) and len(focal_length) == 2:
            # Convert rational to float: numerator/denominator
            focal_length = focal_length[0] / focal_length[1]

        sensor_width_mm = 36.0  # default full-frame sensor width (mm)

        if 'Model' in exif:
            model = exif['Model']
            # Optionally implement sensor size lookup per camera model here
            print(f"Camera model: {model} (assuming sensor width = {sensor_width_mm}mm)")

        width, height = img.size
        cx = width / 2
        cy = height / 2

        if focal_length is not None:
            # Calculate focal length in pixels based on sensor width and image size
            fx = (focal_length * width) / sensor_width_mm
            fy = (focal_length * height) / sensor_width_mm
            print(f"Extracted focal length: {focal_length}mm → fx={fx:.2f}, fy={fy:.2f}")
        else:
            warnings.warn("Focal length not found in EXIF, using default values.")
            fx, fy = default_fx, default_fy

    except Exception as e:
        warnings.warn(f"Failed to extract intrinsics from metadata: {e}. Using defaults.")
        width, height = 1920, 1080  # fallback image dimensions
        fx, fy = default_fx, default_fy
        cx, cy = width / 2, height / 2

    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
