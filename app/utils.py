import base64
from io import BytesIO
from PIL import Image

def decode_base64_image(base64_str: str) -> Image.Image:
    """
    Decodes a base64-encoded image string into a PIL Image.

    Args:
        base64_str (str): Base64 string (can include header like 'data:image/jpeg;base64,').

    Returns:
        PIL.Image.Image: The decoded image.
    """
    # Remove data URI scheme header if present
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",", 1)[1]
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encodes image bytes to a base64 string.

    Args:
        image_bytes (bytes): Raw image bytes.

    Returns:
        str: Base64-encoded string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")
