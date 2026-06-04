from PIL import Image

def preprocess_image(image: Image.Image) -> Image.Image:
    if image is None:
        raise ValueError("No image provided")
        
    w, h = image.size
    if w < 224 or h < 224:
        raise ValueError(f"Image is too small ({w}x{h}). Minimum size is 224x224 px.")
        
    # Convert RGBA or L to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Center crop / Resize if too large (> 1280px on any dimension)
    max_size = 1280
    if w > max_size or h > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    return image
