import pytest
from PIL import Image
from src.pipeline.preprocess import preprocess_image

def test_preprocess_image_valid():
    img = Image.new("RGBA", (1500, 1000), color="red")
    processed = preprocess_image(img)
    assert processed.mode == "RGB"
    assert processed.size[0] <= 1280
    assert processed.size[1] <= 1280

def test_preprocess_image_too_small():
    img = Image.new("L", (100, 100))
    with pytest.raises(ValueError, match="Image is too small"):
        preprocess_image(img)

def test_preprocess_image_none():
    with pytest.raises(ValueError, match="No image provided"):
        preprocess_image(None)

def test_preprocess_image_resize_and_aspect_ratio():
    # Verify image of size 1500x1000 is rescaled to exactly 1280x853
    img = Image.new("RGB", (1500, 1000))
    processed = preprocess_image(img)
    assert processed.size == (1280, 853)

    # Verify image of size 800x600 is not resized
    img_small = Image.new("RGB", (800, 600))
    processed_small = preprocess_image(img_small)
    assert processed_small.size == (800, 600)

def test_preprocess_image_boundaries():
    # Width is 223 (less than 224), height is 224
    with pytest.raises(ValueError, match="Image is too small"):
        preprocess_image(Image.new("RGB", (223, 224)))

    # Width is 224, height is 223 (less than 224)
    with pytest.raises(ValueError, match="Image is too small"):
        preprocess_image(Image.new("RGB", (224, 223)))

    # Both are 224 (minimum boundary)
    img_boundary = Image.new("RGB", (224, 224))
    processed = preprocess_image(img_boundary)
    assert processed.size == (224, 224)

