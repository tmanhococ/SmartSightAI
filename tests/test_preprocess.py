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
