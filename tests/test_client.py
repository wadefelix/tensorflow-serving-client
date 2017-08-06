import numpy as np

from PIL import Image
from tensorflow_serving_client import TensorflowServingClient


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def load_image(image_path, target_size=None):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if target_size:
        width_height = (target_size[0], target_size[1])
        if img.size != width_height:
            img = img.resize(width_height)
    image_data = np.asarray(img, dtype=np.float32)
    image_data = preprocess_input(image_data)
    image_data = np.expand_dims(image_data, axis=0)
    return image_data


def test_client_make_prediction():
    client = TensorflowServingClient('localhost', 9000)
    image = load_image('tests/fixtures/files/cat.jpg', (224, 224))
    result = client.make_prediction(image, 'image')
    assert 'class_probabilities' in result
    assert len(result['class_probabilities']) == 1
    assert len(result['class_probabilities'][0]) == 7
    assert len(result['class_probabilities'][0][0]) == 7
    assert len(result['class_probabilities'][0][0][0]) == 1024
