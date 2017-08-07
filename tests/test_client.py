from tensorflow_serving_client import TensorflowServingClient
from tensorflow_serving_client.utils import load_image


def test_client_make_prediction():
    client = TensorflowServingClient('localhost', 9000)
    image = load_image('tests/fixtures/files/cat.jpg', (224, 224))
    result = client.make_prediction(image, 'image')
    assert 'class_probabilities' in result
    assert len(result['class_probabilities']) == 1
    assert len(result['class_probabilities'][0]) == 7
    assert len(result['class_probabilities'][0][0]) == 7
    assert len(result['class_probabilities'][0][0][0]) == 1024
