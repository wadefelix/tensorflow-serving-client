from tensorflow_serving_client import TensorflowServingClient


def test_classify_image():
    client = TensorflowServingClient('localhost', 9000, 'mobilenet_v1-1', (224, 224))
    result = client.classify_image('tests/fixtures/files/cat.jpg')
    assert len(result) > 0
