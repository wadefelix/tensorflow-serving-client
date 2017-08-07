from tensorflow_serving_client import TensorflowServingClient
from tensorflow_serving_client.utils import load_image


def test_client_make_prediction(imagenet_dictionary):
    client = TensorflowServingClient('localhost', 9000)
    image = load_image('tests/fixtures/files/cat.jpg', (224, 224))
    result = client.make_prediction(image, 'image')
    assert 'class_probabilities' in result
    assert len(result['class_probabilities']) == 1
    assert len(result['class_probabilities'][0]) == 1000
    predictions = result['class_probabilities'][0]
    predictions = list(zip(imagenet_dictionary, predictions))
    predictions = sorted(predictions, reverse=True, key=lambda kv: kv[1])[:5]
    predictions = [(label, float(score)) for label, score in predictions]
    expected = [
        ('impala, Aepyceros melampus', 0.334694504737854),
        ('llama', 0.2851393222808838),
        ('hartebeest', 0.15471667051315308),
        ('bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 0.03160465136170387),
        ('mink', 0.030886519700288773),
    ]
    assert predictions == expected
