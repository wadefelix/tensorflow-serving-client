from tensorflow_serving_client import TensorflowServingClient
from tensorflow_serving_client.utils import load_image, MODEL_SPECS


MODEL_SERVING_PORTS = {
    'mobilenet_v1': 9001,
    'inception_v3': 9002,
    'xception': 9003,
    'resnet50': 9004,
    'vgg16': 9005,
    'vgg19': 9006,
}


def query_model(model_spec_name):
    model_spec = MODEL_SPECS[model_spec_name]
    client = TensorflowServingClient('localhost', MODEL_SERVING_PORTS[model_spec_name])
    image = load_image('tests/fixtures/files/cat.jpg',
                       model_spec['target_size'],
                       model_spec['preprocess_input'])
    return client.make_prediction(image, 'image')


def assert_predictions(response, expected_top_5, imagenet_dictionary):
    assert 'class_probabilities' in response
    assert len(response['class_probabilities']) == 1
    assert len(response['class_probabilities'][0]) == 1000
    predictions = response['class_probabilities'][0]
    predictions = list(zip(imagenet_dictionary, predictions))
    predictions = sorted(predictions, reverse=True, key=lambda kv: kv[1])[:5]
    predictions = [(label, float(score)) for label, score in predictions]
    print(predictions)
    assert predictions == expected_top_5


def test_mobilenet_v1(imagenet_dictionary):
    response = query_model('mobilenet_v1')
    assert_predictions(response, [
        ('tiger cat', 0.334695041179657),
        ('Egyptian cat', 0.28513845801353455),
        ('tabby, tabby cat', 0.1547166407108307),
        ('kit fox, Vulpes macrotis', 0.03160473331809044),
        ('lynx, catamount', 0.030886217951774597)
    ], imagenet_dictionary)


def test_inception_v3(imagenet_dictionary):
    response = query_model('inception_v3')
    assert_predictions(response, [
        ('tiger cat', 0.47168827056884766),
        ('Egyptian cat', 0.1279538869857788),
        ('Pembroke, Pembroke Welsh corgi', 0.07338253408670425),
        ('tabby, tabby cat', 0.052391838282346725),
        ('Cardigan, Cardigan Welsh corgi', 0.008323835209012032)
    ], imagenet_dictionary)


def test_xception(imagenet_dictionary):
    response = query_model('xception')
    assert_predictions(response, [
        ('red fox, Vulpes vulpes', 0.10058525949716568),
        ('weasel', 0.09152577072381973),
        ('Pembroke, Pembroke Welsh corgi', 0.07581677287817001),
        ('tiger cat', 0.07467170804738998),
        ('kit fox, Vulpes macrotis', 0.06751599907875061)
    ], imagenet_dictionary)


def test_resnet50(imagenet_dictionary):
    response = query_model('resnet50')
    assert_predictions(response, [
        ('red fox, Vulpes vulpes', 0.3193321228027344),
        ('kit fox, Vulpes macrotis', 0.19359812140464783),
        ('weasel', 0.14291061460971832),
        ('Pembroke, Pembroke Welsh corgi', 0.13959810137748718),
        ('lynx, catamount', 0.0461868941783905)
    ], imagenet_dictionary)


def test_vgg16(imagenet_dictionary):
    response = query_model('vgg16')
    assert_predictions(response, [
        ('kit fox, Vulpes macrotis', 0.3090210556983948),
        ('red fox, Vulpes vulpes', 0.21598467230796814),
        ('Egyptian cat', 0.13274021446704865),
        ('tiger cat', 0.11005253344774246),
        ('tabby, tabby cat', 0.08285782486200333)
    ], imagenet_dictionary)


def test_vgg19(imagenet_dictionary):
    response = query_model('vgg19')
    assert_predictions(response, [
        ('red fox, Vulpes vulpes', 0.3812934458255768),
        ('kit fox, Vulpes macrotis', 0.2726273238658905),
        ('tiger cat', 0.0855349525809288),
        ('lynx, catamount', 0.05379558727145195),
        ('Egyptian cat', 0.04786992818117142)
    ], imagenet_dictionary)
