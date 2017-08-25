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
        ('tiger cat', 0.334694504737854),
        ('Egyptian cat', 0.2851393222808838),
        ('tabby, tabby cat', 0.15471667051315308),
        ('kit fox, Vulpes macrotis', 0.03160465136170387),
        ('lynx, catamount', 0.030886519700288773)
    ], imagenet_dictionary)


def test_inception_v3(imagenet_dictionary):
    response = query_model('inception_v3')
    assert_predictions(response, [
        ('tiger cat', 0.4716886878013611),
        ('Egyptian cat', 0.127954363822937),
        ('Pembroke, Pembroke Welsh corgi', 0.07338221371173859),
        ('tabby, tabby cat', 0.052391838282346725),
        ('Cardigan, Cardigan Welsh corgi', 0.008323794230818748)
    ], imagenet_dictionary)


def test_xception(imagenet_dictionary):
    response = query_model('xception')
    assert_predictions(response, [
        ('red fox, Vulpes vulpes', 0.10058529675006866),
        ('weasel', 0.09152575582265854),
        ('Pembroke, Pembroke Welsh corgi', 0.07581676542758942),
        ('tiger cat', 0.0746716633439064),
        ('kit fox, Vulpes macrotis', 0.06751589477062225)
    ], imagenet_dictionary)


def test_resnet50(imagenet_dictionary):
    response = query_model('resnet50')
    assert_predictions(response, [
        ('red fox, Vulpes vulpes', 0.3193315863609314),
        ('kit fox, Vulpes macrotis', 0.19359852373600006),
        ('weasel', 0.14291106164455414),
        ('Pembroke, Pembroke Welsh corgi', 0.1395975947380066),
        ('lynx, catamount', 0.04618712514638901)
    ], imagenet_dictionary)


def test_vgg16(imagenet_dictionary):
    response = query_model('vgg16')
    assert_predictions(response, [
        ('kit fox, Vulpes macrotis', 0.3090206980705261),
        ('red fox, Vulpes vulpes', 0.21598483622074127),
        ('Egyptian cat', 0.1327403038740158),
        ('tiger cat', 0.11005250364542007),
        ('tabby, tabby cat', 0.08285804092884064)
    ], imagenet_dictionary)


def test_vgg19(imagenet_dictionary):
    response = query_model('vgg19')
    assert_predictions(response, [
        ('red fox, Vulpes vulpes', 0.3812929391860962),
        ('kit fox, Vulpes macrotis', 0.27262774109840393),
        ('tiger cat', 0.08553500473499298),
        ('lynx, catamount', 0.05379556491971016),
        ('Egyptian cat', 0.047869954258203506)
    ], imagenet_dictionary)
