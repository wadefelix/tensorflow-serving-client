import pytest
import csv
import codecs

from urllib.request import urlopen


@pytest.fixture
def imagenet_dictionary():
    response = urlopen('https://storage.googleapis.com/tf-serving-docker-http-eae9e0c7-661d-4cca-836a-0433a8da44ba/imagenet/dictionary.csv')
    reader = csv.reader(codecs.iterdecode(response, 'utf-8'))
    dictionary = dict(list(reader))
    return [dictionary[key] for key in sorted(dictionary)]
