import pytest
import csv
import codecs

from urllib.request import urlopen


@pytest.fixture
def imagenet_dictionary():
    response = urlopen('https://s3.amazonaws.com/tf-models-839c7ddd-9cab-49fa-9b42-bde1a842086e/dictionary.csv')
    reader = csv.reader(codecs.iterdecode(response, 'utf-8'))
    dictionary = dict(list(reader))
    return [dictionary[key] for key in sorted(dictionary)]
