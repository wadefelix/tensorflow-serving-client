import pytest
import csv
import requests


@pytest.fixture
def imagenet_dictionary():
    response = requests.get('https://s3.amazonaws.com/tf-models-839c7ddd-9cab-49fa-9b42-bde1a842086e/dictionary.csv')
    reader = csv.reader(response.text.splitlines())
    dictionary = dict(list(reader))
    return [dictionary[key] for key in dictionary]
