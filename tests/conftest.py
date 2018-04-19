import pytest
import csv
import requests


@pytest.fixture(scope='session')
def imagenet_dictionary():
    response = requests.get('https://storage.googleapis.com/tf-models-839c7ddd-9cab-49fa-9b42-bde1a842086e/dictionary.csv')
    reader = csv.reader(response.text.splitlines())
    items = sorted(list(reader), key=lambda kv: int(kv[0]))
    return [name for label, name in items]
