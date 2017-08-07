Python client for communicating with [tensorflow-serving](https://github.com/tensorflow/serving).

[![Build Status](https://travis-ci.org/triagemd/tensorflow-serving-client.svg?branch=master)](https://travis-ci.org/triagemd/tensorflow-serving-client)

## Getting started

Install dependencies and start the Docker container with:
```
script/up
```

Soruce install your Python virtualenv with:
```
script/env
```

Run tests with:
```
script/test
```

Upload a new version to PyPI with:
```
script/distribute
```

Query the running tensorflow-serving Docker container instance with:
```
tensorflow_serving_client --host localhost --port 9000 --image tests/fixtures/files/cat.jpg --size 224x224
```
