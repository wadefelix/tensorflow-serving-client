Python client for communicating with [tensorflow-serving](https://github.com/tensorflow/serving).

[![CircleCI](https://circleci.com/gh/triagemd/tensorflow-serving-client.svg?style=svg)](https://circleci.com/gh/triagemd/tensorflow-serving-client)

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

Query the running tensorflow-serving Docker container instance with:
```
tensorflow_serving_client --host localhost --port 9000 --image tests/fixtures/files/cat.jpg --size 224x224
```
