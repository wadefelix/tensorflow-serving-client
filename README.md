Python client for communicating with [tensorflow-serving](https://github.com/tensorflow/serving).


# About

Python client to communicate with the TensorFlow RPC API. Similar to the examples in the documentation of [tensorflow-serving](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example), we use gRPC as the RPC client.


# Attribution

Originally forked from [sebastian-schlecht/tensorflow-serving-python](https://github.com/sebastian-schlecht/tensorflow-serving-python).


# TODO

  - [ ] Link TensorFlow Serving as a sub-module here. To compile protobufs properly during build.  
  - [ ] When distributing, get rid of tf.contrib functions to remove TensorFlow as a dependency.
