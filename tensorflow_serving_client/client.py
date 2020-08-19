import tensorflow as tf
import grpc

from tensorflow_serving_client.protos import prediction_service_pb2_grpc, predict_pb2
from tensorflow_serving_client.proto_util import copy_message


class TensorflowServingClient(object):

    def __init__(self, host, port, cert=None):
        self.host = host
        self.port = port
        self.channel = grpc.insecure_channel('%s:%s' % (host, port))
        if cert is None:
            self.channel = grpc.insecure_channel('%s:%s' % (host, port))
        else:
            with open(cert,'rb') as f:
                trusted_certs = f.read()
            credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
            self.channel = grpc.secure_channel('%s:%s' % (host, port), credentials, options=None)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    def execute(self, request, timeout=10.0):
        return self.stub.Predict(request, timeout)

    def make_prediction(self, input_data, input_tensor_name, timeout=10.0, model_name=None):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name or 'model'

        copy_message(tf.contrib.util.make_tensor_proto(input_data, dtype='float32'), request.inputs[input_tensor_name])
        response = self.execute(request, timeout=timeout)

        results = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = tf.contrib.util.make_ndarray(tensor_proto)
            results[key] = nd_array

        return results
