# -*- coding: utf-8 -*-
from typing import List
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from bert import tokenization
from bert.run_classifier import convert_examples_to_features
from bert.run_classifier import input_fn_builder
from bert.run_classifier import InputExample

import grpc
import numpy as np
import tensorflow as tf
from absl import flags
from absl import app


FLAGS = flags.FLAGS
flags.DEFINE_string("vocab", "vocab.txt", "词典文件")
flags.DEFINE_integer("max_seq_len", 64, "送入模型的最大句子长度")
flags.DEFINE_string("label_file", "intention.v0.3.labels", "预测的标签文件")
flags.DEFINE_string("host", "127.0.0.1", "服务器主机")
flags.DEFINE_integer("port", 8500, "服务器端口")
flags.DEFINE_float("timeout", 1, "服务超时时间")


def get_label_map(label_file):
  with open(label_file, 'r') as f:
    return {item[0] : item[1].strip() for item in [line.split(":") for line in f.readlines()]}


def get_input(text_a, vocab_file, label_list, max_seq_length):
  """
  Params:
    @vocab_file    :
    @label_list    :
    @max_seq_length:
  """
  tokenizer = tokenization.FullTokenizer(vocab_file)
  examples = [InputExample("predict", text_a, label='0')]
  features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)

  input_ids = np.expand_dims(np.array(features[0].input_ids), axis=0).tolist()
  input_mask = np.expand_dims(np.array(features[0].input_mask), axis=0).tolist()
  segment_ids = np.zeros((64), dtype=int).tolist()
  return input_ids, input_mask, segment_ids


def get_request(host, port, timeout):
  channel = grpc.insecure_channel('{}:{}'.format(host, port))
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'bert'
  request.model_spec.signature_name = 'serving_default'
  return request, stub


def get_albert_predict_result(text_a):
  request, stub = get_request(FLAGS.host, FLAGS.port, FLAGS.timeout)
  label_map = get_label_map(FLAGS.label_file)
  input_ids, input_mask, segment_ids = get_input(
    text_a, FLAGS.vocab, label_map.keys(), FLAGS.max_seq_len)

  request.inputs['input_ids'].CopyFrom(tf.make_tensor_proto(input_ids, shape=[1, 64]))
  request.inputs['input_mask'].CopyFrom(tf.make_tensor_proto(input_mask, shape=[1, 64]))
  request.inputs['label_ids'].CopyFrom(tf.make_tensor_proto([0], shape=[1, 1]))
  request.inputs['segment_ids'].CopyFrom(tf.make_tensor_proto(segment_ids, shape=[1, 64]))

  result = stub.Predict(request, FLAGS.timeout)
  index = np.argmax(result.outputs['probabilities'].float_val).item()
  return label_map[str(index)], result.outputs['probabilities'].float_val[index]


def main(_):
  text_a = "我要买流量"
  text_list = [text_a] * 10
  label, score = get_albert_predict_result(text_a)
  print("Label: ", label)
  print("Score: ", score)


if __name__ == '__main__':
  app.run(main)
