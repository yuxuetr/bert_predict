import tensorflow as tf
import numpy as np
from bert.run_classifier import InputExample
from bert.run_classifier import convert_examples_to_features
from bert import tokenization

def get_label_map(label_file='./intention.v0.3.labels'):
  with open(label_file, 'r') as f:
    return {item[0]: item[1].strip() for item in [line.split(':') for line in f.readlines()]}


def get_input(text_a, vocab_file='./vocab.txt', max_seq_len=64, guid=''):
  label_list = get_label_map().keys()
  tokenizer = tokenization.FullTokenizer(vocab_file)
  examples = [InputExample(guid, text_a, label='0')]
  features = convert_examples_to_features(examples, label_list, max_seq_len, tokenizer)
  input_ids = np.expand_dims(np.array(features[0].input_ids, dtype=np.int32), axis=0).tolist()
  input_mask = np.expand_dims(np.array(features[0].input_mask, dtype=np.int32), axis=0).tolist()
  segment_ids = np.expand_dims(np.array(features[0].segment_ids, dtype=np.int32), axis=0).tolist()
  label_ids = [0]
  return tf.constant(input_ids), tf.constant(input_mask), tf.constant(segment_ids)


def get_interpreter(tflite_model_file):
  interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
  interpreter.allocate_tensors()
  return interpreter

def predict(text_a, tflite_model_file):
  input_ids, input_mask, segment_ids = get_input(text_a)
  label_map = get_label_map()
  interpreter = get_interpreter(tflite_model_file)

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]["index"], input_ids)
  interpreter.set_tensor(input_details[1]["index"], segment_ids)
  interpreter.set_tensor(input_details[2]["index"], tf.constant([0]))
  interpreter.set_tensor(input_details[3]["index"], input_mask)
  interpreter.invoke()

  output = interpreter.get_tensor(output_details[0]["index"])
  index = np.argmax(output[0])
  return label_map[str(index)], output[0][index]


if __name__ == '__main__':
  tflite_model_file = 'model.tflite'
  text_a = "我要买流量，三天"
  label, score = predict(text_a, tflite_model_file)
  print("Label:", label)
  print("Score:", score)
  text_a = "有多少流量"
  label, score = predict(text_a, tflite_model_file)
  print("Label:", label)
  print("Score:", score)
