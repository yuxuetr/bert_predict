import time
import tensorflow as tf
import numpy as np
from bert.run_classifier import InputExample
from bert.run_classifier import convert_examples_to_features
from bert import tokenization

def load_model(model_dir):
  return tf.saved_model.load(model_dir)


def get_label_map(label_file='./intention.v0.3.labels'):
  with open(label_file, 'r') as f:
    return {item[0]: item[1].strip() for item in [line.split(':') for line in f.readlines()]}


def get_input(text_a, vocab_file='./vocab.txt', max_seq_len=64, guid=''):
  label_list = get_label_map().keys()
  tokenizer = tokenization.FullTokenizer(vocab_file)
  examples = [InputExample(guid, text_a, label='0')]
  features = convert_examples_to_features(examples, label_list, max_seq_len, tokenizer)
  input_ids = np.expand_dims(np.array(features[0].input_ids), axis=0).tolist()
  input_mask = np.expand_dims(np.array(features[0].input_mask), axis=0).tolist()
  segment_ids = np.zeros((max_seq_len), dtype=int).tolist()
  return input_ids, input_mask, segment_ids


model_dir = './saved_model/1573466695'
model = load_model(model_dir)
infer = model.signatures['serving_default']
label_map = get_label_map()


def albert_sentences_predict(text_a):
  input_ids, input_mask, segment_ids = get_input(text_a)
  result = infer(input_ids=tf.constant(input_ids),
                   input_mask=tf.constant(input_mask),
                   segment_ids=tf.constant(segment_ids),
                   label_ids=tf.constant([0]))
  index = np.argmax(result['probabilities'].numpy())
  print("ALBERT Label:", label_map[str(index)])
  print("ALBERT Score:", result['probabilities'].numpy()[0][index])
  return label_map[str(index)], result['probabilities'].numpy()[0][index]


if __name__ == '__main__':
  albert_sentences_predict()
