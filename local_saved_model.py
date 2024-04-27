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


def albert_sentences_predict():
  model_dir = './saved_model/1573466695'
  model = load_model(model_dir)
  infer = model.signatures['serving_default']
  label_map = get_label_map()

  text_a = "我要买流量"
  text_list = [text_a] * 10
  label_ids = [0]
  def process_predict_v1(text_list):
    for text_a in text_list:
      input_ids, input_mask, segment_ids = get_input(text_a)
      result = infer(input_ids=tf.constant(input_ids),
                     input_mask=tf.constant(input_mask),
                     segment_ids=tf.constant(segment_ids),
                     label_ids=tf.constant(label_ids))
      index = np.argmax(result['probabilities'].numpy())
      yield label_map[str(index)], result['probabilities'].numpy()[0][index]

  def process_predict_v2(text_a):
    input_ids, input_mask, segment_ids = get_input(text_a)
    start_time = time.time()
    result = infer(input_ids=tf.constant(input_ids),
                   input_mask=tf.constant(input_mask),
                   segment_ids=tf.constant(segment_ids),
                   label_ids=tf.constant(label_ids))
    print("Time :", time.time() - start_time)
    index = np.argmax(result['probabilities'].numpy())
    print("BERT   Label:", label_map[str(index)])
    print("BERT   Score:", result['probabilities'].numpy()[0][index])
    print("ALBERT Label:", )
    print("ALBERT Score:")
    return label_map[str(index)], result['probabilities'].numpy()[0][index]
  result = map(process_predict_v2, text_list)
  print(list(result))
  print('-' * 30)
  print(list(process_predict_v1(text_list)))


def test_positive(file_path):
  count = 0
  total = 0
  with open(file_path, 'r') as f:
    results = []
    for idx, line in enumerate(f.readlines()):
      total += 1
      print('-' * 30)
      print("Index  :", idx + 1)
      print("Text   :", line.strip())
      label, score = predict_single_sentence(line.strip())
      results.append(label)
      if label != '00020101':
        count += 1
    print('=' * 30)
    print("Total  :", total)
    print("Count  :", count)
    print("Accuary:", (total - count) / total)
    print('=' * 30)
    counter = Counter(results)
    print(counter.most_common(len(set(results))))

def test_negative(file_path):
  count = 0
  total = 0
  with open(file_path, 'r') as f:
    results = []
    for idx, line in enumerate(f.readlines()):
      total += 1
      print('-' * 30)
      print("Index  :", idx)
      print("Text   :", line.strip())
      label, score = predict_single_sentence(line.strip())
      results.append(label)
      if label == '00020101':
        count += 1
    print('=' * 30)
    print("Total  :", total)
    print("Count  :", count)
    print("Accuary:", (total - count) / total)
    print('=' * 30)
    counter = Counter(results)
    print(counter.most_common(len(set(results))))


if __name__ == '__main__':
  albert_sentences_predict()
