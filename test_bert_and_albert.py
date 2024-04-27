import albert_predict as albert
import bert_predict as bert
from collections import Counter


def test_positive(file_path):
  bert_count = 0
  albert_count = 0
  total = 0
  with open(file_path, 'r') as f:
    bert_results = []
    albert_results = []
    for idx, line in enumerate(f.readlines()):
      total += 1
      print('-' * 30)
      print("Index       :", idx + 1)
      print("Text        :", line.strip())
      bert_label, bert_score = bert.predict_single_sentence(line.strip())
      albert_label, albert_score = albert.albert_sentences_predict(line.strip())
      bert_results.append(bert_label)
      albert_results.append(albert_label)
      if bert_label != '00020101':
        bert_count += 1
      if albert_label != '00020101':
        albert_count += 1
    print('=' * 30)
    print("Total         :", total)
    print("BERT     Count:", bert_count)
    print("BERT   Accuary:", (total - bert_count) / total)
    print("ALBERT   Count:", albert_count)
    print("ALBERT Accuary:", (total - albert_count) / total)
    print('=' * 30)
    bert_counter = Counter(bert_results)
    print(bert_counter.most_common(len(set(bert_results))))
    albert_counter = Counter(albert_results)
    print(albert_counter.most_common(len(set(albert_results))))

def test_negative(file_path):
  bert_count = 0
  albert_count = 0
  total = 0
  with open(file_path, 'r') as f:
    bert_results = []
    albert_results = []
    for idx, line in enumerate(f.readlines()):
      total += 1
      print('-' * 30)
      print("Index       :", idx)
      print("Text        :", line.strip())
      bert_label, bert_score = bert.predict_single_sentence(line.strip())
      albert_label, albert_score = albert.albert_sentences_predict(line.strip())
      bert_results.append(bert_label)
      albert_results.append(albert_label)
      if bert_label == '00020101':
        bert_count += 1
      if albert_label == '00020101':
        albert_count += 1
    print('=' * 30)
    print("Total         :", total)
    print("BERT     Count:", bert_count)
    print("BERT   Accuary:", (total - bert_count) / total)
    print("ALBERT   Count:", albert_count)
    print("ALBERT Accuary:", (total - albert_count) / total)
    print('=' * 30)
    bert_counter = Counter(bert_results)
    print(bert_counter.most_common(len(set(bert_results))))
    albert_counter = Counter(albert_results)
    print(albert_counter.most_common(len(set(albert_results))))


if __name__ == '__main__':
  test_positive("../test/liuliangbao_positive.txt")
  #test_negative("../test/liuliangbao_negative.txt")
