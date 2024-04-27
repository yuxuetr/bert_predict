import requests
import json
import os
import time


def get_predict_result(text, idx):
  text_list = []
  text_list.append(text)
  data = {
    'id'   : idx,
    'texts': text_list,
    "is_tokenized": False
  }

  headers = {'Content-Type': 'application/json'}
  url = 'http://127.0.0.1:8091/encode'
  start_time = time.time()
  response = requests.post(url=url, headers=headers, data=json.dumps(data))
  end_time = time.time()
  #print("Time          :", end_time - start_time)
  result_list = json.loads(response.text).get("result")
  # print(result_list)
  return result_list


def ids_convert_labels(file_path='./intention.v0.3.labels'):
  ids_to_labels = {}
  with open(file_path, "r", encoding="utf-8") as f:
    all_lines = f.readlines()
    for line in all_lines:
      line = line.split(":")
      ids_to_labels[line[0]] = line[1].strip()
  return ids_to_labels


def parse_result(result_list, ids_to_labels):
  for result in result_list:
    label = result["pred_label"][0]
    score = result["score"][0]
    print("BERT   Label:", ids_to_labels[label])
    print("BERT   Score:", score)
    return ids_to_labels[label], score


def predict_single_sentence(text, idx=0):
  result_list = get_predict_result(text, idx)
  ids_to_labels = ids_convert_labels()
  label, score = parse_result(result_list, ids_to_labels)
  return label, score


if __name__ == "__main__":
  text = "我要买流量啊"
  # print("Text :", text)
  # predict_single_sentence(text)
  for _ in range(100):
    predict_single_sentence(text)
