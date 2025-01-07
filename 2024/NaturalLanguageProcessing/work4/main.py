"""
采用BIO方式代码实现命名实体识别和分类任务。
"""
import json
def get_data(path):
    """
    获取数据
    :param path:
    :return:
    """
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i > 5:
                continue
            sample = json.loads(line.strip())
            print(sample)
            data.append(sample)
    return data


if __name__ == "__main__":
    data = get_data("BERT-BiLSTM-CRF-NER-pytorch/cluener2020.json")

