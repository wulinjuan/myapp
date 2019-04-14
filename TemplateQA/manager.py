import math
import random


class BatchManager(object):
    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))  # 计算处理批次数
        sorted_data = sorted(data, key=lambda x: len(x[0]))  # 以句子的长度排序
        batch_data = list()
        # 按批次处理数据，使_data中的数据长度一样
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size) : (i+1)*int(batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])   # 句子的最长长度
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]