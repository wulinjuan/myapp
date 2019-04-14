# encoding=utf8
"""
  程序是在github中的一个程序基础上的进行改动
  链接为：https://github.com/zjy-ucas/ChineseNER
"""
import os
import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from TemplateQA import loader
from TemplateQA.manager import BatchManager
from TemplateQA.model import Model
from TemplateQA.utils import clean, make_path, load_config, save_config, get_logger, print_config, create_model, save_model, \
    test_ner, input_from_line

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Wither train the model")
# 模型配置的一些参数
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")  # LSTM的隐藏层数
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")  # 标签模式

# 训练的参数
flags.DEFINE_float("clip",          5,          "Gradient clip")  # 控制梯度，防止梯度爆炸
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")  # 防止过拟合
flags.DEFINE_float("batch_size",    35,         "batch size")  # 批处理的大小
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")  # 初始学习率
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "./TemplateQA/ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "./TemplateQA/maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "./TemplateQA/config_file",  "File for config")
flags.DEFINE_string("script",       "./TemplateQA/conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "./TemplateQA/result",       "Path for results")
flags.DEFINE_string("emb_file",     "./TemplateQA/wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "train_set.txt"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "dev_set.txt"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "test_set.txt"),   "Path for test data")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower  # 判断小写
    return config


def train():
    train_sentences = loader.load_sentences(FLAGS.train_file)
    dev_sentences = loader.load_sentences(FLAGS.dev_file)
    test_sentences = loader.load_sentences(FLAGS.test_file)
    # 检查句子里的标签是不是符合标准，并转化为IOBES的模式
    loader.update_tag_scheme(train_sentences, FLAGS.tag_schema)
    loader.update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # 创建映射文件，包括字符映射和标签映射
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        _c, char_to_id, id_to_char = loader.char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = loader.tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # 预先处理数据，获得含有词和标签的索引
    train_data = loader.prepare_dataset(train_sentences, char_to_id, tag_to_id, FLAGS.lower)
    dev_data = loader.prepare_dataset(dev_sentences, char_to_id, tag_to_id, FLAGS.lower)
    test_data = loader.prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    # 创建存日志和模型的文件
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # 限制GPU内存
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    # 开启会话，开始训练
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, loader.load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(50):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)   # 评估模型
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def evaluate_line(line):
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, loader.load_word2vec, config, id_to_char, logger)
        while True:
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                return result


# def main(_):

    #if FLAGS.train:
     #   if FLAGS.clean:
     #       clean(FLAGS)     # 如果之前训练过，则先清除以前的训练结果
     #   train()
    # else:
     #   evaluate_line()


if __name__ == "__main__":
    tf.app.run(main)