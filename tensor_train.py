#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-16
# function:

import tensorflow as tf
import numpy as np
from tflearn.data_utils import pad_sequences
import imdb_data
from tensor_fasttext_model import fastTextB
from tensor_TextCNN_model import TextCNN
from tensor_TextRNN_model import TextRNN
from tensor_config import fastText_conf,textCNN_conf,textRNN_conf
import time
import csv
import os,sys

config = tf.ConfigProto()
config.gpu_options.allow_growth = False   #让TensorFlow在运行过程中动态申请显存，需要多少就申请多少
MODEL = 'fasttext'   # value:  fasttext   textcnn   textrnn
TRAIN_DATA_SIZE = 10000   # value:  10000      20000     30000     40000

# 模型相关设置
model_conf = None
if MODEL=='fasttext':
    model_conf = fastText_conf
    model = fastTextB(model_conf.label_size, model_conf.learning_rate, model_conf.batch_size, model_conf.decay_steps,
                      model_conf.decay_rate, model_conf.num_sampled, model_conf.sentence_len, model_conf.vocab_size,
                      model_conf.embed_size, model_conf.is_training)
    train_fetches_str = "{'curr_loss':model.loss_val,'curr_acc':model.accuracy,'train_op':model.train_op}"
    eval_fetches = "[model.loss_val, model.accuracy]"
    train_feed_dict_str = "{model.sentence: trainX[start:end],model.labels: trainY[start:end]}"
    eval_feed_dict_str = "{model.sentence: evalX[start:end],model.labels: evalY[start:end]}"
    file_epoch_result_csv = f'tensor_epoch_results/fastText-tf-gpu-train-{TRAIN_DATA_SIZE}-epoch-{model_conf.num_epochs}-batch-{model_conf.batch_size}.csv'
elif MODEL=='textcnn':
    model_conf = textCNN_conf
    model = TextCNN(model_conf.filter_sizes, model_conf.num_filters, model_conf.label_size, model_conf.learning_rate,
                    model_conf.batch_size, model_conf.decay_steps, model_conf.decay_rate, model_conf.sentence_len,
                    model_conf.vocab_size, model_conf.embed_size, model_conf.is_training)
    train_fetches_str = "{'curr_loss':model.loss_val, 'curr_acc':model.accuracy, 'predict':model.predictions, 'W':model.W_projection, 'train_op':model.train_op}"
    eval_fetches = "[model.loss_val, model.accuracy]"
    train_feed_dict_str = "{model.input_x: trainX[start:end], model.input_y: trainY[start:end],model.dropout_keep_prob: model_conf.dropout_keep_prob}"
    eval_feed_dict_str = "{model.input_x: evalX[start:end], model.input_y: evalY[start:end],model.dropout_keep_prob: 1}"
    file_epoch_result_csv = f'tensor_epoch_results/textCNN-tf-gpu-train-{TRAIN_DATA_SIZE}-epoch-{model_conf.num_epochs}-batch-{model_conf.batch_size}.csv'
elif MODEL=='textrnn':
    model_conf = textRNN_conf
    model = TextRNN(model_conf.label_size, model_conf.learning_rate, model_conf.batch_size, model_conf.decay_steps,
                    model_conf.decay_rate, model_conf.sentence_len, model_conf.vocab_size,model_conf.embed_size,
                    model_conf.is_training)
    train_fetches_str = "{'curr_loss':model.loss_val, 'curr_acc':model.accuracy, 'train_op':model.train_op}"
    eval_fetches = "[model.loss_val, model.accuracy]"
    train_feed_dict_str="{model.input_x: trainX[start:end], model.input_y: trainY[start:end],model.dropout_keep_prob: model_conf.dropout_keep_prob}"
    eval_feed_dict_str="{model.input_x: evalX[start:end], model.input_y: evalY[start:end],model.dropout_keep_prob: 1}"
    # file_epoch_result_csv = f'tensor_epoch_results/textRNN-tf-gpu-train-{TRAIN_DATA_SIZE}-epoch-{model_conf.num_epochs}-batch-{model_conf.batch_size}.csv'
    csv_dir = 'results/tensor_epoch_results/'


if not os.path.exists('tensor_epoch_results'):
    os.mkdir('tensor_epoch_results')


# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    # 1. load data (already sequence padding)
    trainX, trainY, testX, testY = load_imdb_data(train_size = TRAIN_DATA_SIZE,vocab_size=model_conf.vocab_size,sentence_len=model_conf.sentence_len)

    # 存储模型文件的路径
    if not os.path.exists(model_conf.ckpt_dir):
        os.mkdir(model_conf.ckpt_dir)
    # 2.create session.
    with tf.Session(config=config) as sess:
        # Instantiate Model....

        # Initialize
        saver = tf.train.Saver()
        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())
        if model_conf.use_embedding:  # load pre-trained word embedding
            assign_pretrained_word_embedding(sess, model_conf.vocab_size, model_conf.embed_size, model)

        '''
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word embedding
                assign_pretrained_word_embedding(sess, FLAGS.vocab_size, model)'''

        curr_epoch = sess.run(model.epoch_step)
        # 3.feed data & training
        number_of_training_data = len(trainX)
        best_val_acc = 0.0

        csv_content_list = [['loss', 'acc', 'time', 'val_loss', 'val_acc']]  # 记录训练结果，用于输出到csv
        total_time = 0   # 记录纯训练过程所用时间

        for epoch_id in range(curr_epoch, model_conf.num_epochs):  # range(start,stop,step_size)
            loss, acc, counter = 0.0, 0.0, 0
            curr_epoch_content_list = []
            for start, end in zip(range(0, number_of_training_data, model_conf.batch_size),
                                  range(model_conf.batch_size, number_of_training_data, model_conf.batch_size)):

                # 训练部分
                start_time = time.time()
                result_dict = sess.run(eval(train_fetches_str),feed_dict=eval(train_feed_dict_str))
                total_time += (time.time() - start_time)

                # 统计数值
                loss, acc, counter = loss + result_dict['curr_loss'], acc + result_dict['curr_acc'], counter + 1

                # 输出batch在训练集上准确率
                if counter % 5 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (epoch_id, counter, loss / float(counter), acc / float(counter)))

            # 训练结果统计
            curr_epoch_content_list += [loss / float(counter), acc / float(counter), total_time]

            # epoch increment
            sess.run(model.epoch_increment)

            # 4.validation  每N轮进行一次验证
            if epoch_id % model_conf.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, model, testX, testY, model_conf.batch_size)
                print("%d/%d Epoch;\tValidation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch_id, model_conf.num_epochs, eval_loss, eval_acc))

                curr_epoch_content_list += [eval_loss,eval_acc]
                csv_content_list.append(curr_epoch_content_list)

                # save model to checkpoint
                if eval_acc > best_val_acc:
                    best_val_acc = eval_acc
                    save_path = model_conf.ckpt_dir + f"model.ckpt-tensor_train_size_{TRAIN_DATA_SIZE}_gpu_epoch_{epoch_id}_batch_{model_conf.batch_size}_acc_{eval_acc:.4f}.pth"
                    saver.save(sess, save_path, global_step=model.epoch_step)  # global_step参数将训练的次数作为后缀加入到模型名字中。

        # 训练/测试结果写入csv文件
        with open(file_epoch_result_csv, 'w') as f:
            w = csv.writer(f)
            for ls in csv_content_list:
                w.writerow(ls)

        # # 5.最后在测试集上做测试，并报告测试准确率 Test
        # test_loss, test_acc = do_eval(sess, model, testX, testY, model_conf.batch_size)
        # print("最终测试结果：\tLoss:%.3f\tAccuracy: %.3f" % (test_loss, test_acc))


def load_imdb_data(train_size, vocab_size,sentence_len):
    def load_data(train_size, num_words=None, skip_top=0, seed=113, start_char=1, oov_char=2, index_from=3):
        path = f'imdb_npz_data/imdb_train{train_size}_test10000.npz'
        # 1. load data
        with np.load(path) as f:
            x_train, labels_train = f['x_train'], f['y_train']
            x_test, labels_test = f['x_test'], f['y_test']

        # 2. shuffle train/test
        np.random.seed(seed)
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        xs = np.concatenate([x_train, x_test])
        labels = np.concatenate([labels_train, labels_test])

        # 保留前3个index
        if start_char is not None:
            xs = [[start_char] + [w + index_from for w in x] for x in xs]
        elif index_from:
            xs = [[w + index_from for w in x] for x in xs]

        if not num_words:
            num_words = max([max(x) for x in xs])

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if oov_char is not None:
            xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
        else:
            xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

        idx = len(x_train)
        x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
        x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

        return (x_train, y_train), (x_test, y_test)

    # 1. load original data
    print('loading data...')
    (trainX, trainY), (testX, testY) = load_data(train_size,num_words=vocab_size)
    print('train_data length:',len(trainX))
    print('test_data length:',len(testX))

    # 2.Data preprocessing      Sequence padding
    print("start padding & transform to one hot...")
    trainX = pad_sequences(trainX, maxlen=sentence_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=sentence_len, value=0.)  # padding to max length

    print("end padding & transform to one hot...")
    return trainX,trainY,testX,testY


def assign_pretrained_word_embedding(sess, vocab_size, embed_size, model):
    print("using pre-trained word emebedding.started...")
    print('Pretrained embeddings GloVe is loading...')

    embeddings_index = {}
    f = open('/liruishaer/Work2/NLP_models/glove.6B/glove.6B.%id.txt'%(embed_size))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors in GloVe embedding' % len(embeddings_index))

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0

    embedding_matrix = np.zeros((vocab_size, embed_size))
    word_index = imdb_data.get_word2index_dict()

    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count_exist = count_exist + 1
        else:
            embedding_matrix[i] = np.random.uniform(-bound, bound, embed_size);
            count_not_exist = count_not_exist + 1
    word_embedding_final = embedding_matrix


    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding, word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, model, evalX, evalY, batch_size):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        # curr_eval_loss, curr_eval_acc = sess.run([model.loss_val, model.accuracy],
        #                                           feed_dict={model.sentence: evalX[start:end],
        #                                                      model.labels: evalY[start:end]})
        curr_eval_loss, curr_eval_acc = sess.run(eval(eval_fetches),feed_dict=eval(eval_feed_dict_str))
        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)


if __name__ == "__main__":
    tf.app.run()



