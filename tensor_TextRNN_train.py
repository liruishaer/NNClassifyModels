#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-16
# function:

import tensorflow as tf
import numpy as np
from tflearn.data_utils import pad_sequences
import os
import word2vec
from tensor_TextRNN_model import TextRNN
import imdb_data


# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size", 2, "number of label")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "size of vocabulary")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.8, "Rate of decay for learning rate.")  # 0.5一次衰减多少
tf.app.flags.DEFINE_integer("num_sampled", 5, "number of noise sampling")  # 使用NCE计算损失函数时需要用到
tf.app.flags.DEFINE_string("ckpt_dir", "fastText_tensor_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 300, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 100, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 10, "embedding size")
tf.app.flags.DEFINE_integer("validate_every", 2, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")  # load pre-trained word embedding
tf.app.flags.DEFINE_string("cache_path", "fast_text_checkpoint/data_cache.pik", "checkpoint location for the model")

config = tf.ConfigProto()
config.gpu_options.allow_growth = False


# TODO  需要加上n-gram特征
# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    # 1. load data (already sequence padding)
    trainX, trainY, testX, testY = load_imdb_data()

    # 2.create session.
    with tf.Session(config=config) as sess:
        # Instantiate Model
        # model = fastTextB(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,FLAGS.decay_rate,
        #                   FLAGS.num_sampled, FLAGS.sentence_len, FLAGS.vocab_size, FLAGS.embed_size,FLAGS.is_training)

        model = TextRNN(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len, FLAGS.vocab_size,
                        FLAGS.embed_size, FLAGS.is_training)

        # Initialize Save
        saver = tf.train.Saver()
        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())
        if FLAGS.use_embedding:  # load pre-trained word embedding
            assign_pretrained_word_embedding(sess, FLAGS.vocab_size, model)
        '''saver = tf.train.Saver()
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
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):  # range(start,stop,step_size)
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                # if epoch == 0 and counter == 0:
                #     # print("trainX[start:end]:", trainX[start:end])
                #     # print("trainY[start:end]:", trainY[start:end])
                #     pass

                curr_loss, curr_acc, _ = sess.run([model.loss_val, model.accuracy, model.train_op],
                                         feed_dict={model.input_x: trainX[start:end],
                                                    model.input_y:  trainY[start:end],
                                                    model.dropout_keep_prob: 0.5})

                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 5 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (
                    epoch, counter, loss / float(counter), acc / float(counter)))

            # epoch increment
            # print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

            # 4.validation  每5轮进行一次验证
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, model, testX, testY, batch_size)
                print("%d/%d Epoch;\tValidation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, FLAGS.num_epochs, eval_loss, eval_acc))

                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=model.epoch_step)  # model.epoch_step

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, model, testX, testY, batch_size)
        print("最终测试结果：\tLoss:%.3f\tAccuracy: %.3f" % (test_loss, test_acc))


def load_imdb_data():
    def load_data(path='imdb.npz', num_words=None, skip_top=0, seed=113, start_char=1, oov_char=2, index_from=3):
        path = 'imdb_npz_data/imdb_train10000_test10000.npz'

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
    (trainX, trainY), (testX, testY) = load_data(num_words=FLAGS.vocab_size)
    print('train_data length:',len(trainX))
    print('test_data length:',len(testX))

    # 2.Data preprocessing      Sequence padding
    print("start padding & transform to one hot...")
    trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length

    # print("testX[0]:", testX[0]);
    # print("testX[1]:", testX[1]);  # [17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
    # # Converting labels to binary vectors
    # print("testY[0]:", testY[0])  # 0 ;print("testY[1]:",testY[1]) #0
    print("end padding & transform to one hot...")
    return trainX,trainY,testX,testY


def assign_pretrained_word_embedding(sess, vocab_size, model):
    print("using pre-trained word emebedding.started...")
    print('Pretrained embeddings GloVe is loading...')

    embeddings_index = {}
    f = open('/liruishaer/Work2/NLP_models/glove.6B/glove.6B.%id.txt'%(FLAGS.embed_size))
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

    embedding_matrix = np.zeros((FLAGS.vocab_size, FLAGS.embed_size))
    word_index = imdb_data.get_word2index_dict()

    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count_exist = count_exist + 1
        else:
            embedding_matrix[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1
    word_embedding_final = embedding_matrix


    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, model, evalX, evalY, batch_size):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        # curr_eval_loss, curr_eval_acc, = sess.run([model.loss_val, model.accuracy],
        #                                           feed_dict={model.sentence: evalX[start:end],
        #                                                      model.labels: evalY[start:end]})

        curr_eval_loss, curr_eval_acc = sess.run([model.loss_val, model.accuracy],
                                          feed_dict={model.input_x: evalX[start:end], model.input_y: evalY[start:end],
                                                     model.dropout_keep_prob: 1})

        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)


if __name__ == "__main__":
    tf.app.run()



