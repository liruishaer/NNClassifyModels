#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-16 
# function: 
# fast text. using: very simple model;n-gram to captrue location information;h-softmax to speed up training/inference
# for the n-gram you can use data_util to generate. see method process_one_sentence_to_get_ui_bi_tri_gram under aa1_data_util/data_util_zhihu.py
print("started...")
import tensorflow as tf
import numpy as np

class fastTextB:
    def __init__(self, label_size, learning_rate, batch_size, decay_steps, decay_rate,num_sampled,sentence_len,vocab_size,embed_size,is_training):
        """init all hyperparameter here"""
        # set hyperparamter
        self.label_size = label_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len=sentence_len
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.is_training=is_training
        self.learning_rate=learning_rate

        # add placeholder (X,label)
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")  # X
        self.labels = tf.placeholder(tf.int32, [None], name="Labels")  # y

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.labels) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.get emebedding of words in the sentence
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding,self.sentence)  # [None,self.sentence_len,self.embed_size]

        # 2.average vectors, to get representation of the sentence
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)  # [None,self.embed_size]

        # 3.linear classifier layer
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b #[None, self.label_size]==tf.matmul([None,self.embed_size],[self.embed_size,self.label_size])
        return logits

    def loss(self,l2_lambda=0.01): #0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        if self.is_training: #training
            labels=tf.reshape(self.labels,[-1])               #[batch_size,1]------>[batch_size,]
            labels=tf.expand_dims(labels,1)                   #[batch_size,]----->[batch_size,1]
            loss = tf.reduce_mean( #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                tf.nn.nce_loss(weights=tf.transpose(self.W),  #[embed_size, label_size]--->[label_size,embed_size]. nce_weights:A `Tensor` of shape `[num_classes, dim].O.K.
                               biases=self.b,                 #[label_size]. nce_biases:A `Tensor` of shape `[num_classes]`.
                               labels=labels,                 #[batch_size,1]. train_labels, # A `Tensor` of type `int64` and shape `[batch_size,num_true]`. The target classes.
                               inputs=self.sentence_embeddings,# [None,self.embed_size] #A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                               num_sampled=self.num_sampled,  #scalar. 100
                               num_classes=self.label_size,partition_strategy="div"))  #scalar. 1999
        else:#eval/inference
            #logits = tf.matmul(self.sentence_embeddings, tf.transpose(self.W)) #matmul([None,self.embed_size])--->
            #logits = tf.nn.bias_add(logits, self.b)
            labels_one_hot = tf.one_hot(self.labels, self.label_size) #[batch_size]---->[batch_size,label_size]
            #sigmoid_cross_entropy_with_logits:Computes sigmoid cross entropy given `logits`.Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.  For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot,logits=self.logits) #labels:[batch_size,label_size];logits:[batch, label_size]
            print("loss0:", loss) #shape=(?, 1999)
            loss = tf.reduce_sum(loss, axis=1)
            print("loss1:",loss)  #shape=(?,)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

#test started
def test():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=2
    learning_rate=0.01
    batch_size=10
    decay_steps=1000
    decay_rate=0.9
    sequence_length=5
    vocab_size=10000
    embed_size=100
    is_training=True
    dropout_keep_prob=1
    fastText=fastTextB(num_classes, learning_rate, batch_size, decay_steps, decay_rate,5,sequence_length,vocab_size,embed_size,is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x=np.ones((batch_size,sequence_length),dtype=np.int32) #[None, self.sequence_length]
            input_y=input_y=np.array([1,0,1,0,1,0,1,1,0,1],dtype=np.int32) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss,acc,predict,_=sess.run([fastText.loss_val,fastText.accuracy,fastText.predictions,fastText.train_op],
                                        feed_dict={fastText.sentence:input_x,fastText.labels:input_y})
            print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
# test()
print("ended...")






import tensorflow as tf
import numpy as np
# from p5_fastTextB_model import fastTextB as fastText
# from p4_zhihu_load_data import load_data,create_voabulary,create_voabulary_label
from tflearn.data_utils import pad_sequences
import os
import word2vec
import pickle
import imdb_data

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size",1999,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.8, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_integer("num_sampled",50,"number of noise sampling") #100
tf.app.flags.DEFINE_string("ckpt_dir","fast_text_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",200,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",100,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",15,"embedding size")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("cache_path","fast_text_checkpoint/data_cache.pik","checkpoint location for the model")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #1.load data(X:list of lint,y:int).
    #if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    #else:
    if 1==0:
        trainX, trainY, testX, testY = None, None, None, None
        vocabulary_word2index, vocabulary_index2word = create_voabulary()
        vocab_size = len(vocabulary_word2index)
        vocabulary_word2index_label,_ = create_voabulary_label()
        train, test, _ = load_data(vocabulary_word2index, vocabulary_word2index_label,data_type='train')
        trainX, trainY = train
        testX, testY = test
        # (x_train, y_train), (x_test, y_test) = load_data()
        
        print("testX.shape:", np.array(testX).shape)  # 2500个list.每个list代表一句话
        print("testY.shape:", np.array(testY).shape)  # 2500个label
        print("testX[0]:", testX[0])  # [17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
        print("testX[1]:", testX[1])  #;
        print("testY[0]:", testY[0])  # 0 ;print("testY[1]:",testY[1]) #0

        # 2.Data preprocessing
        # Sequence padding
        print("start padding & transform to one hot...")
        trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
        testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
        ###############################################################################################
        #with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
        #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
        ###############################################################################################
    else:
        (trainX, trainY), (testX, testY) = imdb_data.load_data(num_words=1000)
        vocabulary_index2word = imdb_data.get_index2word_dict()
        vocabulary_word2index = imdb_data.get_word2index_dict()
        vocab_size = 1000
        # 2.Data preprocessing      Sequence padding
        print("start padding & transform to one hot...")
        trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
        testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length

    print("testX[0]:", testX[0]) ;print("testX[1]:", testX[1]); #[17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
    # Converting labels to binary vectors
    print("testY[0]:", testY[0])  # 0 ;print("testY[1]:",testY[1]) #0
    print("end padding & transform to one hot...")


    #2.create session.
    config=tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    config.gpu_options.allow_growth=False
    with tf.Session(config=config) as sess:
        #Instantiate Model
        fast_text=fastTextB(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,FLAGS.num_sampled,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, fast_text)

        curr_epoch=sess.run(fast_text.epoch_step)
        print('curr_epoch:',curr_epoch)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):#range(start,stop,step_size)
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
                    print("trainY[start:end]:",trainY[start:end])
                curr_loss,curr_acc,_=sess.run([fast_text.loss_val,fast_text.accuracy,fast_text.train_op],feed_dict={fast_text.sentence:trainX[start:end],fast_text.labels:trainY[start:end]})
                loss,acc,counter=loss+curr_loss,acc+curr_acc,counter+1
                if counter %500==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter)))

            #epoch increment
            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss, eval_acc=do_eval(sess,fast_text,testX,testY,batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_acc))

                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=fast_text.epoch_step) #fast_text.epoch_step

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc = do_eval(sess, fast_text, testX, testY, batch_size)


def assign_pretrained_word_embedding11(sess,vocabulary_index2word,vocab_size,fast_text):
    print("using pre-trained word emebedding.started...")
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    # word2vec_model = word2vec.load('zhihu-word2vec-multilabel.bin-100', kind='bin')

    word2vec_model=word2vec.load('/liruishaer/Work2/NLP_models/glove.6B/glove.6B.100d.txt')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(fast_text.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")



def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,fast_text):
    print("using pre-trained word emebedding.started...")
    print('Pretrained embeddings GloVe is loading...')

    embeddings_index = {}
    f = open('/liruishaer/Work2/NLP_models/glove.6B/glove.6B.100d.txt')
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

    embedding_matrix = np.zeros((vocab_size, 100))
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


    # -------------------------------------------------------------------------------------
    # word2vec_model=word2vec.load('/liruishaer/Work2/NLP_models/glove.6B/glove.6B.100d.txt')
    # word2vec_dict = {}
    # for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
    #     word2vec_dict[word] = vector
    # word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    # word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    # bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    # count_exist = 0;
    # count_not_exist = 0
    # for i in range(1, vocab_size):  # loop each word
    #     word = vocabulary_index2word[i]  # get a word
    #     embedding = None
    #     try:
    #         embedding = word2vec_dict[word]  # try to get vector:it is an array.
    #     except Exception:
    #         embedding = None
    #     if embedding is not None:  # the 'word' exist a embedding
    #         word_embedding_2dlist[i] = embedding;
    #         count_exist = count_exist + 1  # assign array to this word.
    #     else:  # no embedding for this word
    #         word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
    #         count_not_exist = count_not_exist + 1  # init a random value for the word.
    # word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.

    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(fast_text.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度
def do_eval(sess,fast_text,evalX,evalY,batch_size):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        curr_eval_loss, curr_eval_acc, = sess.run([fast_text.loss_val, fast_text.accuracy],
                                          feed_dict={fast_text.sentence: evalX[start:end],fast_text.labels: evalY[start:end]})
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

def load_data111(cache_file_h5py,cache_file_pickle):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download cache file, it include training data and vocabulary & labels. "
                           "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
                           "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
    print("INFO. cache file exists. going to load cache file")
    f_data = h5py.File(cache_file_h5py, 'r')
    print("f_data.keys:",list(f_data.keys()))
    train_X=f_data['train_X'] # np.array(
    print("train_X.shape:",train_X.shape)
    train_Y=f_data['train_Y'] # np.array(
    print("train_Y.shape:",train_Y.shape,";")
    vaild_X=f_data['vaild_X'] # np.array(
    valid_Y=f_data['valid_Y'] # np.array(
    test_X=f_data['test_X'] # np.array(
    test_Y=f_data['test_Y'] # np.array(
    #print(train_X)
    #f_data.close()

    word2index, label2index=None,None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y

if __name__ == "__main__":
    tf.app.run()

