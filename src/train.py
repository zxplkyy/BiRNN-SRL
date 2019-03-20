import tensorflow as tf
from src.config import configure
from src.bilstm import SRLModel
from src.data_utils import load_dictionary,train_data_loader,load_dictionaries
import numpy as np
import logging
class Chinese_SRL:
    def __init__(self,configure):
        self.configure = configure
        self.classes_num = configure["n_label"]
        self.hidden_num = configure["RNN_dim"]  # LSTM 隐藏层单元数量
        self.max_sentence_len = configure["max_len"]
        self.word_embedding_size = configure["embedding_dim"]
        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(
                shape=[self.hidden_num * 2, self.classes_num],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            self.b = tf.Variable(tf.zeros([self.classes_num], name="bias"))
        with tf.variable_scope("embedding"):
            embeddings = np.random.uniform(-1, 1, (self.configure["vocab_size"], self.configure['embedding_dim']))
            self.word_embedding = tf.get_variable("Embed_word",
                                                  shape=(self.configure['vocab_size'], self.configure['embedding_dim']),
                                                  dtype='float32', initializer=tf.constant_initializer(embeddings))
            self.postag_embedding = tf.get_variable("Embed_postag", shape=(34, self.configure['postag_dim']),
                                                    dtype='float32',
                                                    initializer=tf.truncated_normal_initializer(
                                                        stddev=0.01))  # postag size is fixed to 32+2(eos, bos)
            self.distance_embedding = tf.get_variable("Embed_dist", shape=(240, self.configure['distance_dim']),
                                                      dtype='float32',
                                                      initializer=tf.truncated_normal_initializer(
                                                          stddev=0.01))  # we observed in training set max dist is 240
            self.mark_embedding = tf.get_variable(
                name="mark_embedding_matrix", shape=[3, self.configure["embedding_dim"]],
                initializer=tf.random_normal_initializer(-0.05, 0.05), dtype=tf.float32)
        self.curword = tf.placeholder(dtype="int32", shape=(None, self.max_sentence_len))
        self.lastword = tf.placeholder(dtype="int32", shape=(None,self.max_sentence_len))
        self.nextword = tf.placeholder(dtype="int32", shape=(None, self.max_sentence_len))
        self.predicate = tf.placeholder(dtype="int32", shape=(None, self.max_sentence_len))
        self.curpostag = tf.placeholder(dtype="int32", shape=(None, self.max_sentence_len))
        self.lastpostag = tf.placeholder(dtype="int32", shape=(None, self.max_sentence_len))
        self.nextpostag = tf.placeholder(dtype="int32", shape=(None, self.max_sentence_len))
        self.length = tf.placeholder(tf.int32, shape=[None, 1], name="input_length")
        self.distance = tf.placeholder(dtype="int32", shape=(None, self.configure['max_len']))
        #self.seq_length = tf.placeholder(dtype="int32", shape=(None,))
        self.model = SRLModel(
            self.hidden_num, self.max_sentence_len, self.classes_num)


    def inference(self,curword,lastword,nextword,predicate,curpostag,lastpostag,nextpostag,dist,length,reuse=None,trainModel = True):
        """

        :param curword:
        :param lastword:
        :param nextword:
        :param predicate:
        :param curpostag:
        :param lastpostag:
        :param nextpostag:
        :param length:
        :param reuse:
        :param trainModel:
        :return:
        """
        curword_emb = tf.nn.embedding_lookup(self.word_embedding, curword)
        lastword_emb = tf.nn.embedding_lookup(self.word_embedding, lastword)
        nextword_emb = tf.nn.embedding_lookup(self.word_embedding, nextword)
        predicate_emb = tf.nn.embedding_lookup(self.mark_embedding, predicate)
        curpos_emb = tf.nn.embedding_lookup(self.postag_embedding, curpostag)
        lastpos_emb = tf.nn.embedding_lookup(self.postag_embedding, lastpostag)
        nextpos_emb = tf.nn.embedding_lookup(self.postag_embedding, nextpostag)
        dist_emb=tf.nn.embedding_lookup(self.distance_embedding, dist)
        word_vectors = tf.concat([curword_emb, lastword_emb, nextword_emb, curpos_emb, lastpos_emb, nextpos_emb,predicate_emb, dist_emb],axis=2)
        unary_scores, length = self.model.inference(word_vectors, length,
                                                    reuse)  # [batch_size, self.max_seq_len, self.num_tags]
        return unary_scores, length

    def test_unary_score(self):
        P,sequence_length = self.inference(self.curword,self.lastword,self.nextword,self.predicate,self.curpostag,self.lastpostag,
                                           self.nextpostag,self.distance,self.length,reuse=True,trainModel=False)
        return P, sequence_length
    def loss(self,curword,lastword,nextword,predicate,curpostag,lastpostag,nextpostag,dist,length,Y):
        P, sequence_length = self.inference(curword,lastword,nextword,predicate,curpostag,lastpostag,
                                           nextpostag,dist,length)
        log_likehood, self.trains_params = tf.contrib.crf.crf_log_likelihood(P, Y, sequence_length)
        loss = tf.reduce_mean(-log_likehood)
        tf.summary.scalar('loss', loss)
        return loss

def cal_true_accuracy(gold_sequence,pred_sequence):
    """

    :param gold_sequence:
    :param pred_sequence:预测结果
    :return:
    """
    correct_labels = 0
    total_labels = 0
    predict_label = 0

    label2idx, idx2label= load_dictionary(configure["label2id_file"])
    predicate_id = label2idx["rel"]
    O_id = label2idx["O"]
    assert(len(gold_sequence)==len(pred_sequence))
    for i in range(len(pred_sequence)):
#         if gold_sequence[i] != predicate_id and gold_sequence[i] != O_id:
          if pred_sequence[i] != predicate_id and pred_sequence[i] != O_id:
              predict_label += 1
    for i in range(len(gold_sequence)):
        if gold_sequence[i] != predicate_id and gold_sequence[i] != O_id:
            total_labels += 1
            if gold_sequence[i] == pred_sequence[i]:
                correct_labels += 1
        else:
            continue
    return correct_labels, total_labels, predict_label
def test_evaluate(sess, unary_score,test_sequence_length,transMatrix,
                  input_curword,input_lastword,input_nextword,input_predicate,input_curpostag,input_lastpostag,input_nextpostag,input_dist,input_length,
                  curword,lastword,nextword,predicate,curpostag,lastpostag,nextpostag,dist,length,tY
                  ):
    """
    计算一个batch的结果
    :param sess:
    :param unary_score:
    :param test_sequence_length:
    :param transMatrix:
    :param input_curword:
    :param input_lastword:
    :param input_nextword:
    :param input_predicate:
    :param input_curpostag:
    :param input_lastpostag:
    :param input_nextpostag:
    :param input_dist:
    :param input_length:
    :param curword:
    :param lastword:
    :param nextword:
    :param predicate:
    :param curpostag:
    :param lastpostag:
    :param nextpostag:
    :param dist:
    :param length:
    :param tY:
    :return:
    """
    batchSize = configure["batch_size"]
    totalLen = curword.shape[0]
    numBatch = int((totalLen-1)/batchSize)+1
    correct_labels = 0
    total_labels = 0
    predict_labels = 0 #所有预测出来的标签
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = tY[i * batchSize:endOff]
        feed_dict = {input_curword: curword[i * batchSize:endOff],
                     input_lastword: lastword[i * batchSize:endOff],
                     input_nextword: nextword[i * batchSize:endOff],
                     input_predicate:predicate[i * batchSize:endOff],
                     input_curpostag: curpostag[i*batchSize:endOff],
                     input_lastpostag:lastpostag[i*batchSize:endOff],
                     input_nextpostag:nextpostag[i*batchSize:endOff],
                     input_dist:dist[i*batchSize:endOff],
                     input_length: length[i * batchSize:endOff]
                     }
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)
        for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_sequence_length_val):
            # print("seg len:%d" % (sequence_length_))
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_] #实际值
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, transMatrix)
            y_ = y_.tolist()
            correct_label, total_label, predict_label = cal_true_accuracy(y_, viterbi_sequence)
            correct_labels += correct_label
            total_labels += total_label
            predict_labels += predict_label
        precision = 0
        recall = 0
        f1 = 0
        if predict_labels != 0:
            precision = 100.0 * correct_labels / float(predict_labels)
        if total_labels != 0:
            recall = 100.0 * correct_labels / float(total_labels)
        if correct_labels != 0:
            f1 = 2.0 * recall * precision / (recall + precision)
        print("Accuracy: %.3f%%" % precision, "recall: %.3f%%" % recall, "f1: %.3f%%" % f1)
        return f1

def main(unused_argv):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    model = Chinese_SRL(configure=configure)
    print("Loading data...")
    dicts = load_dictionaries(configure)  # word2idx, idx2word, postag2idx, idx2postag, label2idx, idx2label
    training_data, feats, labels = train_data_loader(dicts, configure, configure["training_data_path"])
    val_data, val_feats, val_labels = train_data_loader(dicts, configure, configure["validate_data_path"])
    # Training
    curword = tf.placeholder(dtype="int32", shape=(None, configure["max_len"]),name="curword")
    lastword = tf.placeholder(dtype="int32", shape=(None, configure["max_len"]),name="lastword")
    nextword = tf.placeholder(dtype="int32", shape=(None, configure["max_len"]))
    predicate = tf.placeholder(dtype="int32", shape=(None, configure["max_len"]))
    curpostag = tf.placeholder(dtype="int32", shape=(None, configure["max_len"]))
    lastpostag = tf.placeholder(dtype="int32", shape=(None, configure["max_len"]))
    nextpostag = tf.placeholder(dtype="int32", shape=(None, configure["max_len"]))
    dist = tf.placeholder(dtype="int32", shape=(None, configure["max_len"]),name="distance")
    seq_length = tf.placeholder(tf.int32, shape=[None, 1],name="seq_length")
    Y = tf.placeholder(tf.int32, [None, configure["max_len"]])
    total_loss = model.loss(curword,lastword,nextword,predicate,curpostag,lastpostag,nextpostag,dist,seq_length,Y)
    tf.summary.scalar('total_loss', total_loss)
    train_op = tf.train.AdamOptimizer(configure["lrate"]).minimize(total_loss)
    test_unary_score, test_sequence_length = model.test_unary_score()
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    lengths = np.array([len(sent) for sent in training_data], dtype=np.int32)#[一维数组]
    val_lengths = np.array([len(sent) for sent in val_data], dtype=np.int32)
    with tf.Session() as sess:
        sess.run(init)
        rng = np.random.RandomState(seed=1701214021)
        bestF1 = 0
        for epoch in range(configure['num_spoch']):
            logging.info("Epoch %d started:" % epoch)
            ids = np.arange(len(training_data))
            rng.shuffle(ids)
            batch_num = len(training_data) // configure['batch_size']
            print("batch_num",batch_num)
            for iter in range(batch_num):
                data_id = ids[iter * configure['batch_size']:(iter + 1) * configure['batch_size']]
                features = feats[data_id] #三维数组
                length = lengths[data_id]#一维数组
                length = length[:,np.newaxis]
                label = labels[data_id]#二维数组
                feed_dict = {
                    curword: features[:, :, 0],
                    lastword: features[:, :, 1],
                    nextword: features[:, :, 2],
                    predicate: features[:, :, 3],
                    curpostag: features[:, :, 4],
                    lastpostag: features[:, :, 5],
                    nextpostag: features[:, :, 6],
                    dist:features[:, :, 7],
                    seq_length: length,
                    Y: label
                }
                t_loss, _, trainsMatrix, summary, g_step = sess.run([total_loss,train_op,model.trains_params,merged,global_step],feed_dict=feed_dict)
                if iter % 100 == 0:
                    print("[%d] loss: [%r]" % (iter, t_loss))
                    # cal validation
                    val_features = val_feats
                    val_length = val_lengths
                    val_length = val_length[:, np.newaxis]
                    val_label = val_labels
                    val_curword = val_features[:, :, 0]
                    val_lastword = val_features[:, :, 1]
                    val_nextword = val_features[:, :, 2]
                    val_predicate = val_features[:, :, 3]
                    val_curpostag = val_features[:, :, 4]
                    val_lastpostag = val_features[:, :, 5]
                    val_nextpostag = val_features[:, :, 6]
                    val_dist = val_features[:, :, 7]
                    val_sequence = val_length
                    val_Y = val_label
                    f1 = test_evaluate(sess, test_unary_score, test_sequence_length, trainsMatrix, model.curword,
                           model.lastword, model.nextword, model.predicate
                              ,model.curpostag,model.lastpostag,model.nextpostag,model.distance,model.length,
                             val_curword,val_lastword,val_nextword,val_predicate,val_curpostag,val_lastpostag,val_nextpostag,val_dist,val_sequence,val_Y)
                    if f1 > bestF1:
                        bestF1 = f1
if __name__ == '__main__':
    tf.app.run()

























