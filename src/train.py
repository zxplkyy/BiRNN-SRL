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

def modify(sequence):
    label2id, id2label = load_dictionary(configure["label2id_file"])
    lastname = ""
    for i in range(len(sequence)):
        tagid = sequence[i]
        tag = id2label[tagid]
        if tag[0] == 'B':
            lastname = tag[2:]
            continue
        if tag[0] == 'I' or tag[0] == "E":
            if tag[2:] != lastname:
                lastname = tag[2:]
                sequence[i] = label2id["B-"+lastname]
    return sequence

def recover(sentence_origin,scores,labels,length,matrix):
    new_inputs = []
    new_preds = []
    new_golds = []
    for line, score, gold, length in zip(sentence_origin, scores, labels, length):
        length = min(length,configure["max_sentence_len"])
        line = line[:length]  # 实际长度
        if configure["use_crf"]:
            pred,_ = tf.contrib.crf.viterbi_decode(score,matrix)
        else:
            pred = np.argmax(score,axis=1)
        pred = modify(pred)
        pred = pred[:length]
        gold = gold[:length]
        # print line, pred, gold
        new_inputs.append(line)
        new_preds.append(pred)
        new_golds.append(gold)
    return new_inputs, new_preds, new_golds
def compare(role1,role2):
    """
    :param role1: RR_model识别出的角色 O,B-role,I-role,re-role
    :param role2: 实际预测出的结果
    :return:
    """
    flag1 = role1[0]
    flag2 = role2[0]
    new_role = role2 #用于做改进
    if flag1 == flag2:
        return new_role
    if flag1 != flag2 and flag1 == "O":
        new_role = role1
    if flag1 != flag2 and flag1 == "B" and flag2 == "I":
        new_role = flag1+role2[1:]
    if flag1 != flag2 and flag1 == "I" and flag2 == "B":
        new_role = flag1+role2[1:]
    return new_role

def transform(inputs,id2word):
    """
    对输入的序列进行转化
    :param inputs:
    :param id2word:
    :return:
    """
    new_inputs = []
    for word in inputs:
        new_inputs.append(id2word[word])
    return new_inputs


def convert(inputs, labels, is_training):
    '''
    用于转换为原来的句子
    :param inputs: word_id
    :param labels: label_id
    :param is_training:
    :return: convert id to original word
    '''
    word_vocab_train = configure["word2id_train"]
    word_vocab_valid = configure["word2id_valid"]
    label_vocab = configure["label2id_file"]
    if is_training:
        word2id,id2word = load_dictionary(word_vocab_train)
    else:
        word2id, id2word = load_dictionary(word_vocab_valid)
    label2id,id2label = load_dictionary(label_vocab)
    # pred_id = label2id.get("rel")
    # pred_pos = 0
    new_inputs = []
    for line, label in zip(inputs, labels):
        new_input = []
        # print(label,"label.................")
        # label_list = label.tolist()
        # pred_pos = label_list.index(pred_id) if pred_id in label_list else 0  # 通过label序列得到谓语所在的位置
        for word_id, label_id in zip(line, label):
            new_input.append(id2word[word_id] + '/' + id2label[label_id])
        new_inputs.append(new_input)
    return new_inputs

def eval(inputs, pred_labels, gold_labels,is_training):
    '''
    :param inputs: sentence
    :param pred_labels: labels predicted by model
    :param gold_labels: True labels
    :param is_training: convert id to words based on train_vocab or valid_vocab
    :return: recall, precision, F1
    '''
    case_true, case_recall, case_precision = 0.0, 0.0, 0.0
    golds = convert(inputs, gold_labels, is_training) #[[word1/label1, word2/label2,...],[],]
    preds = convert(inputs, pred_labels,is_training)
    assert len(golds) == len(preds), "length of prediction file and gold file should be the same."
    for gold, pred in zip(golds, preds):
        lastname = ''
        keys_gold, keys_pred = {}, {}
        for item in gold:
            word, label = item.split('/')[0], item.split('/')[-1]
            flag, name = label[:label.find('-')], label[label.find('-') + 1:]
            if flag == 'O':
                continue
            if flag == 'S':
                if name not in keys_gold:
                    keys_gold[name] = [word]
                else:
                    keys_gold[name].append(word)
            else:
                if flag == 'B':
                    if name not in keys_gold:
                        keys_gold[name] = [word]
                    else:
                        keys_gold[name].append(word)
                    lastname = name
                elif flag == 'I' or flag == 'E':
                    assert name == lastname, "the I-/E- labels are inconsistent with B- labels in gold file."
                    keys_gold[name][-1] += ' ' + word

        for item in pred:
            word, label = item.split('/')[0], item.split('/')[-1]
            flag, name = label[:label.find('-')], label[label.find('-') + 1:]
            if name == 'O':
                continue
            if flag == 'S':
                if name not in keys_pred:
                    keys_pred[name] = [word]
                else:
                    keys_pred[name].append(word)
            else:
                if flag == 'B':
                    if name not in keys_pred:
                        keys_pred[name] = [word]
                    else:
                        keys_pred[name].append(word)
                    lastname = name
                elif flag == 'I' or flag == 'E':
                    assert name == lastname, "the I-/E- labels are inconsistent with B- labels in pred file."
                    keys_pred[name][-1] += ' ' + word
        for key in keys_gold:
            case_recall += len(keys_gold[key])
        for key in keys_pred:
            case_precision += len(keys_pred[key])
        for key in keys_pred:
            if key in keys_gold:
                for word in keys_pred[key]:
                    if word in keys_gold[key]:
                        case_true += 1
                        keys_gold[key].remove(word)  # avoid replicate words
    return case_true, case_precision, case_recall
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
        twX_origin = curword[i * batchSize:endOff]
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
        inputs, preds, golds = recover(twX_origin, unary_score_val, y, test_sequence_length_val, transMatrix)
        case_true, case_precision, case_recall = eval(inputs,preds,golds,False)
        total_true += case_true
        total_precision += case_precision
        total_recall += case_recall
        if case_precision and case_recall:
            precision = 100.0*case_true /float(case_precision)
            recall = 100.0* case_true/float(case_recall)
            if precision and recall:
                f1 = 2.0 * recall * precision / (recall + precision)
                print("Accuracy: %.3f%%" % precision, "recall: %.3f%%" % recall, "f1: %.3f%%" % f1)
    if total_precision and total_recall:
        precision = 100.0 * total_true / float(total_precision)
        recall = 100.0 * total_true / float(total_recall)
        if precision and recall:
            average_f1 = 2.0 * recall * precision / (recall + precision)
            print(" Average Accuracy: %.3f%%" % precision, "Average recall: %.3f%%" % recall, " Average f1: %.3f%%" % average_f1)
    return average_f1

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

























