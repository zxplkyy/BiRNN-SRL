#_*_coding:utf-8_*_
#作者  :zxp
#创建时间  :2019/1/18
#文件  :bilstm.py
#IDE  :PyCharm
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
class SRLModel:
    def __init__(self,
                 numHidden,
                 maxSeqLen,
                 numTags):
        """

        :param numHidden: 隐藏层单元数量
        :param maxSeqLen: 最大序列长度
        :param numTags: 标签数量
        """
        self.num_hidden = numHidden
        self.num_tags = numTags
        self.max_seq_len = maxSeqLen
        self.W = tf.get_variable(
            shape=[numHidden * 2, numTags],
            initializer=tf.contrib.layers.xavier_initializer(),
            name="weights",
            regularizer=tf.contrib.layers.l2_regularizer(0.001))
        self.b = tf.Variable(tf.zeros([numTags], name="bias"))

    def inference(self, X, length, reuse=False):
        """

        :param X:
        :param length:
        :param reuse:
        :return:  [batch_size, self.max_seq_len, self.num_tags]
        """
        # length = tf.reshape(length,shape=[configure["batch_size"]]) #转化成向量的形式
        length = tf.squeeze(length)
        length_32 = tf.cast(length, tf.int32)
        with tf.variable_scope("bilstm", reuse=reuse):
            cell_forward = tf.contrib.rnn.LSTMCell(self.num_hidden,
                                        reuse=reuse)
            forward_output, _ = tf.nn.dynamic_rnn(
                cell_forward,
                X,
                dtype=tf.float32,
                sequence_length=length_32,
                scope="RNN_forward")
            backward_output_, _ = tf.nn.dynamic_rnn(
                cell_forward,
                inputs=tf.reverse_sequence(X,
                                           length_32,
                                           seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backword")

        backward_output = tf.reverse_sequence(backward_output_,
                                              length_32,
                                              seq_dim=1)

        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.num_hidden * 2])
        matricized_unary_scores = tf.matmul(output, self.W) + self.b
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, self.max_seq_len, self.num_tags],
            name="Predict_Prob" if reuse else None)
        return unary_scores,length_32
