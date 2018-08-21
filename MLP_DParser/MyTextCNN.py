# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# Graph.
class DepCNNv6(object):

    # num_classes -> Type 종류의 개수, vocab_size -> 총 어절 수, pos_size -> 총 pos 사이즈, hc_size -> 총 hc 사이즈
    def __init__(
            self, num_classes, vocab_size, pos_size, hc_size,
            embedding_size, mlp_size, l2_reg_lambda=0.0):
        # 윈도우 3개, 어절 앞2개 뒤2개, stack, queue
        self.input_x_mor = tf.placeholder(tf.int32, [None, 3 * 4 * 2])
        self.input_x_pos = tf.placeholder(tf.int32, [None, 3 * 4 * 2])

        # stack의 top3 어절의 앞2개 뒤2개
        self.input_x_child_mor = tf.placeholder(tf.int32, [None, 3 * 4])
        self.input_x_child_pos = tf.placeholder(tf.int32, [None, 3 * 4])

        self.input_x_hc = tf.placeholder(tf.float32, [None, hc_size]) # 모델의 hc_size

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y') # num_classes -> 의존관계(타입) 개수
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"): # Embedding Scope
            self.W_MOR = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_MOR' # -1.0 ~ 1.0 랜덤
            )
            # lookup 테이블을 이용 [vocab_size , embedding_size] -> [ input_x_mor , embedding_size ] 차원으로 바꿔줌
            self.embedded_chars_mor = tf.nn.embedding_lookup(self.W_MOR, self.input_x_mor)

            self.W_POS = tf.Variable(
                tf.random_uniform([pos_size, embedding_size], -1.0, 1.0), name='W_POS'
            )
            # [pos_size, embedding_size] -> [input_x_pos, embedding_size]
            self.embedded_chars_pos = tf.nn.embedding_lookup(self.W_POS, self.input_x_pos)

            self.embedded_chars_child_mor = tf.nn.embedding_lookup(self.W_MOR, self.input_x_child_mor)
            self.embedded_chars_child_pos = tf.nn.embedding_lookup(self.W_POS, self.input_x_child_pos)

        # tf.reshape(tensor, shape, name=None) , 텐서의 구조를 변형
        # Ex) tensor 't' = [[[1, 1, 1],
        #                     [2, 2, 2]],
        #                    [[3, 3, 3],
        #                     [4, 4, 4]]] 일 때
        # reshape(t, [-1]) ==> 평평하게 핌 ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        self.mor_flat = tf.reshape(self.embedded_chars_mor, [-1, embedding_size * 3 * 4 * 2])
        self.pos_flat = tf.reshape(self.embedded_chars_pos, [-1, embedding_size * 3 * 4 * 2])

        self.child_mor_flat = tf.reshape(self.embedded_chars_child_mor, [-1, embedding_size * 3 * 4])
        self.child_pos_flat = tf.reshape(self.embedded_chars_child_pos, [-1, embedding_size * 3 * 4])

        # tf.concat(values, axis, name='concat')
        # t1 = [[1, 2, 3], [4, 5, 6]]
        # t2 = [[7, 8, 9], [10, 11, 12]]
        # tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
        self.h_flat = tf.concat([self.mor_flat, self.pos_flat, self.child_mor_flat, self.child_pos_flat], 1)

        # Add dropout
        with tf.name_scope("dropout"):
            self._h_drop = tf.nn.dropout(self.h_flat, self.dropout_keep_prob)
            self.h_drop = tf.concat([self._h_drop, self.input_x_hc], 1)

        # Final (unnormalized) scores and predictions
        # Graph 'output'
        with tf.name_scope("Output"):

            W1 = tf.get_variable("W1", shape=[embedding_size * 3 * 4 * 2 * 2 + embedding_size * 3*4*2+hc_size, mlp_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[mlp_size]), name="b1")
            L1 = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop, W1, b1))
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            L1_drop = tf.nn.dropout(L1, self.dropout_keep_prob)

            # W2 = tf.get_variable("W2", shape=[mlp_size, mlp_size],
            #                      initializer=tf.contrib.layers.xavier_initializer())
            # b2 = tf.Variable(tf.constant(0.1, shape=[mlp_size]), name="b2")
            # L2 = tf.nn.relu(tf.nn.xw_plus_b(L1_drop, W2, b2))
            # l2_loss += tf.nn.l2_loss(W2)
            # l2_loss += tf.nn.l2_loss(b2)
            # L2_drop = tf.nn.dropout(L2, self.dropout_keep_prob)

            Wout = tf.get_variable("Wout", shape=[mlp_size, num_classes],
                                 initializer=tf.contrib.layers.xavier_initializer())
            bout = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="bout")
            l2_loss += tf.nn.l2_loss(Wout)
            l2_loss += tf.nn.l2_loss(bout)

            # self.scores = tf.nn.softmax(tf.nn.xw_plus_b(L2_drop, Wout, bout))
            self.scores = tf.nn.relu(tf.nn.xw_plus_b(L1_drop, Wout, bout), name="scores")
            # self.scores = tf.nn.xw_plus_b(L2_drop, Wout, bout, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy1")



