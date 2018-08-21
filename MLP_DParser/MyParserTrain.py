# -*- coding: utf-8 -*-

'''

'''

import pickle
import random

import numpy as np

import DataPreProc as dp
import tensorflow as tf
import MyTransitionParser as tp

import MyTextCNN as mycnn



tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("mlp_size", 256, "Dimensionality of classifier layer (default: 256)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 32, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("모델 로드 중")
f = open('/home/kang/Development/wangki/parsing/MLP_Parser/model/f_model.dat','rb')
model = pickle.load(f)
parser = tp.TransitionParse(model)

graph = tf.Graph()
with graph.as_default(): # Define operations and tensors in 'graph'
    session_conf = tf.ConfigProto()
    # 실행 과정에서 요구되는 만큼의 GPU 메모리만 할당
    session_conf.gpu_options.allow_growth = True

    sess = tf.Session(config=session_conf)

    with sess.as_default(): # with 사용시, Operation.run(), Tensor.run()이 이 세션에서 실행, sess.close()로 명시적으로 세션을 닫아줘야함
        cnn = mycnn.DepCNNv6(
            num_classes=model.type_size,
            vocab_size=model.mor_size + 1,
            pos_size=model.pos_size + 1,
            hc_size=model.hc_feature_size,
            embedding_size=FLAGS.embedding_dim,
            mlp_size=FLAGS.mlp_size,
            l2_reg_lambda=FLAGS.l2_reg_lambda
        )

        def train_step(x_mor, x_pos, x_child_mor, x_child_pos, x_hc, y1_batch):

            feed_dict = {
                cnn.input_x_mor: x_mor,
                cnn.input_x_pos: x_pos,
                cnn.input_x_child_mor: x_child_mor,
                cnn.input_x_child_pos: x_child_pos,
                cnn.input_x_hc : x_hc,
                cnn.input_y: y1_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _ = sess.run(
                [train_op], feed_dict)

            return

        def test_step(x_mor, x_pos, x_child_mor, x_child_pos, x_hc):
            feed_dict = {
                cnn.input_x_mor : x_mor,
                cnn.input_x_pos : x_pos,
                cnn.input_x_child_mor : x_child_mor,
                cnn.input_x_child_pos : x_child_pos,
                cnn.input_x_hc : x_hc,
                cnn.dropout_keep_prob: 1.0 #dropout 사용 x
            }
            # Returns the Operation with the given 'output/predictions'.
            # Get the placeholders from the graph by name
            predictions = graph.get_operation_by_name("Output/predictions").outputs[0] # ?

            predictions = sess.run(
                [predictions],
                feed_dict=feed_dict
            )

            return predictions


        #Define Training procedure
        optimizer = tf.train.AdamOptimizer(0.0001) #optimizer 설정
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars)

        # 나중에 추가적인 트레이닝이나 평가를 위해 모델을 복구하는데 쓰일 수 있는 checkpoint 파일을 내보내기 위해서
        # tf.train.Saver 인스턴스 생성
        # Create a Saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Before starting, initialize the variables.
        init = tf.global_variables_initializer()

        # Launch the graph.
        sess.run(init)

        print('데이터 로드 중')
        sample_list = dp.make_sample_from_input_data('/home/kang/Development/wangki/parsing/MLP_Parser/data/train.out', model)
        print('데이터 로드 끝')

        total_batch = len(sample_list) / FLAGS.batch_size
        print('total_batch = ' + str(total_batch))

        for ep in range(FLAGS.num_epochs):
            # 학습 phase
            # sample shuffle
            random.shuffle(sample_list)
            # batch 만큼 데이터 가져오기
            batch_number = 0
            while True:
                sample_batch = sample_list[batch_number * FLAGS.batch_size : (batch_number + 1) * FLAGS.batch_size] # 0 ~ 64 -> 65 ~ 128 -> 생략
                batch_number += 1

                x_mor, x_pos, x_left_mor, x_left_pos, x_right_mor, x_right_pos, \
                x_position_mark_batch, x_child_mor, x_child_pos, x_hc_batch, y_batch\
                    = dp.convert_to_input_vector(sample_batch, model)

                train_step(x_mor, x_pos, x_child_mor, x_child_pos, x_hc_batch, y_batch)

                if batch_number * FLAGS.batch_size > len(sample_list):
                    break

            # print(str(ep+1) + 'epoch trained')

            #테스트 phase

            total_arc = 0
            correct_arc = 0
            correct_sentence = 0
            total_sentence = 0
            correct_arc_with_tag = 0
            correct_sentence_with_tag = 0


            cr_test = dp.CorpusReader()
            cr_test.set_file('/home/kang/Development/wangki/parsing/MLP_Parser/data/test.txt')

            # 문장을 배치 사이즈만큼 한번에 분석하도록...

            while True:
                data = cr_test.get_next()
                if data is None:
                    break
                parser.initialize(data)
                while not parser.is_final_state():
                    left_mor, left_pos, right_mor, right_pos, child_mor, child_pos, hc = parser.make_input_vector(data)
                    x_mor = left_mor + right_mor
                    x_pos = left_pos + right_pos

                    hc = model.convert_to_zero_one(hc, model.hc_feature_size)
                    next_action = test_step(np.array([x_mor]), np.array([x_pos]), np.array([child_mor]),
                                            np.array([child_pos]), np.array([hc]))

                    next_action = next_action[0][0]  # 어떤 index 값인듯
                    parser.run_action(next_action, model)

                # 성.능.평.가
                predicts = parser.get_result()  # transition parser가 예측한 의존관계 결과 predict list
                golds = data.correct_dep_list  # 실제 데이터의 정답 의존관계

                sentence_flag = True
                sentence_with_tag_flag = True

                for i in range(len(predicts)):
                    if predicts[i].head == golds[i].head:  # 예측한 지배소 번호와 실제 지배소의 번호가 일치할 경우
                        correct_arc += 1
                        if predicts[i].type == golds[i].type:  # 타입까지 맞을경우
                            correct_arc_with_tag += 1
                        else:
                            sentence_with_tag_flag = False
                    else:
                        sentence_flag = False
                        sentence_with_tag_flag = False
                    total_arc += 1  # 전체 arc의 개수 -> 한 문장 내에 여러개의 arc가 나옴

                if sentence_flag is True:
                    correct_sentence += 1  # 모든 지배소 번호가 일치한 경우,
                if sentence_with_tag_flag is True:
                    correct_sentence_with_tag += 1

                total_sentence += 1  # 전체 문장의 개수

                if (total_sentence % 300) == 0:
                    pass
                    # print('Check point')

            cr_test.close_file()

            if ep is 0:
                print('total_arc = ' + str(total_arc), 'total_sentence = ' + str(total_sentence))

            if (ep % FLAGS.num_checkpoints) is 0:
                path = saver.save(sess, '/home/kang/Development/wangki/parsing/MLP_Parser/data/chk_point/')
                print('Saved model checkpoint to {}\n'.format(path)) # format 함수를 사용하면 {} 부분에 path 값이 들어감

            print('epoch = ' + str(ep + 1) + ', acc = ' + str(correct_arc / float(total_arc))
                + ', sen_acc = ' + str(correct_sentence / float(total_sentence))
                + ', acc_with_tag = ' + str(correct_sentence_with_tag / float(total_arc))
                + ', sen_acc_with_tag = ' + str(correct_sentence_with_tag / float(total_sentence))
                )
                
























        
