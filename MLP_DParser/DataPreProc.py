# -*- coding: utf-8 -*-

import MyFeatureModel as f_model
import numpy as np


def convert_to_input_vector(sample_batch, model):
    y_batch = []
    x_left_mor_cnn_batch = []
    x_left_pos_cnn_batch = []
    x_right_mor_cnn_batch = []
    x_right_pos_cnn_batch = []
    x_position_batch = []
    x_child_mor_batch = []
    x_child_pos_batch = []
    x_hc_batch = []

    for sample in sample_batch:
        # word embedding에 들어갈 것은 index 그대로 넣기
        x_left_mor_cnn_batch.append(sample.left_mor_cnn)
        x_left_pos_cnn_batch.append(sample.left_pos_cnn)
        x_right_mor_cnn_batch.append(sample.right_mor_cnn)
        x_right_pos_cnn_batch.append(sample.right_pos_cnn)
        x_child_mor_batch.append(sample.child_mor)
        x_child_pos_batch.append(sample.child_pos)

        x_position_batch.append(model.position_mark) # ????

        # 그외 자질은 zero-one representation
        hc = [0] * model.hc_feature_size
        for idx_hc in sample.hand_crafted:
            hc[idx_hc] = 1 # 있으면 1인듯
        x_hc_batch.append(hc)

        y = [0] * model.type_size
        y[sample.y] = 1
        y_batch.append(y)


    #기본 자질
    x_mor_batch = []
    x_pos_batch = []
    for sample in sample_batch:
        # word embedding에 들어갈 것은 index 그대로 넣기
        x_mor_batch.append(sample.left_mor_cnn + sample.right_mor_cnn)
        x_pos_batch.append(sample.left_pos_cnn + sample.right_pos_cnn)

        y = [0] * model.type_size
        y[sample.y] = 1

    return np.array(x_mor_batch), np.array(x_pos_batch), \
           np.array(x_left_mor_cnn_batch), np.array(x_left_pos_cnn_batch), \
           np.array(x_right_mor_cnn_batch), np.array(x_right_pos_cnn_batch), \
           np.array(x_position_batch), np.array(x_child_mor_batch), np.array(x_child_pos_batch), \
           np.array(x_hc_batch), np.array(y_batch)


def make_sample_from_line(line, model):
    sample = f_model.InputFeature()
    input_list = line.split()

    sample.y = int(input_list[0])

    sample.left_mor_cnn = [0] * model.max_length # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, model.max_length + 1):
        sample.left_mor_cnn[i - 1] = int(input_list[i]) # input_list 1 ~ 12 까지 left_mor_cnn[0~11] 에 넣음

    sample.left_pos_cnn = [0] * model.max_length
    for i in range(model.max_length + 1, model.max_length * 2 + 1):
        sample.left_pos_cnn[i - (model.max_length + 1)] = int(input_list[i])

    sample.right_mor_cnn = [0] * model.max_length
    for i in  range(model.max_length * 2 + 1, model.max_length * 3 + 1):
        sample.right_mor_cnn[i - (model.max_length * 2 + 1)] = int(input_list[i])

    sample.right_pos_cnn = [0] * model.max_length
    for i in  range(model.max_length * 3 + 1, model.max_length * 4 + 1):
        sample.right_pos_cnn[i - (model.max_length * 3 + 1)] = int(input_list[i])

    sample.child_mor = [0] * model.max_length
    for i in  range(model.max_length * 4 + 1, model.max_length * 5 + 1):
        sample.child_mor[i - (model.max_length * 4 + 1)] = int(input_list[i])

    sample.child_pos = [0] * model.max_length
    for i in range(model.max_length * 5 + 1, model.max_length * 6 + 1):
        sample.child_pos[i - (model.max_length * 5 + 1)] = int(input_list[i])

    sample.hand_crafted = [] # hc feature는 길이 정해지지 않음
    for i in range(model.max_length * 6 + 1, len(input_list)):
        sample.hand_crafted.append(int(input_list[i]))

    return sample


def make_sample_from_input_data(file_path, model):
    sample_list = []
    with open(file_path, 'r') as f:


        while True:
            # 한 line이 하나의 코퍼스로부터 뽑아낸 feature
            line = f.readline()
            if not line: break
            # sample 은 이것들은 다시 각각의 feature list로 분할
            sample = make_sample_from_line(line, model)
            sample_list.append(sample)

    return sample_list


class Sentence:
    def __init__(self):
        self.raw_sentence = None
        self.correct_dep_list = []
        self.predict_dep_list = []
        self.eojeol_list = []

    def add_dependency(self, line):
        str_list = line.split()
        dep = Dependency()
        dep.index = int(str_list[0]) - 1 # 인덱스 번호 저장
        dep.head = int(str_list[1]) - 1 # 지배소 번호 저장
        dep.type = str_list[2]  # 타입 저장
        self.correct_dep_list.append(dep)
        eoj = Eojeol()
        mor_list = str_list[4].split('|') # 남/NNP | 과/JC 분리
        for mor in mor_list:
            idx = mor.rfind('/') # 뒤에서부터 / index 찾음
            new_mor = Morpheme()
            new_mor.pos = mor[idx + 1:] # 남/NNP 에서 'NNP'를 저장
            new_mor.lex = mor[:idx] #'남' 저장
            eoj.morpheme_list.append(new_mor) # '남/NNP' 어절 단위로 저장

        self.eojeol_list.append(eoj) # 어절 리스트에 어절 저장

        return

    def get_size(self):
        return len(self.eojeol_list)


class Dependency(object): #의존관계 및 관계 타입 정의
    def __init__(self):
        self.index = -1
        self.head = -1
        self.type = ''

class Eojeol:
    def __init__(self):
        self.morpheme_list = []
        self.raw_eojeol = None


class Morpheme:
    def __init__(self):
        self.lex = '' # 어휘
        self.pos = '' # 형태소태깅
        self.pos = '' # 형태소태깅

class CorpusReader(object):
    def __init__(self):
        self.file = None
        self.str_sentence = None
        self.out_file = None

    def set_file(self, file_path, out_file_path = None):
        self.file = open(file_path, 'r')

        if out_file_path is not None:
            self.out_file = open(out_file_path, 'w')

    def close_files(self):
        if self.file is not None:
            self.out_file.close()
            self.out_file = None

    # train.out 파일로 만들어주는 함수
    def write_out_data(self, total_sample):
        for sample in total_sample:
            #sample.y 는 무슨값? 약간 시작 토큰같은건가?
            line = str(sample.y)
            for val in sample.left_mor_cnn:
                line += ' ' + str(val)
            for val in sample.left_pos_cnn:
                line += ' ' + str(val)

            for val in sample.right_mor_cnn:
                line += ' ' + str(val)

            for val in sample.right_pos_cnn:
                line += ' ' + str(val)

            for val in sample.child_mor:
                line += ' ' + str(val)

            for val in sample.child_pos:
                line += ' ' + str(val)

            for val in sample.hand_crafted:
                line += ' ' + str(val)

            self.out_file.write(line + '\n')


    def get_next(self): #하나의 코퍼스 읽어오는 함수
        sentence = Sentence()
        while True:
            line = self.file.readline()
            if not line: break
            # raw sentence
            if line[0] == ';':
                sentence.raw_sentence = line
            # 1개 corpus 끝
            elif len(line) < 3:
                return sentence
            # 어절 단위 추가
            else:
                sentence.add_dependency(line)

        return None

    def close_file(self):
        self.file.close()



