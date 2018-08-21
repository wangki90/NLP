# -*- coding: utf-8 -*-

import pickle
import DataPreProc as dp
import MyTransitionParser as tp
import MyFeatureModel as fm

def make_corpus_and_model():

    model = fm.FeatureModel() # Feature model class

    parser = tp.TransitionParse(model)
    cr = dp.CorpusReader()
    cr.set_file("/home/kang/Development/wangki/parsing/MLP_Parser/data/train.txt", "/home/kang/Development/wangki/parsing/MLP_Parser/data/train.out")

    num_data = 0
    # 데이터 계속 읽는 부분
    while True:
        # data = 1개 코퍼스
        data = cr.get_next()
        if data is None:
            break

        num_data += 1
        total_sample = parser.make_gold_corpus(data)
        cr.write_out_data(total_sample)

    print('데이터 생성 : ' + str(num_data) + '문장')
    cr.close_file()

    return model

def save_model():
    f_model = make_corpus_and_model()
    f = open('/home/kang/Development/wangki/parsing/MLP_Parser/model/f_model.dat', 'rw')
    pickle.dump(f_model, f)
    f.close()


save_model()

