# -*- coding: utf-8 -*-

import DataPreProc as dp

class FeatureModel:
    def __init__(self):
        self.type_map = {} #dic(map)
        self.type_size = 0
        self.num_actions = 2
        self.rev_type_map = {}
        self.mor_map = {}
        self.mor_size = 0 #형태소 개수
        self.pos_map = {}
        self.pos_size = 0
        self.window_size = 3 # window size
        self.max_length = self.window_size * 4 # buffer, stack, 2개 어절, 2개 POS
        self.feature_size = 0 # feature 개수

        # hand crafted feature 초기화
        self.hc_feature_map = {} # dic
        self.hc_feature_size = 0

        # empty를 위한 더미 쌓기 -> 그냥 더미값 넣어주는거임
        self.empty_dummy = dp.Morpheme() #Morpheme
        self.empty_dummy.lex = '$empty' #lexical = empty
        self.empty_dummy.pos = '$empty' #pos = $empty
        str_mor = self.empty_dummy.lex + '/' + self.empty_dummy.pos
        self.mor_map[str_mor] = self.mor_size + 1 # mor_map의 형태 -> [형태소/POS : 형태소 번호]
        self.mor_size += 1

        # self.pos_map[self.empty_dummy.pos] = self.pos_size + 1
        # self.pos_size += 1

        # position_mark? [sequence_len, sequence_len]
        self.position_mark = [] #list

        for i in range(self.max_length): # position mark initialize
            _position_mark = [float(0)] * self.max_length # 0.0 * 12
            _position_mark[i] = float(1) # _position_mark[0~11] = 1.0
            self.position_mark.append(_position_mark)

        # root를 위한 더미 넣기

        self.root_dummy = dp.Morpheme() #root 더미 넣기
        self.root_dummy.lex = '$root'
        self.root_dummy.pos = '$root'
        str_mor = self.root_dummy.lex + '/' + self.root_dummy.pos
        self.mor_map[str_mor] = self.mor_size + 1 #[형태소/POS : 형태소/POS 번호]
        self.mor_size += 1
        self.pos_map[self.root_dummy.pos] = self.pos_size + 1 #[ POS : POS 번호]
        self.pos_size += 1

    def get_feature_size(self):
        size = 0
        #morph 입력
        size += self.mor_size * 2 # mor/mor 2개씩 있으니까?
        #pos 입력
        size += self.pos_size * 2 # pos/pos 2개씩 있으니까?
        #hc
        size += self.hc_feature_size

        return size

    def add_feature(self, sentence):
        for i in range(len(sentence.eojeol_list)):
            dep_type = sentence.correct_dep_list[i]
            if not dep_type in self.type_map:
                # tag 추가
                self.type_map[dep_type] = self.type_size
                self.rev_type_map[self.type_size] = [dep_type]
                self.type_size += 1

            eoj = sentence.eojeol_list[i]
            for mor in eoj.morpheme_list:
                str_mor = mor.lex + '/' + mor.pos

                if not str_mor in self.mor_map:
                    # feature 추가
                    self.mor_map[str_mor] = self.mor_size + 1
                    self.mor_size += 1

                str_pos = mor.pos

                if not str_pos in self.pos_map:
                    #feature 추가
                    self.pos_map[str_pos] = self.pos_size + 1
                    self.pos_size += 1

    # 해당 어절이 몇번 어절인지, 어절마다 순차별로 인덱스를 붙임
    def get_mor_feature_idx(self, mor, mode):
        if mor in self.mor_map: # mor_map dic 에 mor 이 있으면
            return self.mor_map.get(mor) # 해당 mor 값을 리턴
        elif mode == 'train': # train 모드일 경우, train 할 때는 모두 이 경우로 들어옴, mor map 에 추가
            self.mor_map[mor] = self.mor_size + 1
            self.mor_size += 1
            return self.mor_map.get(mor)
        else:
            return 0

    def get_pos_featrue_idx(self, pos, mode):
        if pos in self.pos_map:
            return self.pos_map.get(pos)
        elif mode == 'train':
            self.pos_map[pos] = self.pos_size + 1
            self.pos_size += 1
            return self.pos_map.get(pos)
        else:
            return 0

    def get_hc_feature_idx(self, hc, mode):
        # train mode가 아닐 경우에는 hc_feature_map에 저장 된 index 값을 가져오면 됨
        if hc in self.hc_feature_map:
            return self.hc_feature_map.get(hc)
        # train mode일 경우, 맵에 저장, hc_feature의 크기만큼 index를 붙임
        elif mode == 'train':
            self.hc_feature_map[hc] = self.hc_feature_size
            self.hc_feature_size += 1
            return self.hc_feature_map.get(hc)
        else:
            return 0

    def get_type_idx(self, type):
        if type in self.type_map:
            return self.type_map.get(type)
        else:
            self.type_map[type] = self.type_size
            self.rev_type_map[self.type_size] = type
            self.type_size += 1
            return self.type_map.get(type)
    
    # 해당 인덱스의 type을 가져옴
    def get_str_type(self, idx):
        return self.rev_type_map.get(idx)


    def make_feature_vector(self, state, data, mode='test'): #state = 현재 상태, data = 한개 코퍼스, mode = 'test'

        # 입력 만들기
        # left_mor_cnn_input 에 저장되는 값 = Buffer 쪽 Top과(포함) 가까운 2개의 어절 정보를 저장
        # 한 어절당 4개의 형태소를 저장 할 수 있음 -> left_mor_cnn_input []의 공간 4개를 할당받음
        left_mor_cnn_input = [0] * self.max_length # 왼쪽 mor [0.0] * 12개
        left_pos_cnn_input = [0] * self.max_length # 왼쪽 pos [0.0] * 12개
        nth_queue = 0    #buffer
        idx_eoj = state.nth_queue(nth_queue) # Buffer의 0번째 있는 어절의 인덱스, 즉 맨 끝 어절의 index (10)
        idx_input = self.max_length - 1 # max length = 12, idx_input = 11


        while idx_input > 0:

            eoj = data.eojeol_list[idx_eoj]  # eoj 에 eojeol_list에 저장된 마지막 값을 불러옴
            mor_list = eoj.morpheme_list # mor_list 에 현재 어절의 morpheme_list를 복사, Ex) [lex:조화, pos:NNG] , [lex:다, pos:EF], [lex:., pos:SF]

            # get_mor_feature_idx return value = 현재 어절/POS 의 인덱스
            idx_mor0 = self.get_mor_feature_idx(mor_list[0].lex + '/' + mor_list[0].pos, mode) # 맨 앞 형태소, 조화/NNG 인덱스 번호
            idx_mor1 = self.get_mor_feature_idx(self.empty_dummy.lex + '/' + self.empty_dummy.pos, mode)
            idx_mor2 = self.get_mor_feature_idx(self.empty_dummy.lex + '/' + self.empty_dummy.pos, mode)
            idx_mor3 = self.get_mor_feature_idx(mor_list[-1].lex + '/' + mor_list[-1].pos, mode) # 맨 뒤 형태소, ./JX 의 인덱스 번호

            idx_pos0 = self.get_pos_featrue_idx(mor_list[0], mode)
            idx_pos1 = self.get_pos_featrue_idx(self.empty_dummy, mode)
            idx_pos2 = self.get_pos_featrue_idx(self.empty_dummy, mode)
            idx_pos3 = self.get_pos_featrue_idx(mor_list[-1], mode)

            if len(mor_list) >= 2:  # 저장된게 2개 이상일 경우,
                idx_mor1 = self.get_mor_feature_idx(mor_list[1].lex + '/' + mor_list[1].pos, mode) # 가운/NNG 의 인덱스 번호
                idx_mor2 = self.get_mor_feature_idx(mor_list[-2].lex + '/' + mor_list[-2].pos, mode) # 가운/NNG 의 인덱스 번호

                idx_pos1 = self.get_pos_featrue_idx(mor_list[1].pos, mode)
                idx_pos2 = self.get_pos_featrue_idx(mor_list[-2].pos, mode)


            left_mor_cnn_input[idx_input] = idx_mor3
            left_pos_cnn_input[idx_input] = idx_pos3
            idx_input -= 1

            left_mor_cnn_input[idx_input] = idx_mor2
            left_pos_cnn_input[idx_input] = idx_pos2
            idx_input -= 1

            left_mor_cnn_input[idx_input] = idx_mor1
            left_pos_cnn_input[idx_input] = idx_pos1
            idx_input -= 1

            left_mor_cnn_input[idx_input] = idx_mor0
            left_pos_cnn_input[idx_input] = idx_pos0
            idx_input -= 1

            nth_queue += 1

            idx_eoj = state.nth_queue(nth_queue) # idx_eoj = nth_queue 번째에 있는 어절(index), queue에는 몇번째 어절인지 번호만 저장
            if idx_eoj is None:
                break

        # stack 쪽 어절 정보(인덱스) 를 저장하는 듯
        right_mor_cnn_input = [0] * self.max_length # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        right_pos_cnn_input = [0] * self.max_length

        nth_stack = 0   #stack

        idx_eoj = state.nth_stack(nth_stack) # stack에 있는 첫번째 어절, 처음은 -1로 초기화 되어 있음

        idx_input = self.max_length - 1 # 11

        while idx_input > 0:
            # 마지막 어절부터 가져옴, 스택 초기값이 -1이므로, [lex:조화, pos:NNG] , [lex:다, pos:EF], [lex:., pos:SF]
            eoj = data.eojeol_list[idx_eoj] # data.eojeol_list[-1] : 마지막 어절
            mor_list = eoj.morpheme_list


            idx_mor0 = self.get_mor_feature_idx(mor_list[0].lex + '/' + mor_list[0].pos, mode) # 조화/NNG 의 인덱스 번호
            idx_mor1 = self.get_mor_feature_idx(self.empty_dummy.lex + '/' + self.empty_dummy.pos, mode)
            idx_mor2 = self.get_mor_feature_idx(self.empty_dummy.lex + '/' + self.empty_dummy.pos, mode)
            idx_mor3 = self.get_mor_feature_idx(mor_list[-1].lex + '/' + mor_list[-1].pos, mode) # ./SF 의 인덱스 번호

            idx_pos0 = self.get_pos_featrue_idx(mor_list[0].pos, mode)
            idx_pos1 = self.get_pos_featrue_idx(self.empty_dummy.pos, mode)
            idx_pos2 = self.get_pos_featrue_idx(self.empty_dummy.pos, mode)
            idx_pos3 = self.get_pos_featrue_idx(mor_list[-1].pos, mode)

            if len(mor_list) >= 2:
                idx_mor1 = self.get_mor_feature_idx(mor_list[1].lex + '/' + mor_list[1].pos, mode)
                idx_mor2 = self.get_mor_feature_idx(mor_list[-2].lex + '/' + mor_list[-2].pos, mode)
                idx_pos1 = self.get_pos_featrue_idx(mor_list[1].pos, mode)
                idx_pos2 = self.get_pos_featrue_idx(mor_list[-2].pos, mode)

            # 끝에서부터 집어넣기
            right_mor_cnn_input[idx_input] = idx_mor3
            right_pos_cnn_input[idx_input] = idx_pos3
            idx_input -= 1

            right_mor_cnn_input[idx_input] = idx_mor2
            right_pos_cnn_input[idx_input] = idx_pos2
            idx_input -= 1

            right_mor_cnn_input[idx_input] = idx_mor1
            right_pos_cnn_input[idx_input] = idx_pos1
            idx_input -= 1

            right_mor_cnn_input[idx_input] = idx_mor0
            right_pos_cnn_input[idx_input] = idx_pos0
            idx_input -= 1
            
            nth_stack += 1
            idx_eoj = state.nth_stack(nth_stack)

            if idx_eoj is None:
                break

        # top n'child feature

        child_mor = [0] * self.max_length
        child_pos = [0] * self.max_length
        idx_input = self.max_length - 1 # 11

        for i in range(self.window_size): # window size = 3, stack에서 상위 3개 어절의 child만 볼 것임
            # head의 가장 가까운 child
            child = state.get_child_of_stack(i) # i = 0, 1, 2, child = 가장 인덱스 순서번호 차이가 작은 child return
            if child is not None:

                eoj = data.eojeol_list[child.index]
                mor_list = eoj.morpheme_list

                idx_mor0 = self.get_mor_feature_idx(mor_list[0].lex + '/' + mor_list[0].pos, mode)
                idx_mor1 = self.get_mor_feature_idx(self.empty_dummy.lex + '/' + self.empty_dummy.pos, mode)
                idx_mor2 = self.get_mor_feature_idx(self.empty_dummy.lex + '/' + self.empty_dummy.pos, mode)
                idx_mor3 = self.get_mor_feature_idx(mor_list[-1].lex + '/' + mor_list[-1].pos, mode)

                idx_pos0 = self.get_pos_featrue_idx(mor_list[0].pos, mode)
                idx_pos1 = self.get_pos_featrue_idx(self.empty_dummy.pos, mode)
                idx_pos2 = self.get_pos_featrue_idx(self.empty_dummy.pos, mode)
                idx_pos3 = self.get_pos_featrue_idx(mor_list[-1].pos, mode)

                if len(mor_list) >= 2:
                    idx_mor1 = self.get_mor_feature_idx(mor_list[1].lex + '/' + mor_list[1].pos, mode)
                    idx_mor2 = self.get_mor_feature_idx(mor_list[-2].lex + '/' + mor_list[-2].pos, mode)

                    idx_pos1 = self.get_pos_featrue_idx(mor_list[1].pos, mode)
                    idx_pos2 = self.get_pos_featrue_idx(mor_list[-2].pos, mode)


                child_mor[idx_input] = idx_mor3
                child_pos[idx_input] = idx_pos3
                idx_input -= 1

                child_mor[idx_input] = idx_mor2
                child_pos[idx_input] = idx_pos2
                idx_input -= 1

                child_mor[idx_input] = idx_mor1
                child_pos[idx_input] = idx_pos1
                idx_input -= 1

                child_mor[idx_input] = idx_mor0
                child_pos[idx_input] = idx_pos0
                idx_input -= 1

            else:

                child_mor[idx_input] = 0
                child_pos[idx_input] = 0
                idx_input -= 1

                child_mor[idx_input] = 0
                child_pos[idx_input] = 0
                idx_input -= 1

                child_mor[idx_input] = 0
                child_pos[idx_input] = 0
                idx_input -= 1

                child_mor[idx_input] = 0
                child_pos[idx_input] = 0
                idx_input -= 1

        # handcrafted feature

        hc = [] # hand craft feature의 index가 들어감, hc_feature_map의 index, hc list에 append
                # hc feature 의 종류 -> stack과 buffer top 어절간의 거리(dist) -> 1
                #                    -> stack top 3개 어절의 (child가 있을 경우) 지배 관계 타입 -> 2
                #                    -> stack top 3개 어절의 (child가 있을 경우) child의 개수 -> 3
        # 거리
        # self.hc_feature_map = {'dist:1' : 1, 'dist:2' : 2
        # ,'dist:3~4' : 3, 'dist:5~7': 4, 'dist:8~': 5}

        ts = state.top_stack() # stack top 어절의 순서번호(index) 값
        tq = state.top_queue() # buffer top 어절의 순서번호(index) 값

        dist = ts - tq # 두 어절이 떨어져 있는 거리

        if ts is -1:
            dist = 1


        if dist < 2:
            feature_value = "d:0"
        elif dist < 3:
            feature_value = "d:1"
        elif dist < 5:
            feature_value = "d:2"
        elif dist < 8:
            feature_value = "d:3"
        else:
            feature_value = "d:4"

        # get_hc_feature_idx 을 할 경우, hc_feature_map 에 있는 { 'd:0' : 번호 } 에서, 번호를 return
        hc.append(self.get_hc_feature_idx(feature_value, mode)) # hc에 받은 번호 append


        # stack top3의 의존소 레이블
        for i in range(3): # i = 0, 1, 2
            # top1
            child = state.get_child_of_stack(i) # 가장 가까운 거리에 있는 child 가져옴
            feature_value = 'f1_' + str(i) + '_' # 'f1_[0,1,2]_'

            if child is None:
                feature_value += 'no' # 'f1_[0,1,2]_no'
            else:
                # child로 가는 label 가져오기
                feature_value += child.type # 'f1_[0,1,2]_지배관계(타입)'

            hc.append(self.get_hc_feature_idx(feature_value, mode)) # hc에 받은 번호 append

        # stack top3의 child 갯수 0,1,2,~
        for i in range(3): # i = 0, 1, 2
            num_of_child = state.get_num_of_child_of_stack(i) # top3 어절의 child의 개수를 return
            feature_value = 'f2_' + str(i) + '_'
            if num_of_child > 3: # child 개수가 3개가 넘어갈 경우 3으로 fix
                num_of_child = 3

            feature_value += str(num_of_child) # 'f2_[0,1,2]_child개수'

            hc.append(self.get_hc_feature_idx(feature_value, mode)) # hc_feature_map에 저장되고, 그 index를 return, hc 에 받은 번호 append

        return left_mor_cnn_input, left_pos_cnn_input, right_mor_cnn_input, right_pos_cnn_input, \
                child_mor, child_pos, hc


    # hc feature_list에 있는 hc_feature에 대해 hc[hc_index] 값을 0에서 1로 바꿈
    @staticmethod
    def convert_to_zero_one(feature_list, size):
        hc = [0] * size
        for hc_feature in feature_list:
            hc[int(hc_feature)] = 1
        return hc


class InputFeature(object): # 들어가는 feature의 종류,
    def __init__(self):
        self.left_mor_cnn = [] # Buffer 어절
        self.left_pos_cnn = [] # Buffer POS
        self.right_mor_cnn = [] # Stack 어절
        self.right_pos_cnn = [] # Stack POS
        self.child_mor = [] # Child 어절
        self.child_pos = [] # Child POS
        self.hand_crafted = [] # Hand_craft Feature
        self.y = None  # Action+Type

    def set(self, left_mor_cnn_input, left_pos_cnn_input, right_mor_cnn_input, right_pos_cnn_input,
            child_mor, child_pos, hc,
            output_sample=None):
        self.left_mor_cnn = left_mor_cnn_input
        self.left_pos_cnn = left_pos_cnn_input
        self.right_mor_cnn = right_mor_cnn_input
        self.right_pos_cnn = right_pos_cnn_input
        self.child_mor = child_mor
        self.child_pos = child_pos
        self.hand_crafted = hc
        self.y = output_sample #


