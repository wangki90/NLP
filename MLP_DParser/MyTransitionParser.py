# -*- coding: utf-8 -*-

#파서 학습

from DataPreProc import Dependency
from MyFeatureModel import InputFeature

class TransitionState(object): #state 상태에서 초기화
    def __init__(self):
        self.stack = [] # Stack
        self.queue = [] # Buffer
        self.predict = [] # Predict List
        self.transition_sequence = [] # Action Sequence

    def initialize_state(self, data): # data -> 1개 코퍼스
        self.stack = []
        self.queue = []
        self.predict = []
        self.transition_sequence = []

        # Buffer에
        for i in range(data.get_size() - 1, -1, -1): # range(5, 10) --> 해당 범위 5,6,7,8,9  , range(1, 10, 3) --> 1에서 10까지 3단위, 즉 1, 4, 7
            # data.get_size() -> return eoj_list size.
            # Buffer에 끝 어절 인덱스(10)부터 순차적으로 넣어준다
            self.push_queue(i)
        self.push_stack(-1) # -1을 스택에 푸쉬하는 이유? 초기값(root)

    def is_final_state(self):
        if len(self.queue) <= 0:
            return True
        return False

    def push_stack(self, idx): #stack push
        self.stack.append(idx)

    def pop_stack(self): #stack pop
        return self.stack.pop()

    def top_stack(self): #stack top value return
        return self.stack[-1] #stack의 맨 뒤값

    def nth_stack(self, n): #stack의 n번째 value return
        if len(self.stack) <= n:
            return None
        n = - n - 1 #stack top을 기준
        return self.stack[n]

    def push_queue(self, idx):  #queue push
        self.queue.append(idx)

    def pop_queue(self):    #queue pop
        return self.queue.pop(0)

    def top_queue(self): # queue 맨 밑 value return
        return self.queue[0]

    def nth_queue(self, n): # queue n번째 value return
        if len(self.queue) <= n:
            return None

        return self.queue[n]

    def left(self, tag ='none'): # tag = 지배관계(타입)
        i = self.top_stack()
        j = self.pop_queue()
        self.push_stack(j) # buffer에 있는 어절의 순서번호를, stack으로 push
        self.transition_sequence.append('left') # 정답 sequence에 left 저장

        # buffer에 있던 어절을 predict에 저장하려고 함
        dep = Dependency()
        dep.index = j # buffer에 있던 어절의 순서번호
        dep.head = i # buffer에 있던 어절의 지배소 번호
        dep.type = tag # buffer에 있던 어절의 지배관계(타입)
        self.predict.append(dep) # predict list에 저장

    def get_next_gold_transition(self, data, model):
        i = self.top_stack() # i = stack top value
        j = self.top_queue() # j = queue(buffer) top value

        gold_head = data.correct_dep_list[j] # buffer top에 있는 어절의 Dependecy(인덱스 번호, 지배소 번호, 지배소 관계(타입))
        gold_type = gold_head.type # buffer top에 있는 어절의 지배소 관계

        if i == gold_head.head: # buffer top에 있는 어절의 지배소 번호와 stack top에 있는 어절의 순서번호가 같을 경우, 즉 지배소-의존소 관계일 경우,
            self.left(gold_type) # left 함수 호출
            return model.get_type_idx('left_' + gold_type) # type_map에 저장되어있는 index를 return

        # 의존관계가 아닐 경우, reduce
        self.reduce()
        return model.get_type_idx('reduce')


    def reduce(self):
        self.pop_stack() # stack pop
        self.transition_sequence.append('reduce') # 정답 sequence에 reduce 저장

    def get_child_of_stack(self, n):
        if len(self.stack) < n:
            return None

        head_idx = self.nth_stack(n) # stack n번재 어절의 어절 순서번호

        min_len = 99999
        rtn_p = None
        for p in self.predict:
            if p.head == head_idx: # predict list에 있는 p의 지배소 번호와, 지배소의 인덱스 번호가 같으면, 즉 child 관계에 있을 경우
                if abs(head_idx - p.index) < min_len: # 그것들의 어절 순서번호끼리 빼준 값, 즉 가장 인덱스 번호 차이가 작은 child를 찾겠다는 것
                    min_len = abs(head_idx - p.index)
                    rtn_p = p

        return rtn_p

    def get_num_of_child_of_stack(self, n):
        if len(self.stack) < n:
            return -1

        head_idx = self.nth_stack(n)
        num = 0
        for p in self.predict:
            if p.head == head_idx:
                num += 1

        return num


    def get_result(self):
        # predict를 정리해서 출력
        predict = self.state.predict
        result = [0] * len(predict)
        for i in range(len(predict)):
            result[predict[i].index] = predict[i]


        return result

class TransitionParse(object):
    def __init__(self, model):
        self.state = TransitionState() # state 초기화
        self.f_model = model # f_model = FeatureModel()
    
    def get_result(self):
        # predict list에서는 맨 끝 어절부터 순서대로 저장되므로 이 방향을 반대로 바꿔줌
        # 만약 어절이 10개 였을 경우,
        # predict[9] -> result[0]
        # predict[8] -> result[1]
        # predict[7] -> result[2]
        # predict[6] -> result[3] 이런식으로
        predict = self.state.predict
        result = [0] * len(predict)
        for i in range(len(predict)):
            result[predict[i].index] = predict[i]

        return result

    def make_gold_corpus(self, data): # data - 1개 코퍼스
        total_sample = []
        self.state.initialize_state(data) # stack, queue, predict, transition sequence 생성 및 초기화
        # 여기서부터 transition parser state의 시작이라고 볼 수 있음
        # final_state 가 될때까지 코퍼스 생성
        while not self.state.is_final_state(): # Queue(Buffer).size < 0
            sample = InputFeature() # 들어가는 feature 변수 생성 및 초기화, left_mor,left_pos,right_mor,right_pos,child_mor,child_pos,hc,y

            # Feature 벡터를 만듬
            left_mor_cnn_input, left_pos_cnn_input, right_mor_cnn_input, right_pos_cnn_input, child_mor, child_pos, hc \
                = self.f_model.make_feature_vector(self.state, data, 'train') # '\' 다음줄이 이어지는 코드 , train 모드

            output_sample = self.state.get_next_gold_transition(data, self.f_model) # output_sample -> 현 state에서 action을 return (reduce or left_goldtype 의 인덱스)
            # 정리
            # left_mor_cnn,pos_input -> buffer top 어절의 형태소/pos 정보
            # right_mor,pos_cnn_input -> stack top 어절의 형태소/pos 정보
            # child_mor,pos -> stack top3 어절의 가장 번호가 가까운 child 어절에 대한 형태소/pos 정보
            # hc -> 거리, 의존소 레이블, 의존소 개수
            # output_sample -> 현재 state에서 reduce or left_관계타입 의 feature index
            sample.set(left_mor_cnn_input, left_pos_cnn_input, right_mor_cnn_input, right_pos_cnn_input, child_mor, child_pos, hc, output_sample)

            # 한 state에 대한 sample을 append
            total_sample.append(sample)
        

        #정답 검사
        result = self.get_result()
        golds = data.correct_dep_list
        idx = 0

        while idx < len(golds):
            head = result[idx].head
            gold = golds[idx].head
            tag = result[idx].type
            gold_tag = golds[idx].type

            # correct_dep_list 와 predict_dep_list가 같을 경우 문제 없이 잘 된 것임
            if head != gold or tag != gold_tag:
                print('error')
            idx += 1

        return total_sample


    def make_input_vector(self, data):
        return self.f_model.make_feature_vector(self.state, data)

    def initialize(self, data):
        self.state.initialize_state(data)

    def is_final_state(self):
        return self.state.is_final_state()

    def run_action(self, next_action, model):
        next_action = model.get_str_type(next_action) # next_action은 index값이었음 -> 이 index의 타입을 가져옴

        if next_action != 'reduce':
            action = 'left'
            tag = next_action[next_action.find('_') + 1:] # 타입을 tag에 저장
        else:
            action = 'reduce'

        if action == 'left':
            self.state.left(tag)
        elif len(self.state.stack) <= 2: #강제 left 스택 size가 2이하일 경우
            self.state.left('VP')
        else: # action 이 reduce일 경우
            self.state.reduce()







