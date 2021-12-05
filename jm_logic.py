import copy
import csv
import itertools
from collections import Counter, deque

import numpy as np
import tensorflow as tf
import math

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

from hangul_utils import split_syllables, join_jamos

## m1 방향에 따라 분류(손바닥 위)
actions_m1 = ['ㅁ','ㅂ','ㅍ','ㅇ','ㅇ','ㅎ','ㅏ','ㅐ','ㅑ','ㅒ','ㅣ']
interpreter_m1 = tf.lite.Interpreter(model_path="model/JM/scale_norm/model1.tflite")
interpreter_m1.allocate_tensors()

## m2 방향에 따라 분류(손등 위)
actions_m2 = ['ㅇ','ㅎ','ㅗ','ㅚ','ㅛ']
interpreter_m2 = tf.lite.Interpreter(model_path="model/JM/scale_norm/model2.tflite")
interpreter_m2.allocate_tensors()

## m3 방향에 따라 분류(아래)
actions_m3 = ['ㄱ','ㅈ','ㅊ','ㅋ','ㅅ','ㅜ','ㅟ']
interpreter_m3 = tf.lite.Interpreter(model_path="model/JM/scale_norm/model3.tflite")
interpreter_m3.allocate_tensors()

## m4 방향에 따라 분류 (앞)
actions_m4 = ['ㅎ','ㅓ','ㅔ','ㅕ','ㅖ']
interpreter_m4 = tf.lite.Interpreter(model_path="model/JM/scale_norm/model4.tflite")
interpreter_m4.allocate_tensors()

## m5 방향에 따라 분류 (옆)
actions_m5 = ['ㄴ','ㄷ','ㄹ','ㅡ','ㅢ']
interpreter_m5 = tf.lite.Interpreter(model_path="model/JM/scale_norm/model5.tflite")
interpreter_m5.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_m1.get_input_details()
output_details = interpreter_m1.get_output_details()

# -------------------------------------------------- #

# 한글 모델
seq_length = 10
seq = []
action_seq = []
last_action = None
this_action = ''
select_model = ''
wrist_angle = 0
confidence = 0.9
action = ''

status_cnt_conf = 10
status_lst = deque(['Stop']*status_cnt_conf, maxlen=status_cnt_conf)
cnt = 0
jamo_li = deque()
jamo_join_li = deque()
jamo_join_li.append(' ')

# 숫자 모델
text_cnt = 0


def model_access_product(hand_lmlist, detector_thumb, this_action, cnt, jamo_join_li, jamo_li):
    wrist_angle, similar_text_res = wrist_angle_calculator(hand_lmlist)
                    
    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
    input_data = np.array(input_data, dtype=np.float32)
    thumb_index_angle = int(detector_thumb)
    # detector_thumb에 값 새로 구해서 넣어줘야 됨 (새로 추가 된 값임)
    
    # 한글 조합
    cnt += 1
    
    # 한글 모음
    M = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅣ', 'ㅗ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅟ', 'ㅠ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅡ', 'ㅢ', 'ㅘ', 'ㅙ', 'ㅝ', 'ㅞ']
    J = ["ㄱ", "ㅅ", "ㅈ", "ㅊ", "ㅋ", "ㄴ", "ㄷ", "ㄹ", "ㅌ", "ㅁ", "ㅂ", "ㅍ", "ㅇ", "ㅎ", "ㄲ", "ㅆ", "ㅉ", "ㄸ", "ㅃ"]
    JJ_dict = {
        "ㄱ":"ㄲ",
        "ㅅ":"ㅆ",
        "ㅈ":"ㅉ",
        "ㄷ":"ㄸ",
        "ㅂ":"ㅃ"
        }
    siot = ['ㅅ', 'ㅆ']
    MM_lst = ['ㅗ', 'ㅜ']
    MM_dict = {
        "ㅏ":"ㅘ",
        "ㅐ":"ㅙ",
        "ㅓ":"ㅝ",
        "ㅔ":"ㅞ"
        }
    
    status = ''
    
    if this_action not in ['', ' ']:
        cnt += 1
        jamo_li.append(this_action)
        this_action = ''
        # print(cnt, jamo_li)
        
        status_lst.append(status)
        # print(cnt, status_lst)
        
        if cnt >= status_cnt_conf:
            jamo_dict = {}
            for jamo in jamo_li:
                jamo_dict[jamo] = jamo_li.count(jamo)
            jamo_dict = Counter(jamo_dict).most_common()
            # print("jamo_dict", jamo_dict)
            if jamo_dict and jamo_dict[0][1] >= int(status_cnt_conf*0.7):
                tmp = jamo_dict[0][0]
                status_lst_slice = list(deque(itertools.islice(status_lst, int(status_cnt_conf*0.5), status_cnt_conf-1)))
                print("status_lst_slice", status_lst_slice)
                print("tmp", tmp)
                if tmp in siot:
                    if len(jamo_join_li) == 1:
                        if 'Move' in status_lst_slice: 
                            jamo_join_li.append('ㅆ')
                        else:
                            jamo_join_li.append('ㅅ')
                    else:
                        if jamo_join_li[-1] in J:
                            jamo_join_li.append('ㅠ')
                        else:
                            if 'Move' in status_lst_slice: 
                                jamo_join_li.append('ㅆ')
                            else:
                                jamo_join_li.append('ㅅ')
                elif tmp in J:
                    if tmp in JJ_dict.keys():
                        # 쌍자음
                        if 'Move' in status_lst_slice: 
                                jamo_join_li.append(JJ_dict[tmp])
                        else:
                            jamo_join_li.append(tmp)
                    else:
                        jamo_join_li.append(tmp)
                elif tmp in M:
                    # 모음 - 모음 : 이중 모음
                    if jamo_join_li[-1] in MM_lst and tmp in MM_dict.keys():
                        jamo_join_li.pop()
                        jamo_join_li.append(MM_dict[tmp])
                    else:
                        jamo_join_li.append(tmp)
                elif tmp.isdigit():
                    if len(jamo_join_li) >= 3 and jamo_join_li[-2].isdigit() and jamo_join_li[-1].isdigit():
                        if int(jamo_join_li[-2] + jamo_join_li[-1]) % 10 == 0 and len(tmp) == 1:
                            tmp = str(int(jamo_join_li[-2] + jamo_join_li[-1]) + int(tmp))
                            jamo_join_li.pop()
                            jamo_join_li.pop()
                            for i in tmp:
                                jamo_join_li.append(i)
                        elif tmp in ["11", "15", "16"]:
                            if jamo_join_li[-2] == "1" and jamo_join_li[-1] == "0":
                                jamo_join_li.pop()
                                jamo_join_li.pop()
                                for i in tmp:
                                    jamo_join_li.append(i)
                        else:
                            for i in tmp:
                                jamo_join_li.append(i)
                    else:
                        for i in tmp:
                            jamo_join_li.append(i)
            jamo_li = []
            cnt = -int(status_cnt_conf)
            # print("jamo_join_li", jamo_join_li)
            # print("cnt", cnt)
    jamo_li = []
    cnt = -int(status_cnt_conf)
    
    if hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[5][2] < hand_lmlist[0][2] and hand_lmlist[17][2] < hand_lmlist[0][2]:
        i_pred, conf = model_predict(input_data, interpreter_m1)

        action = actions_m1[i_pred]

        if action == 'ㅁ':
            if hand_lmlist[8][2] < hand_lmlist[7][2]:
                action = 'ㅑ'
        elif action == 'ㅑ':
            if hand_lmlist[8][2] > hand_lmlist[7][2]:
                action = 'ㅁ'

    elif hand_lmlist[5][1] < hand_lmlist[17][1] and hand_lmlist[5][2] < hand_lmlist[0][2] and hand_lmlist[17][2] < hand_lmlist[0][2]:
        i_pred, conf = model_predict(input_data, interpreter_m2)

        action = actions_m2[i_pred]

    elif hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[0][2] < hand_lmlist[5][2] and hand_lmlist[0][2] < hand_lmlist[17][2]:
        i_pred, conf = model_predict(input_data, interpreter_m3)

        action = actions_m3[i_pred]
        if action == 'ㄱ':
            if thumb_index_angle > 250:
                action = 'ㅜ'
        elif action == 'ㅜ':
            if 35 < thumb_index_angle < 90:
                action = 'ㄱ'

    elif hand_lmlist[5][1] > hand_lmlist[0][1] and hand_lmlist[5][2] < hand_lmlist[17][2] and (wrist_angle <= 295 or wrist_angle >= 350):
        i_pred, conf = model_predict(input_data, interpreter_m4)

        action = actions_m4[i_pred]
        # pass
    
    elif hand_lmlist[5][1] > hand_lmlist[0][1] and hand_lmlist[5][2] < hand_lmlist[17][2]:
        i_pred, conf = model_predict(input_data, interpreter_m5)

        action = actions_m5[i_pred]
    
        if action == 'ㄹ':
            if similar_text_res < 0:
                action = 'ㅌ'
            elif 0 < similar_text_res < 20:
                action = 'ㄹ'
                
    s_lst = list(join_jamos(split_syllables(jamo_join_li)))
    for i in s_lst[:-1]:
        if i in J or i in M:
            s_lst.remove(i)
            jamo_join_li = s_lst
            
            return jamo_join_li


def number_model(hand_lmlist):
    cnt10 = 0
    min_detec = 10
    max_detec = 30
    dcnt = 0
    text_cnt = 0
    num_lst = [11, 15, 16]
    
    # x축을 기준으로 손가락 리스트
    right_hand_fingersUp_list_a0 = detector.fingersUp(axis=False)
    # y축을 기준으로 손가락 리스트
    right_hand_fingersUp_list_a1 = detector.fingersUp(axis=True)
    # 엄지 끝과 검지 끝의 거리 측정
    thumb_index_length = detector.findLength(4, 8)

    index_finger_angle_1 = int(detector.findHandAngle(img, 8, 9, 5, draw=False))
    index_finger_angle_2 = int(detector.findHandAngle(img, 8, 13, 5, draw=False))
    index_finger_angle_3 = int(detector.findHandAngle(img, 8, 17, 5, draw=False))
    index_finger_angle_4 = int(detector.findHandAngle(img, 4, 3, 0, draw=False))
    total_index_angle = index_finger_angle_1 + index_finger_angle_2 + index_finger_angle_3

    middle_finger_angle_1 = 360 - int(detector.findHandAngle(img, 12, 5, 9, draw=False))
    middle_finger_angle_2 = int(detector.findHandAngle(img, 12, 13, 9, draw=False))
    middle_finger_angle_3 = int(detector.findHandAngle(img, 12, 17, 9, draw=False))
    total_middle_angle = middle_finger_angle_1 + middle_finger_angle_2 + middle_finger_angle_3

    # 손바닥이 보임, 수향이 위쪽
    if hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[4][2] > hand_lmlist[8][2]:
        if right_hand_fingersUp_list_a0 == [0, 1, 0, 0, 0] and hand_lmlist[8][2] < hand_lmlist[7][2]:
            action = 1
        elif right_hand_fingersUp_list_a0 == [0, 1, 1, 0, 0]:
            action = 2
        elif right_hand_fingersUp_list_a0 == [0, 1, 1, 1, 0] or right_hand_fingersUp_list_a0 == [1, 1, 1, 1, 0]:
            action = 3
        elif right_hand_fingersUp_list_a0 == [0, 1, 1, 1, 1]:
            action = 4
        # elif right_hand_fingersUp_list_a0 == [1, 0, 1, 1, 1] and thumb_index_length < 30:
        #     action = 10  # 동그라미 10
        elif thumb_index_length < 30:
            if right_hand_fingersUp_list_a0 == [1, 0, 1, 1, 1]:
                action = 10
            elif right_hand_fingersUp_list_a0 == [1, 0, 0, 0, 0]:
                action = 0    
    # 손바닥이 보임
    if hand_lmlist[5][1] > hand_lmlist[17][1]:
        if right_hand_fingersUp_list_a0 == [1, 0, 0, 0, 0]:
            if right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 1]:
                action = 0
            else:
                action = 5
        # 손가락을 살짝 구부려 10과 20 구분
        if right_hand_fingersUp_list_a0[0] == 0 and right_hand_fingersUp_list_a0[2:] == [0, 0, 0] and total_index_angle < 140 and total_middle_angle > 300:
            action = 10
            cnt10 += 1
        elif right_hand_fingersUp_list_a0[0] == 0 and right_hand_fingersUp_list_a0[3:] == [0, 0] and total_index_angle < 140 and total_middle_angle < 150:
            action = 20

    # 손등이 보임, 수향이 몸 안쪽으로 향함, 엄지가 들려 있음
    if hand_lmlist[5][2] < hand_lmlist[17][2] and hand_lmlist[4][2] < hand_lmlist[8][2]:
        if right_hand_fingersUp_list_a1 == [1, 1, 0, 0, 0]:
            action = 6
        elif right_hand_fingersUp_list_a1 == [1, 1, 1, 0, 0]:
            action = 7
        elif right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 0]:
            action = 8
        elif right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 1]:
            action = 9

    # 손등이 보이고, 수향이 몸 안쪽으로 향함
    if hand_lmlist[5][2] < hand_lmlist[17][2] and hand_lmlist[1][2] < hand_lmlist[13][2]:
        # 엄지가 숨어짐
        if hand_lmlist[4][2] + 30 > hand_lmlist[8][2]:
            if right_hand_fingersUp_list_a1[2:] == [1, 0, 0] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                action = 12
            elif right_hand_fingersUp_list_a1[2:] == [1, 1, 0] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                action = 13
            elif right_hand_fingersUp_list_a1[2:] == [1, 1, 1] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                action = 14
        # 엄지가 보임
        else:
            if right_hand_fingersUp_list_a1[2:] == [1, 0, 0] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                action = 17
            elif right_hand_fingersUp_list_a1[2:] == [1, 1, 0] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                action = 18
            elif right_hand_fingersUp_list_a1[2:] == [1, 1, 1] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                action = 19    

    if cnt10 > (max_detec - min_detec):
        action = 10
        flag = True
        # print("clear")
        # dcnt = 0
        
        
    elif cnt10 > min_detec:
        if hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[4][2] > hand_lmlist[8][2]:
            if right_hand_fingersUp_list_a0 == [0, 1, 0, 0, 0] and hand_lmlist[8][2] < hand_lmlist[7][2]:
                dcnt += 1
                action = ''
                if max_detec > dcnt > min_detec:
                    action = 11
                elif dcnt > max_detec+10:
                    action = 0
                    cnt10 = 0
                    dcnt = 0
                    # print("clear")
        elif hand_lmlist[5][1] > hand_lmlist[17][1]:
            if right_hand_fingersUp_list_a0 == [1, 0, 0, 0, 0]:
                dcnt += 1
                action = ''
                if max_detec > dcnt > min_detec:
                    action = 15
                elif dcnt > max_detec+10:
                    action = ''
                    cnt10 = 0
                    dcnt = 0
        elif hand_lmlist[5][2] < hand_lmlist[17][2] and hand_lmlist[4][2] < hand_lmlist[8][2]:
            if right_hand_fingersUp_list_a1 == [1, 1, 0, 0, 0]:
                dcnt += 1
                action = ''
                if max_detec > dcnt > min_detec:
                    action = 16
                elif dcnt > max_detec+10:
                    action = ''
                    cnt10 = 0
                    dcnt = 0
                    
        if action in num_lst:
            flag = True

    if action != '':
        if flag:
            text_cnt += 1
            if text_cnt % max_detec == 0:
                cnt10 = 0
                text_cnt = 0
                dcnt = 0
                flag = False


def model_predict(input_data, interpreter_model):
    interpreter_model.set_tensor(input_details[0]['index'], input_data)
    interpreter_model.invoke()
    y_pred = interpreter_model.get_tensor(output_details[0]['index'])
    i_pred = int(np.argmax(y_pred[0]))
    conf = y_pred[0][i_pred]

    return i_pred, conf

def wrist_angle_calculator(hand_lmlist):
    radian = math.atan2(hand_lmlist[17][2]-hand_lmlist[0][2],hand_lmlist[17][1]-hand_lmlist[0][1])-math.atan2(hand_lmlist[5][2]-hand_lmlist[0][2],hand_lmlist[5][1]-hand_lmlist[0][1])
    wrist_angle = 360 - int(math.degrees(radian))
    radian_2 = math.atan2(hand_lmlist[9][2]-hand_lmlist[12][2],hand_lmlist[9][1]-hand_lmlist[12][1])
    wrist_angle_2 = int(math.degrees(radian_2))
    radian_3 = math.atan2(hand_lmlist[13][2]-hand_lmlist[16][2],hand_lmlist[13][1]-hand_lmlist[16][1])
    wrist_angle_3 = int(math.degrees(radian_3))

    if wrist_angle < 0:
        wrist_angle += 360
        
    if wrist_angle_2 < 0:
        wrist_angle_2 += 360
    if wrist_angle_3 < 0:
        wrist_angle_3 += 360
        
    similar_text_res = wrist_angle_3 - wrist_angle_2

    return wrist_angle, similar_text_res
