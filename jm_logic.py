import copy
import csv
import itertools
from collections import Counter, deque

import numpy as np
import tensorflow as tf
import math

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from model import PointHistoryClassifier

from hangul_utils import split_syllables, join_jamos


#------------------- scale normalization model ------------------- #

## m1 방향에 따라 분류(손바닥 위)
actions_m1 = ['ㅁ','ㅂ','ㅍ','ㅇ','ㅇ','ㅎ','ㅏ','ㅐ','ㅑ','ㅒ','ㅣ']
interpreter_m1 = tf.lite.Interpreter(model_path="models/JM/scale_norm/model1.tflite")
interpreter_m1.allocate_tensors()

## m2 방향에 따라 분류(손등 위)
actions_m2 = ['ㅇ','ㅎ','ㅗ','ㅚ','ㅛ']
interpreter_m2 = tf.lite.Interpreter(model_path="models/JM/scale_norm/model2.tflite")
interpreter_m2.allocate_tensors()

## m3 방향에 따라 분류(아래)
actions_m3 = ['ㄱ','ㅈ','ㅊ','ㅋ','ㅅ','ㅜ','ㅟ']
interpreter_m3 = tf.lite.Interpreter(model_path="models/JM/scale_norm/model3.tflite")
interpreter_m3.allocate_tensors()

## m4 방향에 따라 분류 (앞)
actions_m4 = ['ㅎ','ㅓ','ㅔ','ㅕ','ㅖ']
interpreter_m4 = tf.lite.Interpreter(model_path="models/JM/scale_norm/model4.tflite")
interpreter_m4.allocate_tensors()

## m5 방향에 따라 분류 (옆)
actions_m5 = ['ㄴ','ㄷ','ㄹ','ㅡ','ㅢ']
interpreter_m5 = tf.lite.Interpreter(model_path="models/JM/scale_norm/model5.tflite")
interpreter_m5.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_m1.get_input_details()
output_details = interpreter_m1.get_output_details()

# -------------------------------------------------- #



# action Variable
cnt10 = 0
text_cnt = 0
dcnt = 0
min_detec = 10
max_detec = 30
num_lst = [11, 15, 16]
flag = False
choice = 0

# Korean Variable
seq_length = 10
seq = []
action_seq = []
last_action = None
this_action = ''
select_model = ''
wrist_angle = 0
confidence = 0.9
action = ''

# User Interface Variable
button_overlay = overlayList[0]

# Keyboard Variable
cnt = 0
jamo_li = deque()
jamo_join_li = deque()
jamo_join_li.append(' ')

status_cnt_conf = 10
status_lst = deque(['Stop']*status_cnt_conf, maxlen=status_cnt_conf)

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
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

point_history_classifier = PointHistoryClassifier()

# Read labels ###########################################################
with open(
        'model/point_history_classifier/point_history_classifier_label_other.csv',
        encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    point_history_classifier_labels = [
        row[0] for row in point_history_classifier_labels
    ]
    
