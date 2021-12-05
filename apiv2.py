# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from model import ScreeningClinicHandsClassifier
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from jm_logic import model_access_product

StandardScaler = StandardScaler()
MinMaxScaler = MinMaxScaler()
MaxAbsScaler = MaxAbsScaler()
RobustScaler = RobustScaler()

app = Flask(__name__)
CORS(app, resources={r'*': {'origins': '*'}})

def predict_hands(history):
    label_csv = pd.read_csv('./label_SC_full.txt')
    label_array = np.array(label_csv)
    point_history = np.array(history)

    exist_left = []
    exist_right = []
    for a in range(60):
        if point_history[a][0] != 0:
            exist_left.append(a)
    for b in range(60):
        if point_history[b][42] != 0:
            exist_right.append(b)

    for c in exist_left:
        if c < 4:
            front = 0
            back = 8
        else:
            front = c - 4
            back = c + 4
        if c > 56:
            front = 52
            back = 60
        else:
            front = c - 4
            back = c + 4
        exist_left_count = 0
        for d in range(front, back):
            if point_history[d][0] != 0:
                exist_left_count += 1
        if exist_left_count < 3:
            for e in range(42):
                point_history[c][e] = 0
                

    for f in exist_right:
        if f < 4:
            front = 0
            back = 8
        else:
            front = f - 4
            back = f + 4
        if f > 56:
            front = 52
            back = 60
        else:
            front = f - 4
            back = f + 4
        exist_right_count = 0
        for g in range(front, back):
            if point_history[g][42] != 0:
                exist_right_count += 1
        if exist_right_count < 3:
            for h in range(42):
                point_history[f][42 + h] = 0




    for i in range(84):
        start = 0
        end = 60
        for j in range(60):
            if point_history[:,i][j] != 0:
                start = j
                break
        for k in range(60 - 1, -1, -1):
            if point_history[:,i][k] != 0:
                end = k
                break
        if end - start == 60:
            pass
        else:
            blank = []
            blankstart = 0
            blankend = 0
            for m in range(start,end+1):
                if point_history[m,i] == 0:
                    blankstart = m
                    if(m <= blankend):
                        pass
                    else:
                        for n in range(m,end+1):
                            if point_history[n,i] != 0:
                                blankend = n - 1
                                break
                        blank.append([blankstart,blankend])
            for fill in blank:
                dif = fill[1] - fill[0] + 1
                num_fill = (point_history[fill[1] + 1,i] - point_history[fill[0] - 1,i])/(dif + 1)
                for filling in range(fill[0],fill[1]+1):
                    point_history[filling,i] = point_history[fill[0]-1,i] + (filling-fill[0] + 1) * num_fill

    absorption = []
    for i in range(60):
        x_left_label = []
        y_left_label = []
        x_right_label = []
        y_right_label = []
        frame_label = []
        for j in range(21):
            x_left_label.append(point_history[i][2 * j] - point_history[i][0])
            y_left_label.append(point_history[i][2 * j + 1] - point_history[i][1])
            x_right_label.append(point_history[i][42 + 2 * j] - point_history[i][42])
            y_right_label.append(point_history[i][42 + 2 * j + 1] - point_history[i][43])

        if max(x_left_label) == min(x_left_label):
            x_left_scale = x_left_label
        else:
            x_left_scale = x_left_label/(max(x_left_label)-min(x_left_label))
        if max(y_left_label) == min(y_left_label):
            y_left_scale = y_left_label
        else:
            y_left_scale = y_left_label/(max(y_left_label)-min(y_left_label))
        if max(x_right_label) == min(x_right_label):
            x_right_scale = x_right_label
        else:
            x_right_scale = x_right_label/(max(x_right_label)-min(x_right_label))
        if max(y_right_label) == min(y_right_label):
            y_right_scale = y_right_label
        else:
            y_right_scale = y_right_label/(max(y_right_label)-min(y_right_label))
        frame_label.append(point_history[i][0])
        frame_label.append(point_history[i][1])
        for m in range(1,21):
            frame_label.append(x_left_scale[m])
            frame_label.append(y_left_scale[m])
        frame_label.append(point_history[i][42])
        frame_label.append(point_history[i][43])
        for n in range(1,21):
            frame_label.append(x_right_scale[n])
            frame_label.append(y_right_scale[n])
        absorption.append(frame_label)
    absorption = np.array(absorption)
    complete = np.concatenate((absorption), axis = None)
    Hands_classifier = ScreeningClinicHandsClassifier()
    reliability, label  = Hands_classifier(complete)
    label_name= label_array[label][1]

    return reliability, label_name



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

menu = []



@app.route('/api_hands', methods=['POST'])
def api_hands():
    point_history = np.array(request.json['data'])
    reliability, label_name = predict_hands(point_history)
    reliability = reliability * 100
    model_access_product(point_history, point_history)
    print(label_name)
    print(reliability)
    if reliability > 85:
        return {"label":label_name, "reliability":reliability}
    else:
        return {"label":"다시"}




if __name__ == '__main__':
    app.run(host='localhost', port=4000, threaded=True)