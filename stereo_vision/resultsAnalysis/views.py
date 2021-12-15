#encoding:utf-8

from flask  import render_template, request
from stereo_vision.resultsAnalysis import  resultsAnalysis
import time
import json
import cv2
import numpy as np
import time
import datetime
import signal
import threading
import os
import stereo_vision.ksj.cam as  kcam

from  stereo_vision.cameraCalibration.utils  import  sig_calibration,stereo_Calibration



@resultsAnalysis.route('/')
def index():

    return render_template('results_analysis/results_analysis.html')

@resultsAnalysis.route('/updateList/',methods=['POST','GET'])
def updateList():
    result = dict()
    type_path = os.path.dirname(os.path.realpath(__file__)).replace("resultsAnalysis", "static/res_pictures/result")
    type_list = os.listdir(type_path)
    for i in range(1, len(type_list) + 1):
        result[i] = type_list[i - 1]
    return result


@resultsAnalysis.route('/selectLog/', methods=['POST', 'GET'])
def selectLog():
    type = request.args.get('mydata')

    print(type)
    return str(1)


@resultsAnalysis.route('/confirmLog/', methods=['POST', 'GET'])
def confirmLog():
    flag = request.args.get('flag')
    print(flag)
    ##拿到flag后开始根据这个索引收集数据




    return {"dd":flag}






