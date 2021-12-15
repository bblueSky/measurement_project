#encoding:utf-8

from flask  import render_template, request
from stereo_vision.resultsAnalysis import  resultsAnalysis
import json
import numpy as np
import time
import datetime
import os
from xml.dom import minidom

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
    # print(flag)
    ##拿到flag后开始根据这个索引收集数据
    global_path = os.path.dirname(os.path.realpath(__file__)).replace("resultsAnalysis", "static/global_info.xml")
    point_path = os.path.dirname(os.path.realpath(__file__)).replace("resultsAnalysis", "static/res_pictures/result/"+flag+"/points_info.xml")
    pose_path = os.path.dirname(os.path.realpath(__file__)).replace("resultsAnalysis", "static/res_pictures/pose_result/"+flag+".xml")

    global_doc = minidom.parse(global_path)
    global_root = global_doc.documentElement

    point_doc = minidom.parse(point_path)
    point_root = point_doc.documentElement

    pose_doc = minidom.parse(pose_path)
    pose_root = pose_doc.documentElement

    ##需要拿的数据：1、全局坐标系12个点 2、两端三维点 3、位姿四件套
    APOX = global_root.getElementsByTagName("APO")[0].childNodes[0].childNodes[0].data
    APOY = global_root.getElementsByTagName("APO")[0].childNodes[1].childNodes[0].data
    APOZ = global_root.getElementsByTagName("APO")[0].childNodes[2].childNodes[0].data

    APHX = global_root.getElementsByTagName("APH")[0].childNodes[0].childNodes[0].data
    APHY = global_root.getElementsByTagName("APH")[0].childNodes[1].childNodes[0].data
    APHZ = global_root.getElementsByTagName("APH")[0].childNodes[2].childNodes[0].data

    APWX = global_root.getElementsByTagName("APW")[0].childNodes[0].childNodes[0].data
    APWY = global_root.getElementsByTagName("APW")[0].childNodes[1].childNodes[0].data
    APWZ = global_root.getElementsByTagName("APW")[0].childNodes[2].childNodes[0].data

    BPOX = global_root.getElementsByTagName("BPO")[0].childNodes[0].childNodes[0].data
    BPOY = global_root.getElementsByTagName("BPO")[0].childNodes[1].childNodes[0].data
    BPOZ = global_root.getElementsByTagName("BPO")[0].childNodes[2].childNodes[0].data

    BPHX = global_root.getElementsByTagName("BPH")[0].childNodes[0].childNodes[0].data
    BPHY = global_root.getElementsByTagName("BPH")[0].childNodes[1].childNodes[0].data
    BPHZ = global_root.getElementsByTagName("BPH")[0].childNodes[2].childNodes[0].data

    BPWX = global_root.getElementsByTagName("BPW")[0].childNodes[0].childNodes[0].data
    BPWY = global_root.getElementsByTagName("BPW")[0].childNodes[1].childNodes[0].data
    BPWZ = global_root.getElementsByTagName("BPW")[0].childNodes[2].childNodes[0].data





    APOXs = global_root.getElementsByTagName("APOs")[0].childNodes[0].childNodes[0].data
    APOYs = global_root.getElementsByTagName("APOs")[0].childNodes[1].childNodes[0].data
    APOZs = global_root.getElementsByTagName("APOs")[0].childNodes[2].childNodes[0].data

    APHXs = global_root.getElementsByTagName("APHs")[0].childNodes[0].childNodes[0].data
    APHYs = global_root.getElementsByTagName("APHs")[0].childNodes[1].childNodes[0].data
    APHZs = global_root.getElementsByTagName("APHs")[0].childNodes[2].childNodes[0].data

    APWXs = global_root.getElementsByTagName("APWs")[0].childNodes[0].childNodes[0].data
    APWYs = global_root.getElementsByTagName("APWs")[0].childNodes[1].childNodes[0].data
    APWZs = global_root.getElementsByTagName("APWs")[0].childNodes[2].childNodes[0].data

    BPOXs = global_root.getElementsByTagName("BPOs")[0].childNodes[0].childNodes[0].data
    BPOYs = global_root.getElementsByTagName("BPOs")[0].childNodes[1].childNodes[0].data
    BPOZs = global_root.getElementsByTagName("BPOs")[0].childNodes[2].childNodes[0].data

    BPHXs = global_root.getElementsByTagName("BPHs")[0].childNodes[0].childNodes[0].data
    BPHYs = global_root.getElementsByTagName("BPHs")[0].childNodes[1].childNodes[0].data
    BPHZs = global_root.getElementsByTagName("BPHs")[0].childNodes[2].childNodes[0].data

    BPWXs = global_root.getElementsByTagName("BPWs")[0].childNodes[0].childNodes[0].data
    BPWYs = global_root.getElementsByTagName("BPWs")[0].childNodes[1].childNodes[0].data
    BPWZs = global_root.getElementsByTagName("BPWs")[0].childNodes[2].childNodes[0].data

    res = {
        "APO": [APOX, APOY, APOZ],
        "APH": [APHX, APHY, APHZ],
        "APW": [APWX, APWY, APWZ],
        "BPO": [BPOX, BPOY, BPOZ],
        "BPH": [BPHX, BPHY, BPHZ],
        "BPW": [BPWX, BPWY, BPWZ],
        "APOs": [APOXs, APOYs, APOZs],
        "APHs": [APHXs, APHYs, APHZs],
        "APWs": [APWXs, APWYs, APWZs],
        "BPOs": [BPOXs, BPOYs, BPOZs],
        "BPHs": [BPHXs, BPHYs, BPHZs],
        "BPWs": [BPWXs, BPWYs, BPWZs]
    }
    return res






