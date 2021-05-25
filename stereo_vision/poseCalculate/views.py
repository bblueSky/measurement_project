#encoding:utf-8

from flask  import render_template, request
from stereo_vision.poseCalculate import poseCalculate
import time
import json
import datetime
import os




@poseCalculate.route('/')
def index():

    return render_template('pose_calculate/pose_calculate.html')


@poseCalculate.route('/updateList/',methods=['POST','GET'])
def updateList():
    result = dict()
    type_path = os.path.dirname(os.path.realpath(__file__)).replace("poseCalculate", "static/res_pictures/result")
    type_list = os.listdir(type_path)
    for i in range(1, len(type_list) + 1):
        result[i] = type_list[i - 1]
    return result


@poseCalculate.route('/selectLog/', methods=['POST', 'GET'])
def selectLog():
    type = request.args.get('mydata')

    print(type)
    return str(1)


@poseCalculate.route('/Calculate/', methods=['POST', 'GET'])
def Calculate():
    flag = request.args.get('flag')
    firmPath = os.path.dirname(os.path.realpath(__file__)).replace("poseCalculate","static/res_pictures/result/")
    filePath = firmPath+flag+'/points_info.xml'
    print(filePath)
    return str(1)










