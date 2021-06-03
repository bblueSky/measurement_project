#encoding:utf-8

from flask  import render_template, request
from stereo_vision.robotMachine import robotMachine
import time
import json
import datetime
import os
from stereo_vision.robotMachine.utils import robotComunicate



@robotMachine.route('/')
def index():

    return render_template('robot_machine/robot_machine.html')



@robotMachine.route('/updateList/',methods=['POST','GET'])
def updateList():
    result = dict()
    type_path = os.path.dirname(os.path.realpath(__file__)).replace("robotMachine", "static/res_pictures/pose_result")
    type_list = os.listdir(type_path)
    for i in range(1, len(type_list) + 1):
        result[i] = type_list[i - 1][:-4]
    return result


@robotMachine.route('/selectLog/', methods=['POST', 'GET'])
def selectLog():
    type = request.args.get('mydata')

    print(type)
    return str(1)


@robotMachine.route('/robotCommunicate/', methods=['POST', 'GET'])
def robotCommunicate():
    type = request.args.get('flag')
    print(type)
    robotComunicate()
    return str(1)



