#encoding:utf-8

from flask  import render_template, request
from stereo_vision.threeDRestructure import threeDRestructure
import time
import json
import datetime
import os
from stereo_vision.threeDRestructure.utils import get_epoch_pairs_points,epoch_3Dpoints



@threeDRestructure.route('/')
def index():

    return render_template('three_d_restructure/three_d_restructure.html')


@threeDRestructure.route('/updateList2/',methods=['POST','GET'])
def  updateList2():
    #这里需要加检索筒端类型文件，把检索结果返回到字典里
    result = dict()
    type_path = os.path.dirname(os.path.realpath(__file__)).replace("threeDRestructure","static/res_pictures/result")
    type_list = os.listdir(type_path)
    for i in range(1,len(type_list)+1):
        result[i] = type_list[i-1]
    return result


@threeDRestructure.route('/selectLog/',methods=['POST','GET'])
def  selectLog():
    type = request.args.get('mydata')


    print(type)
    return  str(1)


@threeDRestructure.route('/Constraint/',methods=['POST','GET'])
def Constraint():
    epoch_name = request.args.get('flag')
    get_epoch_pairs_points(epoch_name)
    result = epoch_3Dpoints(epoch_name)
    return result








