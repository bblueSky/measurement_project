#encoding:utf-8

from flask  import render_template, request
from stereo_vision.poseCalculate import poseCalculate
from stereo_vision.poseCalculate.utils import tubePoseCalculate
import time
import json
import datetime
import os
from xml.dom import minidom



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

    if flag=="2022-00-00-00:00:00":
        result = {
            "Axis": [0,0,1000],
            "Angle": 0,
            "A_center": "None",
            "A_angle": 0,
            "B_center": "None",
            "B_angle": 0,
            "lll": 800,
            "APO": 1,
            "APH": 1,
            "APW": 1,
            "BPO": 1,
            "BPH": 1,
            "BPW": 1,
            "Ascore": 0,
            "Bscore": 0
        }
    else:
        firmPath = os.path.dirname(os.path.realpath(__file__)).replace("poseCalculate","static/res_pictures/result/")
        filePath = firmPath+flag+'/points_info.xml'
        p_doc = minidom.parse(filePath)
        p_root = p_doc.documentElement
        tube_type = p_root.getElementsByTagName("type")[0].childNodes[0].data
        firmPath = os.path.dirname(os.path.realpath(__file__)).replace("poseCalculate","static/priori_data/")
        dataPath = firmPath+tube_type+".xml"
        result = tubePoseCalculate(filePath,dataPath)
        firmPath = os.path.dirname(os.path.realpath(__file__)).replace("poseCalculate", "static/res_pictures/pose_result/")
        posePath = firmPath+flag+".xml"
        ts = time.time()
        dtime = datetime.datetime.fromtimestamp(ts).strftime('%Y' + '-' + '%m' + '-' + '%d' + '-' + '%H' + ':' + '%M' + ':' + '%S')
        pose_doc = minidom.Document()
        pose_root = pose_doc.createElement('poseCaculate_data')
        pose_root.setAttribute('????????????', dtime)
        pose_doc.appendChild(pose_root)
        tube_t = pose_doc.createElement("type")
        tube_t.appendChild(pose_doc.createTextNode(tube_type))
        pose_root.appendChild(tube_t)
        A_end = pose_doc.createElement("A_end")
        B_end = pose_doc.createElement("B_end")
        pose_root.appendChild(A_end)
        pose_root.appendChild(B_end)
        A_center = pose_doc.createElement("center")
        X = pose_doc.createElement("X")
        X.appendChild(pose_doc.createTextNode(str(result["A_center"][0])))
        A_center.appendChild(X)
        Y = pose_doc.createElement("Y")
        Y.appendChild(pose_doc.createTextNode(str(result["A_center"][1])))
        A_center.appendChild(Y)
        Z = pose_doc.createElement("Z")
        Z.appendChild(pose_doc.createTextNode(str(result["A_center"][2])))
        A_center.appendChild(Z)

        A_angle_pt = pose_doc.createElement("angle_point")
        X = pose_doc.createElement("X")
        X.appendChild(pose_doc.createTextNode(str(result["A_angle"][0])))
        A_angle_pt.appendChild(X)
        Y = pose_doc.createElement("Y")
        Y.appendChild(pose_doc.createTextNode(str(result["A_angle"][1])))
        A_angle_pt.appendChild(Y)
        Z = pose_doc.createElement("Z")
        Z.appendChild(pose_doc.createTextNode(str(result["A_angle"][2])))
        A_angle_pt.appendChild(Z)

        B_center = pose_doc.createElement("center")
        X = pose_doc.createElement("X")
        X.appendChild(pose_doc.createTextNode(str(result["B_center"][0])))
        B_center.appendChild(X)
        Y = pose_doc.createElement("Y")
        Y.appendChild(pose_doc.createTextNode(str(result["B_center"][1])))
        B_center.appendChild(Y)
        Z = pose_doc.createElement("Z")
        Z.appendChild(pose_doc.createTextNode(str(result["B_center"][2])))
        B_center.appendChild(Z)

        B_angle_pt = pose_doc.createElement("angle_point")
        X = pose_doc.createElement("X")
        X.appendChild(pose_doc.createTextNode(str(result["B_angle"][0])))
        B_angle_pt.appendChild(X)
        Y = pose_doc.createElement("Y")
        Y.appendChild(pose_doc.createTextNode(str(result["B_angle"][1])))
        B_angle_pt.appendChild(Y)
        Z = pose_doc.createElement("Z")
        Z.appendChild(pose_doc.createTextNode(str(result["B_angle"][2])))
        B_angle_pt.appendChild(Z)
        A_end.appendChild(A_center)
        A_end.appendChild(A_angle_pt)
        B_end.appendChild(B_center)
        B_end.appendChild(B_angle_pt)

        Axis = pose_doc.createElement("Axis")
        X = pose_doc.createElement("X")
        X.appendChild(pose_doc.createTextNode(str(result["Axis"][0])))
        Axis.appendChild(X)
        Y = pose_doc.createElement("Y")
        Y.appendChild(pose_doc.createTextNode(str(result["Axis"][1])))
        Axis.appendChild(Y)
        Z = pose_doc.createElement("X")
        Z.appendChild(pose_doc.createTextNode(str(result["Axis"][2])))
        Axis.appendChild(Z)
        pose_root.appendChild(Axis)

        Angle = pose_doc.createElement("Angle")
        degree = pose_doc.createElement("degree")
        degree.appendChild(pose_doc.createTextNode(str(result["Angle"])))
        Angle.appendChild(degree)
        pose_root.appendChild(Angle)
        with open(posePath, 'w') as fp:
            pose_doc.writexml(fp)

    return result










