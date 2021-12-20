#encoding:utf-8

from flask  import render_template, request
from stereo_vision.resultsAnalysis import  resultsAnalysis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    APOX = round(float(global_root.getElementsByTagName("APO")[0].childNodes[0].childNodes[0].data),3)
    APOY = round(float(global_root.getElementsByTagName("APO")[0].childNodes[1].childNodes[0].data),3)
    APOZ = round(float(global_root.getElementsByTagName("APO")[0].childNodes[2].childNodes[0].data),3)

    APHX = round(float(global_root.getElementsByTagName("APH")[0].childNodes[0].childNodes[0].data),3)
    APHY = round(float(global_root.getElementsByTagName("APH")[0].childNodes[1].childNodes[0].data),3)
    APHZ = round(float(global_root.getElementsByTagName("APH")[0].childNodes[2].childNodes[0].data),3)

    APWX = round(float(global_root.getElementsByTagName("APW")[0].childNodes[0].childNodes[0].data),3)
    APWY = round(float(global_root.getElementsByTagName("APW")[0].childNodes[1].childNodes[0].data),3)
    APWZ = round(float(global_root.getElementsByTagName("APW")[0].childNodes[2].childNodes[0].data),3)

    BPOX = round(float(global_root.getElementsByTagName("BPO")[0].childNodes[0].childNodes[0].data),3)
    BPOY = round(float(global_root.getElementsByTagName("BPO")[0].childNodes[1].childNodes[0].data),3)
    BPOZ = round(float(global_root.getElementsByTagName("BPO")[0].childNodes[2].childNodes[0].data),3)

    BPHX = round(float(global_root.getElementsByTagName("BPH")[0].childNodes[0].childNodes[0].data),3)
    BPHY = round(float(global_root.getElementsByTagName("BPH")[0].childNodes[1].childNodes[0].data),3)
    BPHZ = round(float(global_root.getElementsByTagName("BPH")[0].childNodes[2].childNodes[0].data),3)

    BPWX = round(float(global_root.getElementsByTagName("BPW")[0].childNodes[0].childNodes[0].data),3)
    BPWY = round(float(global_root.getElementsByTagName("BPW")[0].childNodes[1].childNodes[0].data),3)
    BPWZ = round(float(global_root.getElementsByTagName("BPW")[0].childNodes[2].childNodes[0].data),3)


    APOXs = round(float(global_root.getElementsByTagName("APOs")[0].childNodes[0].childNodes[0].data),3)
    APOYs = round(float(global_root.getElementsByTagName("APOs")[0].childNodes[1].childNodes[0].data),3)
    APOZs = round(float(global_root.getElementsByTagName("APOs")[0].childNodes[2].childNodes[0].data),3)

    APHXs = round(float(global_root.getElementsByTagName("APHs")[0].childNodes[0].childNodes[0].data),3)
    APHYs = round(float(global_root.getElementsByTagName("APHs")[0].childNodes[1].childNodes[0].data),3)
    APHZs = round(float(global_root.getElementsByTagName("APHs")[0].childNodes[2].childNodes[0].data),3)

    APWXs = round(float(global_root.getElementsByTagName("APWs")[0].childNodes[0].childNodes[0].data),3)
    APWYs = round(float(global_root.getElementsByTagName("APWs")[0].childNodes[1].childNodes[0].data),3)
    APWZs = round(float(global_root.getElementsByTagName("APWs")[0].childNodes[2].childNodes[0].data),3)

    BPOXs = round(float(global_root.getElementsByTagName("BPOs")[0].childNodes[0].childNodes[0].data),3)
    BPOYs = round(float(global_root.getElementsByTagName("BPOs")[0].childNodes[1].childNodes[0].data),3)
    BPOZs = round(float(global_root.getElementsByTagName("BPOs")[0].childNodes[2].childNodes[0].data),3)

    BPHXs = round(float(global_root.getElementsByTagName("BPHs")[0].childNodes[0].childNodes[0].data),3)
    BPHYs = round(float(global_root.getElementsByTagName("BPHs")[0].childNodes[1].childNodes[0].data),3)
    BPHZs = round(float(global_root.getElementsByTagName("BPHs")[0].childNodes[2].childNodes[0].data),3)

    BPWXs = round(float(global_root.getElementsByTagName("BPWs")[0].childNodes[0].childNodes[0].data),3)
    BPWYs = round(float(global_root.getElementsByTagName("BPWs")[0].childNodes[1].childNodes[0].data),3)
    BPWZs = round(float(global_root.getElementsByTagName("BPWs")[0].childNodes[2].childNodes[0].data),3)



    ##拿到两端三维点的数据
    savepath = os.path.dirname(os.path.realpath(__file__)).replace("resultsAnalysis", "static/res_pictures/temp/")
    fig = plt.figure(dpi=400)
    font = {'serif': 'Times New Roman', 'weight': 'normal'}
    plt.rc('font', **font)
    ax = fig.add_subplot(111, projection='3d')
    A_end = list()
    B_end = list()
    numofAhole = len(point_root.getElementsByTagName("threeD")[0].childNodes)
    numofAtarget = len(point_root.getElementsByTagName("threeD")[1].childNodes)
    numofBhole = len(point_root.getElementsByTagName("threeD")[2].childNodes)
    numofBtarget = len(point_root.getElementsByTagName("threeD")[3].childNodes)
    for i in range(numofAhole):
        x__ = round(float(point_root.getElementsByTagName("X")[i].childNodes[0].data), 3)
        y__ = round(float(point_root.getElementsByTagName("Y")[i].childNodes[0].data), 3)
        z__ = round(float(point_root.getElementsByTagName("Z")[i].childNodes[0].data), 3)
        if i == 0:
            ax.scatter(x__, y__, z__, s=30, linewidth=0.5, label='hole', marker='o', color='blue', alpha=0.3)
        else:
            ax.scatter(x__, y__, z__, s=30, linewidth=0.5, marker='o', color='blue',alpha=0.3)
        A_end.append("特征点"+str(i+1)+"  ("+str(x__)+","+str(y__)+","+str(z__)+")")
    for i in range(numofAtarget):
        x__ = round(float(point_root.getElementsByTagName("X")[i+numofAhole].childNodes[0].data), 3)
        y__ = round(float(point_root.getElementsByTagName("Y")[i+numofAhole].childNodes[0].data), 3)
        z__ = round(float(point_root.getElementsByTagName("Z")[i+numofAhole].childNodes[0].data), 3)
        if i == 0:
            ax.scatter(x__, y__, z__, s=30, linewidth=0.5, label="target",marker='o', color='red', alpha=0.3)
        else:
            ax.scatter(x__, y__, z__, s=30, linewidth=0.5, marker='o', color='red', alpha=0.3)
        A_end.append("特征点"+str(i+1)+"  ("+str(x__)+","+str(y__)+","+str(z__)+")")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.legend()

    plt.savefig(savepath+"Aend.jpg")

    fig = plt.figure(dpi=400)
    font = {'serif': 'Times New Roman', 'weight': 'normal'}
    plt.rc('font', **font)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(numofBhole):
        x__ = round(float(point_root.getElementsByTagName("X")[i+numofAhole+numofAtarget].childNodes[0].data), 3)
        y__ = round(float(point_root.getElementsByTagName("Y")[i+numofAhole+numofAtarget].childNodes[0].data), 3)
        z__ = round(float(point_root.getElementsByTagName("Z")[i+numofAhole+numofAtarget].childNodes[0].data), 3)
        if i == 0:
            ax.scatter(x__, y__, z__, s=30, linewidth=0.5, label='hole', marker='o', color='blue', alpha=0.3)
        else:
            ax.scatter(x__, y__, z__, s=30, linewidth=0.5, marker='o', color='blue', alpha=0.3)
        B_end.append("特征点"+str(i+1)+"  ("+str(x__)+","+str(y__)+","+str(z__)+")")
    for i in range(numofBtarget):
        x__ = round(float(point_root.getElementsByTagName("X")[i+numofAhole+numofAtarget+numofBhole].childNodes[0].data), 3)
        y__ = round(float(point_root.getElementsByTagName("Y")[i+numofAhole+numofAtarget+numofBhole].childNodes[0].data), 3)
        z__ = round(float(point_root.getElementsByTagName("Z")[i+numofAhole+numofAtarget+numofBhole].childNodes[0].data), 3)
        if i == 0:
            ax.scatter(x__, y__, z__, s=30, linewidth=0.5, label="target", marker='o', color='red', alpha=0.3)
        else:
            ax.scatter(x__, y__, z__, s=30, linewidth=0.5, marker='o', color='red', alpha=0.3)
        B_end.append("特征点"+str(i+1)+"  ("+str(x__)+","+str(y__)+","+str(z__)+")")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.legend()

    plt.savefig(savepath + "Bend.jpg")


    ##拿到位姿四件套数据
    Axis1 = [
        round(float(pose_root.getElementsByTagName("Axis")[0].childNodes[0].childNodes[0].data), 3),
        round(float(pose_root.getElementsByTagName("Axis")[0].childNodes[1].childNodes[0].data), 3),
        round(float(pose_root.getElementsByTagName("Axis")[0].childNodes[2].childNodes[0].data), 3)
    ]
    Angle = round(float(pose_root.getElementsByTagName("degree")[0].childNodes[0].data), 3)

    A_center = [
        round(float(pose_root.getElementsByTagName("center")[0].childNodes[0].childNodes[0].data), 3),
        round(float(pose_root.getElementsByTagName("center")[0].childNodes[1].childNodes[0].data), 3),
        round(float(pose_root.getElementsByTagName("center")[0].childNodes[2].childNodes[0].data), 3)
    ]

    B_center = [
        round(float(pose_root.getElementsByTagName("center")[1].childNodes[0].childNodes[0].data), 3),
        round(float(pose_root.getElementsByTagName("center")[1].childNodes[1].childNodes[0].data), 3),
        round(float(pose_root.getElementsByTagName("center")[1].childNodes[2].childNodes[0].data), 3)
    ]

    lll = ((BPOX ** 2 + BPOY ** 2 + BPOZ ** 2) ** 0.5) * 0.5

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
        "BPWs": [BPWXs, BPWYs, BPWZs],
        "Aend":A_end,
        "Bend":B_end,
        "Axis": Axis1,
        "Angle": Angle,
        "A_center": A_center,
        "B_center": B_center,
        "lll":lll
    }
    return res






