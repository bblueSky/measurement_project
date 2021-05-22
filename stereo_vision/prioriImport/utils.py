import numpy as np
import os
import time
import json
from xml.dom import minidom
"""
[['2021-5-19-22:42:12'],
 ['850.00'], 
 ['760.03'],
 ['(0,411.05)', '(205.53,355.98)', '(355.98,205.53)', '(411.05,0.00)', '(355.98,-205.53)', '(205.53,-355.98)', '(0.00,-411.05)', '(-205.53,-355.98)', '(-355.98,-205.53)', '(-411.05,0)', '(-355.98,205.53)', '(-205.53,355.98)'],
 ['(397.04,-106.39)', '(106.39,-397.04)', '(-106.39,-397.04)', '(-290.66,-290.66)', '(-106.39,397.04)'], 
 ['(106.39,397.04)'], 
 ['850.01'],
 ['760.10'], 
 ['(138.52,380.58)', '(260.33,310.25)', '(350.74,202.50)', '(398.85,70.33)', '(398.85,-70.33)', '(350.74,-202.50)', '(260.33,-310.25)', '(138.52,-380.58)', '(0,-405.00)', '(-138.52,-380.58)', '(-260.33,-310.25)', '(-350.74,-202.50)', '(-398.85,-70.33)', '(-398.85,70.33)', '(-350.74,202.50)', '(-260.33,310.25)', '(-138.52,380.58)', '(0,405.00)'], 
 [], 
 []]
"""


def prioriDataInput(res):  ##函数作用：1、把初始列表的数据写进xml 2、用json返回所有二维点的坐标数据
    datatime = res[0][0]
    firmPath = os.path.dirname(os.path.realpath(__file__)).replace("prioriImport","static/priori_data/")
    savePath = firmPath+datatime+".xml"
    p_doc = minidom.Document()
    root = p_doc.createElement('priori_data')
    p_doc.appendChild(root)
    importTime = p_doc.createElement("importTime")
    A_end = p_doc.createElement("A_end")
    B_end = p_doc.createElement("B_end")
    root.appendChild(importTime)
    root.appendChild(A_end)
    root.appendChild(B_end)

    result = dict()
    result["time"] = res[0][0]
    time = p_doc.createElement("time")
    time.appendChild(p_doc.createTextNode(res[0][0]))
    importTime.appendChild(time)
    result["AouterD"] = res[1][0]
    AouterD = p_doc.createElement("AouterD")
    diameter = p_doc.createElement("diameter")
    diameter.appendChild(p_doc.createTextNode(res[1][0]))
    AouterD.appendChild(diameter)
    A_end.appendChild(AouterD)
    result["AinnerD"] = res[2][0]
    AinnerD = p_doc.createElement("AinnerD")
    diameter = p_doc.createElement("diameter")
    diameter.appendChild(p_doc.createTextNode(res[2][0]))
    AinnerD.appendChild(diameter)
    A_end.appendChild(AinnerD)
    result["Amouting"] = list()
    Amouting = p_doc.createElement("Amouting")
    A_end.appendChild(Amouting)
    i = 0
    for s in res[3]:
        p = p_doc.createElement("p"+str(i))
        x,y = s[1:-1].split(',')
        axis_x = p_doc.createElement("x")
        axis_y = p_doc.createElement("y")
        axis_x.appendChild(p_doc.createTextNode(x))
        axis_y.appendChild(p_doc.createTextNode(y))
        p.appendChild(axis_x)
        p.appendChild(axis_y)
        Amouting.appendChild(p)
        i+=1
        result["Amouting"].append([x,y])
    result["Atapped"] = list()
    Atapped = p_doc.createElement("Atapped")
    A_end.appendChild(Atapped)
    i = 0
    for s in res[4]:
        p = p_doc.createElement("p"+str(i))
        x, y = s[1:-1].split(',')
        axis_x = p_doc.createElement("x")
        axis_y = p_doc.createElement("y")
        axis_x.appendChild(p_doc.createTextNode(x))
        axis_y.appendChild(p_doc.createTextNode(y))
        p.appendChild(axis_x)
        p.appendChild(axis_y)
        Atapped.appendChild(p)
        i+=1
        result["Atapped"].append([x, y])
    result["Aquadrant"] = list()
    Aquadrant = p_doc.createElement("Aquadrant")
    A_end.appendChild(Aquadrant)
    i=0
    for s in res[5]:
        p = p_doc.createElement("p"+str(i))
        x, y = s[1:-1].split(',')
        axis_x = p_doc.createElement("x")
        axis_y = p_doc.createElement("y")
        axis_x.appendChild(p_doc.createTextNode(x))
        axis_y.appendChild(p_doc.createTextNode(y))
        p.appendChild(axis_x)
        p.appendChild(axis_y)
        Aquadrant.appendChild(p)
        i+=1
        result["Aquadrant"].append([x, y])
    center = p_doc.createElement("center")
    A_end.appendChild(center)
    sequence = p_doc.createElement("sequence")
    A_end.appendChild(sequence)
    result["BouterD"] = res[6][0]
    BouterD = p_doc.createElement("BouterD")
    diameter = p_doc.createElement("diameter")
    diameter.appendChild(p_doc.createTextNode(res[6][0]))
    BouterD.appendChild(diameter)
    B_end.appendChild(BouterD)
    result["BinnerD"] = res[7][0]
    BinnerD = p_doc.createElement("BinnerD")
    diameter = p_doc.createElement("diameter")
    diameter.appendChild(p_doc.createTextNode(res[7][0]))
    BinnerD.appendChild(diameter)
    B_end.appendChild(BinnerD)
    result["Bmouting"] = list()
    Bmouting = p_doc.createElement("Bmouting")
    B_end.appendChild(Bmouting)
    i=0
    for s in res[8]:
        p = p_doc.createElement("p" + str(i))
        x, y = s[1:-1].split(',')
        axis_x = p_doc.createElement("x")
        axis_y = p_doc.createElement("y")
        axis_x.appendChild(p_doc.createTextNode(x))
        axis_y.appendChild(p_doc.createTextNode(y))
        p.appendChild(axis_x)
        p.appendChild(axis_y)
        Bmouting.appendChild(p)
        i+=1
        result["Bmouting"].append([x, y])
    result["Btapped"] = list()
    Btapped = p_doc.createElement("Btapped")
    B_end.appendChild(Btapped)
    i = 0
    for s in res[9]:
        p = p_doc.createElement("p" + str(i))
        x, y = s[1:-1].split(',')
        axis_x = p_doc.createElement("x")
        axis_y = p_doc.createElement("y")
        axis_x.appendChild(p_doc.createTextNode(x))
        axis_y.appendChild(p_doc.createTextNode(y))
        p.appendChild(axis_x)
        p.appendChild(axis_y)
        Btapped.appendChild(p)
        i += 1
        result["Btapped"].append([x, y])
    result["Bquadrant"] = list()
    Bquadrant = p_doc.createElement("Bquadrant")
    B_end.appendChild(Bquadrant)
    i = 0
    for s in res[10]:
        p = p_doc.createElement("p" + str(i))
        x, y = s[1:-1].split(',')
        axis_x = p_doc.createElement("x")
        axis_y = p_doc.createElement("y")
        axis_x.appendChild(p_doc.createTextNode(x))
        axis_y.appendChild(p_doc.createTextNode(y))
        p.appendChild(axis_x)
        p.appendChild(axis_y)
        Bquadrant.appendChild(p)
        i += 1
        result["Bquadrant"].append([x, y])
    center = p_doc.createElement("center")
    B_end.appendChild(center)
    sequence = p_doc.createElement("sequence")
    B_end.appendChild(sequence)
    fp = open(savePath, 'w')
    p_doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    return result


def dataFitCircle():  ##函数作用：1、读取xml里的数据 2、根据这些数据拟合圆，计算各种孔位序列 3、用json返回圆心坐标 4、把圆心坐标、孔位序列写入xml
    ## 法兰有四种情况： 1、安装孔、螺纹孔、象限孔 2、安装孔、螺纹孔 3、安装孔、象限孔 4、安装孔
    ## 对应解决办法： 一、靶标装在象限孔上：1、3 二、靶标装在安装孔上：2、4,因此该函数应该分两种情况讨论



    return 0