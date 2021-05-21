#encoding:utf-8

from flask  import render_template, request
from stereo_vision.prioriImport import  prioriImport
import time
import json
import datetime
import os
from stereo_vision.prioriImport.utils import prioriDataInput,dataFitCircle



@prioriImport.route('/')
def index():
    return render_template('priori_import/priori_import.html')



@prioriImport.route('/prioriFile/',methods=['POST','GET'])
def prioriFile():
    filePath = os.path.dirname(os.path.realpath(__file__)).replace("prioriImport","exteriorFile/PF.txt") ##先用固定路径，以后再考虑文件传输的事
    formPath = os.path.dirname(os.path.realpath(__file__)).replace("prioriImport","static/priori_data/")
    cur = list() ##每种数据的列表
    res = list() ##总列表
    with open(filePath, "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            if line=="数据时间":
                continue
            if line=="A端法兰外圈直径" or line=="A端法兰内圈直径" or line=="A端安装孔" or line=="A端螺纹孔" or line=="A端象限孔" or line=="B端法兰外圈直径" or line=="B端法兰内圈直径" or line=="B端安装孔" or line=="B端螺纹孔" or line=="B端象限孔":
                res.append(cur)
                cur = list()
                continue
            cur.append(line)
        res.append(cur)
    # print(res)
    result = prioriDataInput(res)  ##函数作用：1、把初始列表的数据写进xml 2、用json返回所有二维点的坐标数据
    print(result)
    return result


@prioriImport.route('/fitCircle/',methods=['POST','GET'])
def fitCircle():
    res = dataFitCircle()  ##函数作用：1、读取xml里的数据 2、根据这些数据拟合圆 3、用json返回圆心坐标 4、把圆心坐标写入xml
    return res

##还差一个函数
