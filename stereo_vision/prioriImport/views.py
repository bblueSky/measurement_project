#encoding:utf-8

from flask  import render_template, request
from stereo_vision.prioriImport import  prioriImport
import time
import json
import datetime
import os




@prioriImport.route('/')
def index():
    return render_template('priori_import/priori_import.html')



@prioriImport.route('/prioriFile/',methods=['POST','GET'])
def prioriFile():
    filePath = os.path.dirname(os.path.realpath(__file__)).replace("prioriImport","exteriorFile/PF.txt")
    # with open(filePath, "r") as f:

    return str(1)






