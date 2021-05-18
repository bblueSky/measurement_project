#encoding:utf-8

from flask  import render_template, request
from stereo_vision.poseCalculate import  poseCalculate
import time
import json
import datetime
import os




@poseCalculate.route('/')
def index():

    return render_template('pose_calculate/pose_calculate.html')










