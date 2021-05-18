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










