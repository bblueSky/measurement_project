#encoding:utf-8

from flask  import render_template, request
from stereo_vision.planoMiller import  planoMiller
import time
import json
import datetime
import os




@planoMiller.route('/')
def index():

    return render_template('plano_miller/plano_miller.html')










