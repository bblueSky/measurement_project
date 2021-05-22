from stereo_vision.cameraCalibration.views import  cameraCalibration
from stereo_vision.testMeasure.views import testMeasure
from stereo_vision.prioriImport.views import prioriImport
from stereo_vision.poseCalculate.views import poseCalculate
from stereo_vision.resultsAnalysis.views import resultsAnalysis
from stereo_vision.planoMiller.views import planoMiller
from stereo_vision.robotMachine.views import robotMachine
from stereo_vision.threeDRestructure.views import threeDRestructure

from  flask import Flask,render_template,request,Blueprint , jsonify


def create_app():
    app = Flask(__name__)
    app.register_blueprint(cameraCalibration,url_prefix='/cameraCalibration')
    app.register_blueprint(testMeasure,url_prefix='/testMeasure')
    app.register_blueprint(prioriImport, url_prefix='/prioriImport')
    app.register_blueprint(poseCalculate, url_prefix='/poseCalculate')
    app.register_blueprint(resultsAnalysis, url_prefix='/resultsAnalysis')
    app.register_blueprint(planoMiller,url_prefix='/planoMiller')
    app.register_blueprint(robotMachine,url_prefix='/robotMachine')
    app.register_blueprint(threeDRestructure,url_prefix='/threeDRestructure')

    return app