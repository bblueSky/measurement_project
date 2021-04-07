from stereo_vision.cameraCalibration.views import  cameraCalibration
from stereo_vision.testMeasure.views import testMeasure
#from stereo_vision.bracketClassification.views import bracketClassification
#from stereo_vision.featureExtraction.views import featureExtraction
from stereo_vision.resultsAnalysis.views import resultsAnalysis
#from stereo_vision.modelCompare.views import modelCompare

from  flask import Flask,render_template,request,Blueprint , jsonify


def create_app():
    app = Flask(__name__)
    app.register_blueprint(cameraCalibration,url_prefix='/cameraCalibration')
    app.register_blueprint(testMeasure,url_prefix='/testMeasure')
    #app.register_blueprint(bracketClassification, url_prefix='/bracketClassification')
    #app.register_blueprint(featureExtraction, url_prefix='/featureExtraction')
    app.register_blueprint(resultsAnalysis, url_prefix='/resultsAnalysis')
    #app.register_blueprint(modelCompare,url_prefix='/modelCompare')

    return app