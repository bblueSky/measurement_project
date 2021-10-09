# 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
from flask import Flask
from flask  import render_template
from stereo_vision import create_app
from optparse import OptionParser
# from stereo_vision.robotMachine import buildServer as bs
import time

optparser = OptionParser()
optparser.add_option('-p','--port',dest='port',help='Server Http Port Number',default=5000,type='int')
(options,args) = optparser.parse_args()

app = create_app()
app.secret_key='stereo_vision'


if __name__ == '__main__':

    app.debug =True
    # bs.__init__("0.0.0.0", 8080)
    app.run(host='0.0.0.0',port=options.port)
    # bs.stopServer()