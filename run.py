from flask import Flask
from flask  import render_template
from stereo_vision import create_app
from optparse import OptionParser


optparser = OptionParser()
optparser.add_option('-p','--port',dest='port',help='Server Http Port Number',default=5000,type='int')
(options,args) = optparser.parse_args()

app = create_app()
app.secret_key='stereo_vision'


if __name__ == '__main__':

    app.debug =True
    app.run(host='0.0.0.0',port=options.port)
