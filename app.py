from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import uuid
from FNNReq import FnnNueralNet, FnnDownloadFile
from CNNReq import cnn_configs, get_convolutional_neural_net, download_cnn_file

app = Flask(__name__)
CORS(app)


#functions for FNN Architecture

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/get-neuralnet', methods=['POST'])
def get_neuralnet():
   return FnnNueralNet()
    

@app.route('/download-neural-net/<config_id>', methods=['GET'])
def download_neural_net(config_id):
    return FnnDownloadFile(config_id)


# Functions for CNN Architecture
    
@app.route("/get-conv-neuralnet", methods=['POST'])
def get_conv_neuralnet():
    return get_convolutional_neural_net()

@app.route("/dowload-conv-nueralnet/<config_id>", methods=["GET"])
def dowload_cnn(config_id):
    return download_cnn_file(config_id)

    
    

if __name__ == '__main__':
    app.run(debug=True)