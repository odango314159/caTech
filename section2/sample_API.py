# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:56:59 2023

@author: odang
"""

import flask
import torch
from PIL import Image
import io
import json
import numpy as np
import torchvision.transforms as transforms



app = flask.Flask(__name__)
model = None



def load_model():
    global net
    global device
    global transformer_pred
    print("Loading...")
    transformer_pred = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5],
    )])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.jit.load('./ResNet.pt',map_location=torch.device('cpu'))
    net = model.to(device)
    print("Loaded!")

@app.route("/predict",methods=['POST'])

def predict():
    
    
    response = {
        "predictions":{
            "classification_results":[
                "label"
                ],
            "score":[
                0
                ]
            }
        }
    
    
    if flask.request.method == 'POST':
        if flask.request.files['image']:
            image_URL = flask.request.files['image']
            image = Image.open(image_URL)
            image = transformer_pred(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            
            output = net(image)
            f = torch.nn.Softmax(dim=1)
            
            output = f(output)
            
            label_list = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
            score,predicted = torch.max(output,1)
            
            response["predictions"]["classification_results"][0] = str(label_list[predicted])
            response["predictions"]["score"][0] = float(score)
    return flask.jsonify(response)
if __name__ == "__main__":
    load_model()
    print("starting server")
    app.run(debug=True)
            
            
        