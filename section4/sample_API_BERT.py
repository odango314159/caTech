# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:00:47 2023

@author: odang
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import transformers
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import logging
import flask

app = flask.Flask(__name__)
model = None

logging.set_verbosity_error()
MAX_LEN = 256
def load_model():
    global net
    global tokenizer
    global device
    print("Loading...")
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model = BERTmodel()
    model.load_state_dict(torch.load('./BERTparam.pth'))
    net = model.to(device)
    print("Loaded!")


class MyDataset(Dataset):
  def __init__(self,data,tokenizer):
    self.data = data
    self.tokenizer = tokenizer
  def __len__(self):
    return len(self.data)
  def __getitem__(self,idx):
    text = self.data["text"].tolist()[idx]
    label = self.data["label"].tolist()[idx]
    label = torch.tensor(label,dtype=torch.int64)
    encodings = self.tokenizer(
        text,padding="max_length",max_length=MAX_LEN,truncation=True
    )

    input_ids = torch.tensor(
        encodings["input_ids"],dtype=torch.long
    )
    
    attention_mask = torch.tensor(
        encodings["attention_mask"],dtype=torch.long
    )

    token_type_ids = torch.tensor(
        encodings["token_type_ids"],dtype=torch.long
    )

    return input_ids,attention_mask,token_type_ids,label

class BERTmodel(nn.Module):
  def __init__(self):
    super(BERTmodel,self).__init__()
    self.bert = AutoModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.linear1 = nn.Linear(in_features=768,out_features=256)
    self.linear2 = nn.Linear(256,64)
    self.linear3 = nn.Linear(64,2)
    self.relu = nn.ReLU(inplace=True)
    self.softmax = nn.functional.softmax
  def forward(self,input_ids,attention_mask,token_type_ids):
    x = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask,
        token_type_ids = token_type_ids
    )
    x = x.pooler_output
    ###こっからヘッダー分類器
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.relu(x)
    x = self.linear3(x)
    output = self.softmax(x,dim=1)
    return output

def prediction(text):
    valid_data = pd.DataFrame()
    valid_data["text"] = np.array([text])
    valid_data["label"] = np.array([2]) ###これは仮に決めた値。0か1に更新される
    valid_data = MyDataset(valid_data,tokenizer)
    valid_loader = DataLoader(valid_data,batch_size=1)
    with torch.no_grad():
      for batch in valid_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        output = net(input_ids,attention_mask,token_type_ids)
        label = ["negative","positive"][int(torch.max(output,1)[1])]
        score = float(torch.max(output,1)[0])
        return label,score
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
        text = flask.request.form.get('text')
        if text is None:
            if flask.request.files['text']:
                file = flask.request.files['text']
                text = file.read().decode('shift-JIS')
    pred,score = prediction(text)
    response["predictions"]["classification_results"][0] = pred
    response["predictions"]["score"][0] = float(score)
    return flask.jsonify(response)
if __name__ == "__main__":
    load_model()
    print("starting server")
    app.run(debug=True)






