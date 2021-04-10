
from flask import Flask, render_template, request,jsonify, send_file,make_response
import base64
import redis
from PIL import Image
import io
from base64 import encodebytes
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import models
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from sklearn.preprocessing import normalize
from LSH import LSHash
import numpy as np
import timeit
from datetime import datetime


app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html',name="Set your query")


def get_response_image(image_path):
   pil_img = Image.open(image_path, mode='r')  # reads the PIL image
   byte_arr = io.BytesIO()
   pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
   encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
   return encoded_img

def load_pickle(namefile):
  with open("features"+namefile+".pickle", 'rb') as features_file:
    features,labels,paths= pickle.load(features_file)
  return features,labels,paths

def create_lsh_index(features,labels,paths,lsh=None):
    for i,feature in enumerate(features):
      lsh.index(feature,str(int(labels[i]))+"/"+paths[i])
    print("index created")

def extract_features():
    datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
    dataset = datagen.flow_from_directory(
            "query/",
            target_size=(224,224),
            batch_size=128,
            shuffle=False)
    features = finetuned_feature_extractor.predict(dataset)
    features = normalize(features)
    return features


@app.route('/handle_data', methods=['POST'])
def handle_data():
    print(request.form)
    print(request.form['search'])
    isThisFile = request.files.get('file')
    print(isThisFile)

    isThisFile.save("query/" +"query/"+ isThisFile.filename)

    query_feature_schema = extract_features()
    start_time = datetime.now()
    # do your work here
    find_labels,_=lsh.query(query_feature_schema[0],int(request.form['k']),request.form['search'])
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    time = str(end_time - start_time)
    #print(find_labels)
    paths=[d[0][1].split("/")[2] for d in find_labels]
    classes=[int(d[0][1].split("/")[1]) for d in find_labels]
    labels={"1":"Sedan","2":"Suv","3":"Van","4":"Hatchbak","5":"Mpv","6":"Pickup","7":"Bus","8":"Truck","9":"Estate","10":"Distractor"}
    distances=[d[1] for d in find_labels]
    encoded_imges = []
    response = dict()
    response['time'] = time
    for i,path in enumerate(paths):
      try:
        encoded_imges.append(get_response_image("database/"+str(classes[i])+"/"+path))
      except:
        pass
    classes = [labels.get(str(e),"") for e in classes]
    os.remove("query/query/"+ isThisFile.filename)
    response['encoded_images'] = encoded_imges
    response['distances'] = distances
    response['classes'] = classes
    return jsonify(response)
   


if __name__ == '__main__':
    
    print("Loading index....")
    data_features,data_labels,paths=load_pickle("/vgg19_finetuned_features")
    #lsh = LSHash(10,512,2,{"redis": {"host": "127.0.0.1", "port": 6379}},'resources/matrice.npz')
    lsh = LSHash(12,512,4,None,None)
    create_lsh_index(data_features,data_labels,paths,lsh)

    opt_vgg19_ft = models.load_model('model/vgg19_finetuned_2_best.h5')
    finetuned_feature_extractor = Model(inputs=opt_vgg19_ft.input, outputs=opt_vgg19_ft.get_layer('gap').output)
    
    print("Model loaded!")
    app.run(debug=False)