from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
from numpy import loadtxt
from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import json
model = tf.keras.models.load_model("models/mobilenetv1_model.h5",custom_objects={'KerasLayer':hub.KerasLayer})
model.build((None, 224, 224, 3))
model1 = tf.keras.models.load_model("models/resnet_model.h5",custom_objects={'KerasLayer':hub.KerasLayer})
model1.build((None, 224, 224, 3))
model2 = tf.keras.models.load_model("models/cnnv1_model.h5",custom_objects={'KerasLayer':hub.KerasLayer})
model2.build((None, 224, 224, 3))
model3 = tf.keras.models.load_model("models/efficientv2.h5",custom_objects={'KerasLayer':hub.KerasLayer})
model3.build((None, 224, 224, 3))
model4 = tf.keras.models.load_model("models/inception_model.h5",custom_objects={'KerasLayer':hub.KerasLayer})
model4.build((None, 224, 224, 3))
# summarize model.
#model.summary()

#print(model.get_config())

#model.summary()









    
COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')



@app.route('/home', methods=['POST'])
def home():
    global COUNT
    
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img = cv2.imread('static/{}.jpg'.format(COUNT))
    
    
    prediction_mobilenet = mobilenet_predict(img)
    classname_mobilenet = list(prediction_mobilenet.keys())[0]
    confidence_mobilenet = list(prediction_mobilenet.values())[0]

    prediction_resnet = resnet_predict(img)
    classname_resnet = list(prediction_resnet.keys())[0]
    confidence_resenet = list(prediction_resnet.values())[0]

    prediction_cnn = cnn_predict(img)
    classname_cnn = list(prediction_cnn.keys())[0]
    confidence_cnn = list(prediction_cnn.values())[0]

    prediction_efficient = efficient_predict(img)
    classname_efficient = list(prediction_efficient.keys())[0]
    confidence_efficient = list(prediction_efficient.values())[0]

    prediction_inception = efficient_predict(img)
    classname_inception = list(prediction_inception.keys())[0]
    confidence_inception = list(prediction_inception.values())[0]

    COUNT += 1
    return render_template('prediction.html', data1=classname_mobilenet, data2=confidence_mobilenet,data3=classname_resnet,data4=confidence_resenet,data5=classname_cnn,data6=confidence_cnn,data7=classname_efficient,data8=confidence_efficient,data9=classname_inception,data10=confidence_inception)


def mobilenet_predict(image):
   
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}



def resnet_predict(image):
    
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = model1.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


def cnn_predict(image):
    
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = model2.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


def efficient_predict(image):
       
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = model3.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}

def inception_predict(image):
       
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = model4.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}




def predict(image):
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



