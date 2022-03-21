import cv2
import s3fs
import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import json
import streamlit as st
from PIL import Image

fs = s3fs.S3FileSystem(anon=False)
@st.cache(ttl=600)
def read_file(filename):
    with fs.open(filename) as f:
        return f.read()



hide_streamlit_style = """
            
            
            
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title('Plant Leaf Disease Prediction')

def main():
    file_uploaded = st.file_uploader('Choose an image...', type = 'jpg')
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        image.save('static/{}.jpg')    

        img = cv2.imread('static/{}.jpg')

        #image1 = cv2.imread('image')

        prediction_mobilenet = mobilenet_predict(img)
        classname_mobilenet = list(prediction_mobilenet.keys())[0]
        confidence_mobilenet = list(prediction_mobilenet.values())[0]
        st.write('Mobilenet')
        st.write('Class Name: {}',format(classname_mobilenet))
        st.write('Confidence: {}%',format(confidence_mobilenet))

        prediction_resnet = resnet_predict(img)
        classname_resnet = list(prediction_resnet.keys())[0]
        confidence_resenet = list(prediction_resnet.values())[0]
        st.write('ResNet50')
        st.write('Class Name: {}',format(classname_resnet))
        st.write('Confidence: {}%',format(confidence_resenet))

        prediction_cnn = cnn_predict(img)
        classname_cnn = list(prediction_cnn.keys())[0]
        confidence_cnn = list(prediction_cnn.values())[0]
        st.write('Custom Cnn Model')
        st.write('Class Name: {}',format(classname_cnn))
        st.write('Confidence: {}%',format(confidence_cnn))

        prediction_efficient = efficient_predict(img)
        classname_efficient = list(prediction_efficient.keys())[0]
        confidence_efficient = list(prediction_efficient.values())[0]
        st.write('EfficientNetB0')
        st.write('Class Name: {}',format(classname_efficient))
        st.write('Confidence: {}%',format(confidence_efficient))

        prediction_inception = inception_predict(img)
        classname_inception = list(prediction_inception.keys())[0]
        confidence_inception = list(prediction_inception.values())[0]
        st.write('Inception')
        st.write('Class Name: {}',format(classname_inception))
        st.write('Confidence: {}%',format(confidence_inception))
        

        

        

        

        

def mobilenet_predict(image):
    classifier_model_mobilenet = read_file("plantdiseasemodel/mobilenetv1_model.h5")
    with st.spinner('Loading Model...'):
        classifier_model_mobilenet = tf.keras.models.load_model(r'models/mobilenetv1_model.h5',custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)
    print(image)
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
    
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = classifier_model_mobilenet.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}

def resnet_predict(image):
    classifier_model_resnet = read_file("plantdiseasemodel/resnet_model.h5")

    with st.spinner('Loading Model...'):
        classifier_model_resnet = tf.keras.models.load_model(r'models/resnet_model.h5',custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = classifier_model_resnet.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


def cnn_predict(image):
    classifier_model_cnn = read_file("plantdiseasemodel/cnn.h5")

    with st.spinner('Loading Model...'):
        classifier_model_cnn = tf.keras.models.load_model(r'models/cnn.h5',custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = classifier_model_cnn.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


def efficient_predict(image):
    classifier_model_efficient = read_file("plantdiseasemodel/efficientv2.h5")

    with st.spinner('Loading Model...'):
        classifier_model_efficient = tf.keras.models.load_model(r'models/efficientv2.h5',custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = classifier_model_efficient.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}

def inception_predict(image):
    classifier_model_inception = read_file("plantdiseasemodel/inception_model.h5")

    with st.spinner('Loading Model...'):
        classifier_model_inception = tf.keras.models.load_model(r'models/inception_model.h5',custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)   
    IMAGE_SHAPE = (224, 224)
    image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]) )
  
    image = image /255
    with open('static/categories.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = list(cat_to_name.values())
    probabilities = classifier_model_inception.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}
a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>
<div >
                <iframe width="560" height="315" src="https://www.youtube.com/embed/mARbQTw6IvQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
              </div>
<div class="footer">
<p align="center"> <a href="https://www.linkedin.com/in/prateek-singh-42356b12b/">Developed with ❤ by Prateek</a></p>
</div>
        """

st.markdown(footer, unsafe_allow_html = True)

if __name__ == '__main__' :
    main()