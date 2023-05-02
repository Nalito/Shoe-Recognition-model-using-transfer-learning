from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_hub as hub

model = tf.keras.models.load_model(('model.h5'), custom_objects={'KerasLayer':hub.KerasLayer})
class_names = ['boot', 'flip flop', 'loafer', 'sandal', 'sneaker', 'soccer shoe']

def load_and_prep_image(filename, img_shape=224):
    '''
    This function reads and image and formats it
    
    Input:
    - filename: path to image file
    - img_shape: preferred shape of image
    
    Output:
    img: formatted image
    '''
    
    img = tf.io.read_file(filename)
    
    img = tf.image.decode_image(img)
    
    img = tf.image.resize(img, size=[224, 224])
    
    img = img/255.
 
    return img

def predict_image(model, filename, class_names=class_names):

    img = load_and_prep_image(filename, img_shape=224)

    pred = model.predict(tf.expand_dims(img, 0))

    index = tf.argmax(pred[0])

    print(index)

    prediction = class_names[int(index)]

    return prediction


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method=='POST':
        img = request.files['image']
        path = './images/' + img.filename
        img.save(path)

        prediction = predict_image(model, path, class_names)

        return render_template('index.html', prediction="The shoe is a ".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
