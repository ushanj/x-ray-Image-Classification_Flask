#http://buffml.com/

from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import image_utils
import numpy as np 
import cv2

app = Flask(__name__)



#dic = {0 : 'Lung Cancer', 1 : 'Healthy', 2 : 'tb' }


#Image Size
img_size=256
#model = load_model('model2.h5')
model = load_model('model_vgg16_2.h5')


model.make_predict_function()

# def predict_label(img_path):
#     img=cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     resized=cv2.resize(gray,(img_size,img_size)) 
#     i = image_utils.img_to_array(resized)/255.0
#     i = i.reshape(1,img_size,img_size,1)
#     #p = model.predict_classes(i)
#     predict_x=model.predict(i) 
#     p=np.argmax(predict_x,axis=1)
#     return dic[p[0]]

# def predict_label(img_path):
#     img = image_utils.load_img(img_path, target_size=(256,256))
#     # This target size should be equal with previously added in code 
#     print("server started")
#     img = image_utils.img_to_array(img)/255
#     img = np.array([img])
#     img.shape

#     predict_x=model.predict(img) 
#     classes_x=np.argmax(predict_x,axis=1)
#     print(classes_x)
#     if classes_x[0]== 0:
#         result = 'Lung Cancer'
#     elif classes_x[0]== 1:
#         result = 'Healthy'
#     else: 
#         result = 'TB'
    
#     return result

def predict_label(img_path):
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # update the data and labels lists, respectively
    #data.append(image)
    #labels.append(label)
    img = np.array(image) / 255.0
    #labels = np.array(labels) 
    img = np.array([img])
    img.shape
    predict_x=model.predict(img) 
    classes_x=np.argmax(predict_x,axis=1)
    if classes_x[0]== 0:
        result = 'Lung Cancer'
    elif classes_x[0]== 1:
        result = 'Healthy'
    else: 
        result = 'Tuberculosis'
    
    return result




# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/predict", methods = ['GET', 'POST'])
def upload():
    
    if request.method == 'POST':
       img = request.files['file']
       img_path = "uploads/" + img.filename  
       print(img_path)  
       img.save(img_path)
       p = predict_label(img_path)
       print("server started")
       print(p)
       return str(p).lower()

if __name__ =='__main__':
    #app.debug = True
    app.run( port = 5000 ,debug = True)