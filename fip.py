import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import cv2
import pickle

def model_prediction(test_images):
      model = tf.keras.models.load_model('model.h5')
      image= tf.keras.preprocessing.image.load_img(test_images,target_size=(224,224))
      input_arr=tf.keras.preprocessing.image.img_to_array(image)
      input_arr = input_arr / 255
      input_arr = np.expand_dims(input_arr,[0])
      pred = model.predict(input_arr)
      y_class = pred.argmax(axis=-1)
      y = " ".join(str(x) for x in y_class)
      y = int(y)
      with open("labels.pkl", "rb") as f:
           labels = pickle.load(f)
      res = labels[y]
      return res


st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

 # Main page
if (app_mode == "Home"):
     st.header("FRUITS AND VEGETABLE RECOGNITION SYSTEM")
     image_path = 'home img.png'
     st.image(image_path , use_column_width=True)

#About project
     
elif(app_mode=="About Project"):
     st.header("About Project")
     st.subheader("About Dataset")
     st.code("This dataset contains images of the following food items:")
     st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
     st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
     st.subheader("Content")
     st.text("This dataset contains three folders:")
     st.text("1.train (100 images each)")
     st.text("2.test (10 images each)")
     st.text("3.validation (10 images each)")
     #prediction page
elif(app_mode=="Prediction"):
     st.header("Model Prediction")
     test_images=st.file_uploader("Choose an image:")
     if(st.button("Show Image")):
          st.image(test_images,width=4,use_column_width=True)

     if test_images is not None:
        image = cv2.imdecode(np.frombuffer(test_images.read(), np.uint8), 1)          
#prediction button
     if(st.button("Predict")):
          st.write("Our Prediction")
          result = model_prediction(test_images)
          st.success(f"Model is predicting. It's a {result}")