import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st
st.title("Dog Cat Classifier using Tensorflow and Keras")

#step 1 :Load the model
model=load_model('cats_dogs_small_3.h5')

# Step2 Load the image 
uploaded_file=st.file_uploader('Choose the database', accept_multiple_files=False)
if uploaded_file is not None:
    file=uploaded_file
else:
    file='image.jpg'

if st.checkbox('View Image', False):
    img=Image.open(file)
    st.image(img)
    
# Step3: Preprocess the uploaded image
img=load_img(file, target_size=(150, 150))
img_array=img_to_array(img)
img_array=np.expand_dims(img_array, axis=0)

# Step4: Predict the image and print the result
pred=int(model.predict(img_array)[0][0])

if st.button('Prediction'):
    if pred==1:
        st.subheader("It is a Dog  ")
    else:
        st.subheader("It is a Cat ")
