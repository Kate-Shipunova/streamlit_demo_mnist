
import numpy as np
import streamlit as st
from PIL import Image 
from tensorflow.keras.models import load_model


MODEL_NAME = 'model_mnist.h5'
INPUT_SHAPE = (28, 28, 1)

def load_img():
  uploaded_file = st.file_uploader('Выберите изображение для распознавания', type=['png', 'jpg'])
  if uploaded_file:
    st.write('Вы загрузили:', uploaded_file)
    img_data = uploaded_file.getvalue()
    st.image(img_data)
    return uploaded_file


def preprocess_img(img):
  image = Image.open(img).convert('L')
  resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
  array = np.array(resized_image, dtype='float64') / 255
  array = array.reshape(-1, 28, 28, 1)

  return array


# app
model = load_model(MODEL_NAME)
st.title('Классификация цифр MNIST')
img = load_img()
result = st.button('Распознать изображение')
if result:
  x = preprocess_img(img)
  model_predict =model.predict(x)
  #st.write('model_predict = ', model_predict)
  model_predict = np.argmax(model_predict)
  st.write('**РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:**')
  st.write('На картинке цифра: ', model_predict)
