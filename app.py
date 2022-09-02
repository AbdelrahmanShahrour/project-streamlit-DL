import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import pandas as pd
import cv2
from PIL import Image, ImageOps
import tensorflow as tf


st.set_page_config(page_title="image classification project", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://lottie.host/16608717-26ef-424f-b7f1-5eeeff8d4f81/EspV8rVdyG.json")
lottie_H = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_ha3tc6tl.json")

# ---- HEADER SECTION ----
with st.container():
    st.subheader("Hi there, I am Abdalrahman Shahrour :bar_chart:")
    # logatta
    # st_lottie(logatta, height=300, key="codi")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Project : image classification project accuracy_score = 88.2")
        # st.write(
        #     "My data contains about 24,000 records"
        # )
        st.write("[My Kaggle accaunt >](https://www.kaggle.com/abdalrahmanshahrour)")
        st.write("[My LinkedIn accaunt >](https://www.linkedin.com/in/shahrour/)")
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")
    
    uploaded_file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    

    if uploaded_file is not None:

        def imagetopredict(image):
            size = (32,32)    
            image = ImageOps.fit(image, size)
            image = np.asarray(image)
            return np.vstack([image])
        image = Image.open(uploaded_file)
        p = imagetopredict(image)
        new_model = tf.keras.models.load_model('cnnmodel')
        lable = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
        st.subheader(lable[int(np.argmax(new_model.predict(np.array([p]))))])
        