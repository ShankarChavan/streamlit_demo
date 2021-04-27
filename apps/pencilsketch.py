import streamlit as st
import cv2
import numpy as np
from PIL import Image

def dodgeV2(x,y):
    return cv2.divide(x,255-y,scale=256)


def pencilsketch(inp_img):
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert,(21,21),sigmaX=0,sigmaY=0)
    final_img = dodgeV2(img_gray,img_smoothing)
    return final_img


def app():

    st.title("PencilSketcher-App")

    st.write(" This app helps you to convert your photos to pencil-sktech")

    file_image=st.sidebar.file_uploader("upload your photos",type=['jpeg','jpg','png'])

    col1,col2=st.beta_columns(2)

    if file_image is None:
        st.write("You haven't uploaded any image file")
    else:
        input_img=Image.open(file_image)
        final_img=pencilsketch(np.array(input_img))
        col1.header("**Original**")
        col1.image(file_image,use_column_width=True)
        col2.header("**Pencil-Sketch**")
        col2.image(final_img,use_column_width=True)



