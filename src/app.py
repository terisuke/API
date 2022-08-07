#ライブラリ
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image



from feat import Detector
# rf: https://py-feat.org/content/intro.html#available-models
face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "svm"
emotion_model = "resmasknet"
detector = Detector(
    face_model=face_model,
    landmark_model=landmark_model,
    au_model=au_model,
    emotion_model=emotion_model
)
from feat.utils import get_test_data_path





#本編  
st.title("Face emotion app")

img_source = st.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "カメラで撮影":
  img_file_buffer = st.camera_input("カメラで撮影")
elif img_source == "画像をアップロード":
  img_file_buffer = st.file_uploader("ファイルを選択")
else:
    pass




  
if img_file_buffer :
  img_file_buffer_2 = Image.open(img_file_buffer)
  img_file = np.array(img_file_buffer_2)
  cv2.imwrite('temporary.jpg', img_file)
  image_prediction = detector.detect_image("temporary.jpg")
  image_prediction = image_prediction[["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]]
  emotion = image_prediction.idxmax(axis = 1)[0]

  st.markdown("#### あなたの表情は")
  st.markdown("### {}です".format(emotion))
  
  
  
  





