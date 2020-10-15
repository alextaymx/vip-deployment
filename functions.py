import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.platform import gfile
# import os

# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.applications import MobileNet
# from keras.applications.mobilenet import preprocess_input
# from keras.layers import *
# from tensorflow import keras
# from keras.losses import sparse_categorical_crossentropy


@st.cache
def get_group_members():
    # Group Members details
    data = {"No": [1, 2, 3],
            "Group Members Name": ["Teo Sheng Pu", "Alex Tay Mao Xiang", "Liew Jun Xian"],
            "Student ID": [1171101665, 1171101755, 1171303519],
            "Contact Number": ["+6010-7976911", "+6014-9888150", "+6012-7385789"],
            "e-Mail Address": ["1171101665@student.mmu.edu.my", "1171101775@student.mmu.edu.my", "1171303519@student.mmu.edu.my"]}

    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.set_index("No")
    return df


@st.cache(allow_output_mutation=True)
def loadData():
    new_model = tf.keras.models.load_model('aslmslNotTrainableLatest.h5')
    return new_model


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(len(scores)):
        (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                      boxes[i][0] * im_height, boxes[i][2] * im_height)

        boxArea = (right-left)*(bottom-top)
        imgArea = (im_width)*(im_height)
        # print(boxArea,"box area")
        # print(imgArea,"img area")
        # print(boxArea/imgArea,"ratio")
        if (boxArea/imgArea >= 0.1):
            # p1 = (int(left), int(top))
            # p2 = (int(right), int(bottom))
            # cv2.rectangle(image_np, p1, p2, (77, 255, 9), 50, 50)
            return image_np[int(top):int(bottom), int(left):int(right), :]


@st.cache(allow_output_mutation=True)
def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(len(scores)):
        (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                      boxes[i][0] * im_height, boxes[i][2] * im_height)

        boxArea = (right-left)*(bottom-top)
        imgArea = (im_width)*(im_height)
        # print(boxArea,"box area")
        # print(imgArea,"img area")
        # print(boxArea/imgArea,"ratio")
        if (boxArea/imgArea >= 0.1):
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            img = cv2.rectangle(image_np, p1, p2, (77, 255, 9), 10, 10)
            return img
            # return image_np[int(top):int(bottom),int(left):int(right),:]
