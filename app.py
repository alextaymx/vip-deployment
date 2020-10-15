import streamlit as st
import numpy as np
# from PIL import Image
PAGE_CONFIG = {"page_title":"VIP.project","page_icon":":smiley:","layout":"centered"}
st.beta_set_page_config(**PAGE_CONFIG)
st.set_option('deprecation.showfileUploaderEncoding', False)
from functions import *
from tensorflow.python.platform import gfile

def main():
  # st.set_option('deprecation.showPyplotGlobalUse', False)
  st.title('** Sign Language Recognition **')
  # st.sidebar.info('**Switch between pages:**')
  st.sidebar.header('**VIP Project DEMO**')
  st.sidebar.subheader('This demo is maintained by:')
  st.sidebar.dataframe(get_group_members())
  # menu = ["Home","Plots","Trained model (SVM)"]
  # choice = st.sidebar.selectbox('Menu',menu)

  data_load_state = st.text('Loading data...')
  # Load model
  new_model = loadData()
  # Notify the reader that the data was successfully loaded.
  data_load_state.text("Model loading Done! (using st.cache)")

  # st.subheader("** Sign Language Recognition ** ")
  _=st.text('\n'),st.text('\n'),st.text('\n')
  st.info('''**MobileNet model**\n
    We are taking the model with highest accuracy to predict the hand gesture images
    The model is trained with ASL and MSL dataset''')
  # st.subheader("_Pretrained model : -> (Preprocessing is not shown)_")
  uploaded_file=None
  if uploaded_file is not None:
    sheader = st.warning('Please upload an image of your hand: ->')
  else:
    sheader = st.success('''**Image uploaded**\n
    Please look at the results below-->''')
  uploaded_file = st.file_uploader("Choose a image file", type=["png","jpg"])

  if uploaded_file is not None:
      st.balloons()
      # Convert the file to an opencv image.
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      opencv_image = cv2.imdecode(file_bytes, 1)

      ori = st.sidebar.image(opencv_image, channels="BGR", caption='image uploaded', use_column_width=True)
      # st.write(type(opencv_image),opencv_image.shape)

      # image = Image.open(uploaded_file)
      # img_array = np.array(image, dtype=np.uint8)
      # img2 = st.image(img_array, caption='Make sure this is a hand', use_column_width=True)
      # st.write(type(img_array),img_array.shape)

      resized = cv2.resize(opencv_image, (192,192), interpolation = cv2.INTER_AREA)
      reshaped = resized.reshape(1,192, 192,3)
      # img = st.image(resized, channels="BGR", caption='Make sure this is a hand', use_column_width=True)
      st.write("Input argument shape--> ",reshaped.shape)

      labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'G', 6: 'H', 7: 'I', 8: 'L', 9: 'O', 10: 'Q', 11: 'R', 12: 'S', 13: 'U', 14: 'V', 15: 'W', 16: 'Y'}
      predictions = new_model.predict(reshaped)
      predicted_class = np.argmax(predictions,axis=1).item(0)
      predicted_label = labels[predicted_class]
      st.warning("Predicted probability for each classes: ")
      st.dataframe(predictions)
      st.info('''**Predicted result: **\n
    Highest probability: {} ---> {}'''.format(predicted_class,predicted_label))
      # st.write(" ",predicted_class," --> ",predicted_label)
      
      # hand tracking
      detection_graph, sess = load_inference_graph()
      # Definite input and output Tensors for detection_graph
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      num_hands_detect=1
      score_thresh = 0.1

      image_np = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
      im_height,im_width,c= image_np.shape
      boxes, scores = detect_objects(image_np,detection_graph, sess)
      img = draw_box_on_image(num_hands_detect, score_thresh,scores, boxes, im_width, im_height,image_np)
      # plt.imshow(img)
      st.image(img, channels="RGB", caption='Bounding box', use_column_width=True)

  st.subheader('This demo is maintained by:')
  st.table(get_group_members())


if __name__ == '__main__':
  main()