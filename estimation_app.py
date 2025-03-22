import streamlit as st
from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

st.title("Human Pose Estimation OpenCV")
st.text('Make Sure you have a clear image with all the parts clearly visible')

img_file_buffer = st.file_uploader("Upload an image, Make sure you have a clear image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    pil_image = Image.open(img_file_buffer)
else:
    pil_image = Image.open(DEMO_IMAGE)

# Convert to numpy array
image = np.array(pil_image)

# Fix: Ensure it's 3 channels
if image.shape[-1] == 4:  # Has alpha channel
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

st.subheader('Original Image')
st.image(image, caption="Original Image", use_column_width=True)

thres = st.slider('Threshold for detecting the key points', min_value=0, value=20, max_value=100, step=5)
thres = thres / 100

@st.cache_data  # Updated cache function for newer versions of Streamlit
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    # Preprocess image
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    out = net.forward()
    out = out[:, :19, :, :]  # Get first 19 parts
    
    assert(len(BODY_PARTS) == out.shape[1])
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
        
    # Draw skeleton
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.circle(frame, points[idFrom], 4, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, points[idTo], 4, (0, 0, 255), cv2.FILLED)
            
    return frame

# Call pose detector
output = poseDetector(image)

st.subheader('Positions Estimated')
st.image(output, caption="Positions Estimated", use_column_width=True)
