import cv2
import numpy as np
import argparse
from utils3 import process_frame, draw_prediction, cutout, grouper
import sys


# parser = argparse.ArgumentParser()
# parser.add_argument('--video', help='Path to input video', required=True)
# args = parser.parse_args()

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

# Read COCO dataset classes
with open('coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the network with YOLOv3 weights and config using darknet framework
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
#net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg", "darknet")

# Get the output layer names used for forward pass
outNames = net.getUnconnectedOutLayersNames()

# define a video capture object
vid = cv2.VideoCapture(0)
fr=0
while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
        break
    
    if frame is not None:
        fr=fr+1
        # Read image from command line arguments
        #image = cv2.imread(frame)
        # Create blob from image
        if fr%5 ==1:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            
            # Set the input
            net.setInput(blob)

            # Run forward pass
            outs = net.forward(outNames)
            process_frame(frame, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD,fr)

            # Display the resulting frame
            #cv2.imshow('output', frame)
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


grouper()
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()