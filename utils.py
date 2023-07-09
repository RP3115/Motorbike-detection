import cv2
import numpy as np
import os
from firebase_admin import credentials, firestore,storage
import firebase_admin
import time

cred = credentials.Certificate("D:\IIT Mandi\DP\Minh\g23dp-a4594-firebase-adminsdk-xe0j0-4187f62533.json")
store = "g23dp-a4594.appspot.com"
app = firebase_admin.initialize_app(cred,{'storageBucket':store})

db = firestore.client()
bucket = storage.bucket()

def draw_prediction(frame, classes, classId, conf, left, top, right, bottom):
    if classId in {3}:
        
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (150, 90, 150))

        # Assign confidence to label
        label = '%.2f' % conf
        
        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def cutout(frame, classes, classId, conf, left, top, right, bottom,c,p,fr):
    #cropping
    # if classId in {0}:
    #     ari=frame[top:bottom, left:right]
        
    #     directory=r'D:\IIT Mandi\DP\Minh\peeps'
    #     os.chdir(directory)
    #     filename = 'person{}_{}.jpg'.format(fr,p)
    #     cv2.imwrite(filename, ari)
        
    #     #print("done{}".format(c))
    #     blob = bucket.blob(filename)
    #     blob.upload_from_filename(filename)
    #     blob.make_public()
    #     time.sleep(0.75)
    #     db.collection('People').document(filename).set({'url': blob.public_url})
    #     os.chdir(r'D:\IIT Mandi\DP\Minh')
        
    if classId in {3}:
        ari=frame[top:bottom, left:right]
        directory=r'D:\IIT Mandi\DP\Minh\bike'
        os.chdir(directory)
        filename = 'bike{}_{}.jpg'.format(fr,p)
        cv2.imwrite(filename, ari)
        #print("done{}".format(c))
        blob = bucket.blob(filename)
        blob.upload_from_filename(filename)
        blob.make_public()
        time.sleep(0.75)
        db.collection('Bikes').document(filename).set({'url': blob.public_url})
        os.chdir(r'D:\IIT Mandi\DP\Minh')
    

# Process frame, eliminating boxes with low confidence scores and applying non-max suppression
def process_frame(frame, outs, classes, confThreshold, nmsThreshold,fr):
    # Get the width and height of the image
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Network produces output blob with a shape NxC where N is a number of
    # detected objects and C is a number of classes + 4 where the first 4
    # numbers are [center_x, center_y, width, height]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if (confidence > confThreshold) :
                # Scale the detected coordinates back to the frame's original width and height
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # Save the classId, confidence and bounding box for later use
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])


    c=0
    p=0
    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        #i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        if classIds[i] in {3}:
            c=c+1
            cutout(frame, classes, classIds[i], confidences[i], left, top-int(0.5*height), left + width, top + (height),c,p,fr)
        # if classIds[i] in {0}:
        #     p=p+1
        #     cutout(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height,c,p,fr)
        
        #draw_prediction(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)
        
    cv2.waitKey(0)