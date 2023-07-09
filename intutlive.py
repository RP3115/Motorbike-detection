import cv2
import numpy as np
from PIL import Image
import imagehash
import os, os.path
from firebase_admin import credentials, firestore,storage
import firebase_admin
import time
import torch
import numpy as np
#from keras.models import load_model
#from PIL import Image
import cv2
import torch.nn as nn
from torchvision.transforms import transforms

cred = credentials.Certificate("D:\IIT Mandi\DP\Minh\g23dp-a4594-firebase-adminsdk-xe0j0-4187f62533.json")
store = "g23dp-a4594.appspot.com"
app = firebase_admin.initialize_app(cred,{'storageBucket':store})

db = firestore.client()
bucket = storage.bucket()

#def build():
transformer=transforms.Compose([
transforms.Resize((150,150)),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                    [0.5,0.5,0.5])
])

class ConvNet(nn.Module):
    def __init__(self,num_classes=6):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,150,150)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (256,12,150,150)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,75,75)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (256,20,75,75)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (256,32,75,75)
        
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
        
        
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
        #Above output will be in matrix form, with shape (256,32,75,75)
            
        output=output.view(-1,32*75*75)
            
            
        output=self.fc(output)
            
        return output
    
#importing model
model = ConvNet(num_classes=2)
checkpoint = torch.load('D:/IIT Mandi/DP/helm/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


def draw_prediction(frame, classes, classId, conf, left, top, right, bottom):
    # if classId in {3}:
    
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

def crop_img(img,root):
    filepath = os.path.join(root, img)
    imp=Image.open(filepath)
    width =imp.width
    height = imp.height
    img_cropped = imp.crop((0, 0, int(width/2), int(height/2)))
    return img_cropped

def uploader():
    model.eval()
    direc=r"D:\IIT Mandi\DP\Minh\Bikes"
    os.chdir(direc)
    for filename in os.listdir(direc):
        file2=crop_img(filename,direc)
        img2=transformer(file2)
        val=model(img2.unsqueeze(0))
        val=val.detach().numpy()
        if val[0][0]<0:
            # print('xd')
            blob = bucket.blob(filename)
            blob.upload_from_filename(filename)
            blob.make_public()
            db.collection('bikeslap').document(filename).set({'url':blob.public_url})
    os.chdir(r'D:\IIT Mandi\DP\Minh')

def grouper():
    #deleting waste files
    for root, _, files in os.walk("D:\IIT Mandi\DP\Minh\Bikes"):
            for f in files:
                fullpath = os.path.join(root, f)
                try:
                    if os.path.getsize(fullpath) < 20 * 1024:   #set file size in kb
                        #print(fullpath)
                        os.remove(fullpath)
                except WindowsError:
                    continue
    #time.sleep(0.5)

    threshold = 28 # Define the threshold value for similarity detection
    image_dir = "D:\IIT Mandi\DP\Minh\Bikes"  # Replace with the path to the directory containing the images
    # Loop over all images in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Check that the file is an image
            filepath = os.path.join(image_dir, filename)
            if os.path.exists(filepath):
                # Load the image using Pillow
                image1 = Image.open(filepath)
                # Calculate the perceptual hash of the image using imagehash
                hash1 = imagehash.phash(image1)
                # Compare the image to all other images in the directory
                for other_filename in os.listdir(image_dir):
                    if other_filename != filename and (other_filename.endswith(".jpg") or other_filename.endswith(".png")):
                        other_filepath = os.path.join(image_dir, other_filename)
                        # Load the other image using Pillow
                        image2 = Image.open(other_filepath)
                        # Calculate the perceptual hash of the other image using imagehash
                        hash2 = imagehash.phash(image2)
                        # If the hamming distance between the hashes is less than or equal to the threshold, delete the current image
                        if hash1 - hash2 <= threshold:
                            if os.path.getsize(filepath)<os.path.getsize(other_filepath):
                                print(os.path.getsize(filepath))
                                os.remove(filepath)
                                break
                            else:
                                print(os.path.getsize(other_filepath))
                                os.remove(other_filepath)
                            # os.remove(filepath)
                            # break  # Once an image is deleted, stop comparing it to other images in the directory
    #time.sleep(1)    

def cutout(frame, classes, classId, conf, left, top, right, bottom,c,p,fr):
    #cropping
    # if classId in {0}:
    #     ari=frame[top:bottom, left:right]
    #     if ari.size !=0:
    #         directory=r'D:\IIT Mandi\DP\Minh\peeps'
    #         os.chdir(directory)
    #         filename = 'person{}_{}.jpg'.format(fr,p)
    #         cv2.imwrite(filename, ari)
    #         os.chdir(r'D:\IIT Mandi\DP\Minh')
    


        
    #     #print("done{}".format(c))
    #     # blob = bucket.blob(filename)
    #     # blob.upload_from_filename(filename)
    #     # blob.make_public()
    #     # time.sleep(0.75)
    #     # db.collection('People').document(filename).set({'url': blob.public_url})

    if classId in {3}:
        ari=frame[top:bottom, left:right]
        if ari.size !=0:
            directory=r'D:\IIT Mandi\DP\Minh\Bikes'
            os.chdir(directory)
            filename = 'bike{}_{}.jpg'.format(fr,c)
            cv2.imwrite(filename, ari)
            #print("done{}".format(c))
            # blob = bucket.blob(filename)
            # blob.upload_from_filename(filename)
            # blob.make_public()
            # time.sleep(0.75)
            # db.collection('Bikes').document(filename).set({'url': blob.public_url})
            os.chdir(r'D:\IIT Mandi\DP\Minh')

    if fr%150==1:
        grouper()
        uploader()
    

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
            cutout(frame, classes, classIds[i], confidences[i], max(0,left-(int((0)*width))), max(0,top-int((0.6*height))), min(frameWidth,left + int(1*width)), min(frameHeight,top + (height)),c,p,fr)
        # if classIds[i] in {0}:
        #     p=p+1
        #     cutout(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height,c,p,fr)

        #draw_prediction(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)

    cv2.waitKey(0)