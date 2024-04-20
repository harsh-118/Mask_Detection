import tensorflow as tf
import cv2
import numpy as np
import PIL
from PIL import Image

model=tf.keras.models.load_model("Harsh.h5")

haarcasccade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Haarcascade is an algorithm to detect objects in image irrespective of scale of image.
#It can be used to detect very simple object.

#Strategy:-First find where there is face
#On those coordinate detect mask through our cnn model
#label that


#In haarcascade classifier , it stores the feature of images with that object
#In the form of pixel correlation in an xml file


cap=cv2.VideoCapture(0)

while cap.isOpened():  
     b,frame=cap.read()   
     faces=haarcasccade.detectMultiScale(frame,scaleFactor=1.10,minNeighbors=4)
     #This function is used to find all coordinates where there is face
     #It returns list of coordinates(x,y,w,h)
     #Exp:-[(23,45,67,89),(120,30,400,50).........]
     #scaleFactor-->Parameter specifying how much the image size is reduced at each
     #image scale
     #minNeighbors-->Parameter specifying how many neighbors each candidate rectangle
     #should have to keep
     for x,y,w,h in faces:
          face=frame[y:y+h,x:x+w]
          cv2.imwrite('face.jpg',face)
          face=tf.keras.preprocessing.image.load_img("face.jpg",target_size=(150,150,3))
          #Converting format of face into keras format
          face=tf.keras.preprocessing.image.img_to_array(face)
          #Coverting image into numpy array
          
          #Converting image into 4D
          face=np.expand_dims(face,axis=0)
          #(150,150,3)----->(1,150,150,3)
          
          ans=model.predict(face)
          if ans>0.5:
              cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
              cv2.putText(frame,"Without Mask",(x//2,y//2),0,2,(0,0,255),3)
          else:
              cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
              cv2.putText(frame,"With Mask",(x//2,y//2),0,2,(0,255,0),3)
     
     cv2.imshow('window',frame)
     if cv2.waitKey(1)==113:
           break
       
cap.release()  #Release out buffer
cv2.destroyAllWindows()        