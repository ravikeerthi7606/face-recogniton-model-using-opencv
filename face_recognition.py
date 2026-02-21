import numpy as n
import cv2 as c

haar_cascade = c.CascadeClassifier('haar_face.xml')

people = ['anupama','rdj','sai pallavi']
features = n.load('features.npy',allow_pickle=True)
labels =n.load('labels.npy')

face_reccognizer = c.face.LBPHFaceRecognizer_create()
face_reccognizer.read('face_trained.yml')

img = c.imread(r'D:\New folder\New folder\openCV\face recognition\train\cdee82ae-3929-4bb2-9cd9-1467565d265a.jpeg')
gray = c.cvtColor(img,c.COLOR_BGR2GRAY)
c.imshow('person',gray)

#detect the person
face_rect = haar_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)

for (x,y,w,h) in face_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label ,confidence = face_reccognizer.predict(faces_roi)
    print(f'label = {people[label]} with confidence of {confidence}')
    c.putText(img,str(people[label]),(100,100),c.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    c.rectangle(img, (x,y),(x+w,y+h),(0,0,255),thickness=1)

c.imshow('detected image',img)
c.waitKey(0)