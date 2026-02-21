import os
import cv2 as c
import numpy as n

people = ['anupama','rdj','sai pallavi']
dir = r'D:\New folder\New folder\openCV\face recognition'
haar_cascade = c.CascadeClassifier('haar_face.xml')

features = []
labels = []

def face_train():
    for person in people:
        path = os.path.join(dir,person)
        # print(path)
        label = people.index(person)
        # print(label)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            # print(img_path)
            img_arr = c.imread(img_path)
            gray= c.cvtColor(img_arr,c.COLOR_BGR2GRAY)
            face_ret = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            # print(face_ret)
            for (x,y,w,h) in face_ret:
                face_roi = gray[y:y+h,x:x+w]
                # print(face_roi)
                features.append(face_roi)
                labels.append(label)

face_train()
print('face done--------------------')
features = n.array(features,dtype='object')
labels= n.array(labels)
face_reccognizer = c.face.LBPHFaceRecognizer_create()

#train face recognizer on the features list and labels list
face_reccognizer.train(features,labels)
face_reccognizer.save('face_trained.yml')

n.save('features.npy',features)
n.save('labels.npy',labels)

# print(features)
# print(labels)

c.waitKey(0)

# print(f'length of feature: {len(features)}')
# print(f'length of labels: {len(labels)}')
            
# p =[]

# for i in os.listdir(r'D:\New folder\New folder\openCV\face recognition'):
#     p.append(i)

# print(p)
