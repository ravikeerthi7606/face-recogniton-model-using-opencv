import cv2 as c

img = c.imread('photos/people 2.jpg')
c.imshow('anupama',img)

gray = c.cvtColor(img, c.COLOR_BGR2GRAY)
c.imshow('gray',gray)

haar_cascade = c.CascadeClassifier('haar_face.xml')
# print(haar_cascade)
face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
# print(face_rect)
print(f'no of faces found:{len(face_rect)}')

for (x,y,w,h) in face_rect:
    c.rectangle(img, (x,y),(x+w,y+h),(0,255,0),thickness=2)
    # print(x,y,w,h)
c.imshow('detected',img)

c.waitKey(0)