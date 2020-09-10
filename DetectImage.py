import cv2
import numpy as np

platecascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

img = cv2.imread('car.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plate = platecascade.detectMultiScale(img,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))

for (x,y,w,h) in plate:
    edge=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    plate=img[y:y+h,x:x+w]
    blur=cv2.blur(plate,ksize=(20,20))
    img[y:y+h,x:x+w]=blur
cv2.imshow('plate',img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
