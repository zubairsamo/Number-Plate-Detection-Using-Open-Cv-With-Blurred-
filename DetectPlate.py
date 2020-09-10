import cv2
import numpy as np
plat_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture('Car.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 80)


if (cap.isOpened()==False):
    print('Error Reading video')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    plate = plat_cascade.detectMultiScale(gray,scaleFactor=1.2,
    minNeighbors = 5)

    for (x,y,w,h) in plate:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        frame[y:y+h,x:x+w]=cv2.blur(frame[y:y+h,x:x+w],(25,25),cv2.BORDER_DEFAULT)
    if ret == True:
        cv2.imshow('Video',frame)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()