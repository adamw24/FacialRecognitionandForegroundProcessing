import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyeglasses_cascade = cv2.CascadeClassifier(
    'haarcascade_eye_tree_eyeglasses.xml')

# read the image and convert to grayscale format
img = cv2.imread('FILE NAME HERE.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# calculate coordinates
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    if(len(eyes) > 2):
        eyes = eyeglasses_cascade.detectMultiScale(roi_gray)
    # draw bounding boxes around detected features
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
# plot the image
plt.imshow(img)
plt.show()
