import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'  # algorithm initializing
datasets = 'datasets'
sub_data = 'divya'  # data which we are going to create newly in datasets

path = os.path.join(datasets, sub_data)  # subdata/Divya
if not os.path.isdir(path):
    os.mkdir(path)  # make directory (this creates a new data)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)  # loading algorithm here (face_cascade)

webcam = cv2.VideoCapture(0)  # cam id

count = 1
while count < 30:  # no.of.photos we need for data
    print(count)
    (_, im) = webcam.read()  # _ tells us that it got the cam or not
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # converting into gray scale image

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # coordinates of the img(grayscaleimg,,)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]  # getting only the face as input data
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)  # storing the imgs in the order of number
    count += 1

    # Remove the cv2.imshow() to avoid the GUI error
    # cv2.imshow('OpenCV', im)  # This line has been removed

    key = cv2.waitKey(10)
    if key == 27:  # Exit the loop if 'ESC' is pressed
        break

webcam.release()
cv2.destroyAllWindows()  # Correct function name to destroy all windows
