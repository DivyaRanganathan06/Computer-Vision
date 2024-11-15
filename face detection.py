import cv2

alg = r"C:\Users\Divya\Desktop\python\haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg) # loading algorithm
#or
#url = "..."
#cam = cv2.VideoCapture(url)
cam = cv2.VideoCapture(0) #cam id

while True:
    _,img = cam.read() #reading the frame from cam

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting colour img to gray scale img

    face = haar_cascade.detectMultiScale(grayImg,1.3,4) #getting coordinates

    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,0,255),2)
    cv2.imshow("FaceDetection",img)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
