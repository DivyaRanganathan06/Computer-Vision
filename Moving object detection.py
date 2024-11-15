import cv2 #library
import imutils #resize

cam = cv2.VideoCapture(0) # cam id

firstFrame=None
area = 500

while True:  
    _,img = cam.read() # read from cam
    text = "Normal"

    img = imutils.resize(img, width=500) #resize

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # colour 2 gray scale img
 
    gaussianImg = cv2.GaussianBlur(grayImg, (21,21),0) #smoothening

    if firstFrame is None:
        firstFrame = gaussianImg #capturing the first time
        continue
    imgDiff = cv2.absdiff(firstFrame, gaussianImg) # absolute difference

    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]

    threshImg = cv2.dilate(threshImg,None,iterations=2) # correct the left over parts

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)#make complete contours
    cnts = imutils.grab_contours(cnts) # grabbing all the contours
    for c in cnts:
        if cv2.contourArea(c) < area: # make full area
            continue
        (x, y, w, h) = cv2.boundingRect(c) # rectangle of the contour
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        text = "Moving objects detected"
    print(text)
    cv2.putText(img, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # text which appear on cam frame
    cv2.imshow("cameraFeed",img)

    key = cv2.waitKey(10) # steps to stop the process
    print(key)
    if key == ord("q"):
        break
  
cam.release()
cv2.destroyAllWindows()









    
