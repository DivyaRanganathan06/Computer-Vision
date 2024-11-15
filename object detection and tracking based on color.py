import imutils # resize
import cv2
import numpy as np # Make sure to import numpy

redLower = np.array([0, 0, 241])
redUpper = np.array([179,124,255])

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read() # read the frame

    frame = imutils.resize(frame, width=1000) #resize

    blurred = cv2.GaussianBlur(frame,(11,11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, redLower, redUpper) #mask the blue color
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"]/ M["m00"]), int(M["m01"]/ M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x),int(y)), int(radius), (0,255,255),2)
            cv2.circle(frame,center,5,(0,0,255),-1)
            #print(center, radius)
            if radius > 250:
                print("stop")
            else:
                if(center[0]<150):
                    print("Right")
                elif(center[0]>450):
                    print("Left")
                elif(radius<250):
                    print("Front")
                else:
                    print("Stop")
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
