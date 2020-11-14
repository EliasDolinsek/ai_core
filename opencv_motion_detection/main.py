import cv2
import imutils

video = cv2.VideoCapture(1)

first_frame = None
while True:
    ret, frame = video.read()
    frame = imutils.resize(frame, width=500)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
    if first_frame is None:
        first_frame = gray_frame
        continue
   
    frames_delta = cv2.absdiff(first_frame, gray_frame)
    thresh = cv2.threshold(frames_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2) 

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    first_frame = gray_frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) % 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
