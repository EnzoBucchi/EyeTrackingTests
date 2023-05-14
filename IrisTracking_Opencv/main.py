import cv2
import numpy as np

cap = cv2.VideoCapture(0)

run = True
while run:
    _, frame = cap.read()
    
    rows, cols, _ = frame.shape
    gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    _, threshold = cv2.threshold(gray_roi, 15, 255, cv2.THRESH_BINARY_INV)
    
    # _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    # for cnt in contours:
    #     (x, y, w, h) = cv2.boundingRect(cnt)
    #     #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
    #     cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
    #     break
    
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Gray", gray_roi)
    cv2.imshow("Iris Tracking Opencv", frame)
    key = cv2.waitKey(1)
    if key == 27:
        run = False
        
cap.release()
cv2.destroyAllWindows()