import cv2 
import HandTrackingModule as HTM
import cvzone
import numpy as np

#resolucion:
resolution = 720, 1080
colorR = 255, 0, 255
cx, cy, w, h = 50, 50, 100, 100

cap = cv2.VideoCapture(1)
cap.set(3, resolution[0])
cap.set(4, resolution[1]) 
detector = HTM.handDetector(detectionCon=0.8)

class DragRect():
    def __init__(self, posCenter, size=[100,100]):
        self.posCenter = posCenter
        self.size = size
    
    def update(self, cursor):

        cx, cy = self.posCenter
        w, h = self.size

        if cx-w//2 < cursor[1] < cx+w//2 and  cy-h//2 < cursor[2] <  cy+h//2:
                self.posCenter = cursor[1], cursor[2]

rectList = []

for x in range(4):
    rectList.append(DragRect([x*150+100,50]))


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)

#----------------------------------------------------------------------------------------------------------------------------
    if lmList:

        l, _, _ = detector.findDistance(8, 12, img, draw=False)

        if l < 38:
            cursor = lmList[8]
            for rect in rectList:
                rect.update(cursor)   
        
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx-w//2, cy-h//2, w, h), rt=0)
    out = img.copy()
    alpha = 0.1
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
#----------------------------------------------------------------------------------------------------------------------------

    cv2.imshow("Image", out)
    
    k = cv2.waitKey(1)
    if k == 27:
        break

