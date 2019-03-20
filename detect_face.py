# -*- coding: utf-8 -*-

import cv2
import imutils
import math

face_cascade = cv2.CascadeClassifier("face_detection_models\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("face_detection_models\haarcascade_eye.xml")

img = cv2.imread("standard_test_images/lena_color_512.tif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sunglasses = cv2.imread("sunglasses.png")

faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
#1.3 is a scaling factor, 
#5 is the minimum number of neighbouring positives for each classifier 
#in the cascade to consider for accepting the feature

for (x,y,w,h) in faces:
#    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    #roi = region of interest
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    if len(eyes)==2:
        (e1x,e1y,e1w,e1h) = eyes[0]
        (e2x,e2y,e2w,e2h) = eyes[1]
#        cv2.rectangle(roi_color, (e1x,e1y), (e1x+e1w, e1y+e1h), (0,255,0), 2)
#        cv2.rectangle(roi_color, (e2x,e2y), (e2x+e2w, e2y+e2h), (0,255,0), 2)
        
        glasses_width = (e2x+e2w-e1x)*1.1
        
        rotated_glasses = imutils.rotate_bound(sunglasses, 
                                      math.atan2((e2h+e2y-e1y-e1h),glasses_width))
        sized_glasses = cv2.resize(rotated_glasses, (int(glasses_width), e1h))
        alpha_s = sized_glasses[:, :] / 255.0
        alpha_l = 1.0 - alpha_s
    
        roi_color[e1y:e1y+e1h, e1x:int(glasses_width)+e1x] = (alpha_l * sized_glasses[:, :] +
                                      alpha_s * roi_color[e1y:e1y+e1h, e1x:int(glasses_width)+e1x])


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()