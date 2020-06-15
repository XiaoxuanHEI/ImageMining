import cv2
import numpy as np
import math

cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')

cpt = 1
while (1):
    ret, frame = cap.read()
    if ret == True:

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1)
        sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0)
        gradxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)


        abs_sobel64f = np.absolute(gradxy)
        img_8u = np.uint8(abs_sobel64f)

        (h, w) = img_8u.shape
        imgRGB = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)

        for y in range(1,h-1):
            for x in range(1,w-1):
                if img_8u[y, x] < 40:
                    imgRGB[y, x] = (0,0,255)


        cv2.imshow('Sequence', imgRGB)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Q3_Frame_%04d.png' % cpt, imgRGB)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
