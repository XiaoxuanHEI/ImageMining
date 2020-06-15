import numpy as np
import cv2
import math

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
	global r,c,w_roi,h_roi,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h_roi = abs(r2-r)
		w_roi = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True


cap = cv2.VideoCapture('Test-Videos/Antoine_Mug.mp4')
ret,frame = cap.read()
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r, c), (r+h_roi, c+w_roi), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function


track_window = (r,c,h_roi,w_roi)

c_y = c + h_roi/2
c_x = r + w_roi/2
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Rtable = [None] * 360
for y in range(r, r + h_roi):
	for x in range(c, c + w_roi):
		gx = img[y, x + 1] - img[y, x]
		gy = img[y + 1, x] - img[y, x]
		angle = math.atan2(gx, gy)
		angle = int(angle * 180 / math.pi)
		if Rtable[angle] == None:
			Rtable[angle] = [(c_y - y,c_x - x)]
		else:
			Rtable[angle].append((c_y - y,c_x - x ))


cpt = 1
while (1):
	ret, frame = cap.read()
	if ret == True:
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		(h, w) = img.shape
		# print("Image dimension:", h, "rows x", w, "columns")
		# H = np.zores((h, w),
		H = np.zeros((h, w),dtype=np.uint8)
		for y in range(0, h - 1):
			for x in range(0, w - 1):
				gx = img[y, x + 1] - img[y, x]
				gy = img[y + 1, x] - img[y, x]
				val = math.sqrt(gx * gx + gy * gy)
				if val > 5:
					angle = math.atan2(gx, gy)
					angle = int(angle * 180 / math.pi)
					if Rtable[angle] != None:
						for i in Rtable[angle]:
							v_y, v_x = i
							y1 = int(v_y+y)
							x1 = int(v_x+x)
							if y1<h and x1<w:
								H[y1, x1] += 1
				H[y,x] = min(H[y,x],255)

		MaxH = 0
		Maxy = 0
		Maxx = 0
		for y in range(1,h):
			for x in range(1,w):
				if H[y,x]>MaxH:
					Maxy = y
					Maxx = x 
					MaxH = H[y,x]

		x = int(Maxy - h_roi / 2)
		y = int(Maxx - w_roi / 2)
		
		print(x,y,MaxH)

		frame_tracked = cv2.rectangle(frame, (x,y), (x+h_roi,y+w_roi), (255,0,0) ,2)
		cv2.imshow('Sequence',frame_tracked)
				
		cv2.imshow('H', H)
		
		

		k = cv2.waitKey(60) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('Q4_H_%04d.png' % cpt, H)
			cv2.imwrite('Q4_Frame_%04d.png' % cpt, frame_tracked)

		cpt += 1
	else:
		break



cv2.destroyAllWindows()
cap.release()

