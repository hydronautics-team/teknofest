import cv2
import numpy as np
from teknofest_object_detection_msgs.msg import Object, ObjectsArray

def detector(self, img):
	kernel = np.ones((5,5), np.uint8)

	frame = cv2.imread(img)

	while True:
		# frame = cv2.bilateralFilter(frame, 2, 75, 75)
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		hl = 150
		sl = 0
		vl = 0

		h = 180
		s = 250
		v = 250

		lower = np.array([hl, sl, vl])
		upper = np.array([h, s, v])
		mask = cv2.inRange(hsv, lower, upper)
		res = cv2.bitwise_and(frame, frame, mask = mask)
		opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		dil = cv2.dilate(closing, kernel, iterations = 1)

		contours, h = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)

		objects_array_msg = ObjectsArray()

		for i in range(len(contours)):
			area = cv2.contourArea(contours[i])
			if area > 100:
				p = cv2.arcLength(contours[i], True)
				num = cv2.approxPolyDP(contours[i], 0.01 * p, True)
				if len(num) < 7 or len(num) > 10:
					continue

				x, y, w, h = cv2.boundingRect(contours[x])
				object_msg = Object()
				object_msg.name = 'container_coordinates'
                object_msg.top_left_x = x
                object_msg.top_left_y = y
                object_msg.bottom_right_x = x+w
                object_msg.bottom_right_y = y+h
                objects_array_msg.objects.append(object_msg)
	return objects_array_msg