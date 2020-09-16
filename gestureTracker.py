# import the necessary packages
import cv2
import numpy as np
from collections import OrderedDict

class GestureTracker():
	def __init__(self, maxDisappeared=40, maxPointsToTrack=30):
		self.leftHandObjId = 0
		self.rightHansObjId = 0
		self.leftHand = OrderedDict()
		self.rightHand = OrderedDict()

		self.leftDisappeared = 0
		self.rightDisappeared = 0

		self.maxDisappeared = maxDisappeared
		self.maxPointsToTrack = maxPointsToTrack

		self.confidence_threshold = 0.1

		self.leftHandSwipeLeft = False
		self.rightHandSwipeLeft = False
		self.leftHandSwipeRight = False
		self.rightHandSwipeRight = False
		self.leftHandSwipeUp = False
		self.rightHandSwipeUp = False
		self.leftHandSwipeDown = False
		self.rightHandSwipeDown = False

		self.leftHandWaving = False
		self.rightHandWaving = False

		self.swipeThreshold = 20

	def register(self, point, right=False):
		if right:
			self.rightHand[self.rightHansObjId] = point
			self.rightHansObjId += 1
		else:
			self.leftHand[self.leftHandObjId] = point
			self.leftHandObjId += 1

	def deregister(self, objectID, right=False):
		if right:
			del self.rightHand[objectID]
		else:
			del self.leftHand[objectID]

	def updateLeftHandMovements(self):
		self.leftHandSwipeLeft = False
		self.leftHandSwipeRight = False
		self.leftHandSwipeDown = False
		self.leftHandSwipeUp = False

		#self.updateLeftHandWaving()

		if self.leftHandObjId > 10 and self.leftHandWaving == False:
			first_item = next(iter(self.leftHand.items()))
			first_item = first_item[1]
			second_item = self.leftHand[self.leftHandObjId - 1]
			thresholdx = abs(second_item[0] - first_item[0])
			thresholdy = abs(second_item[1] - first_item[1])

			if first_item[0] < second_item[0] and thresholdx >= self.swipeThreshold:
				self.leftHandSwipeLeft = True

			if first_item[0] > second_item[0] and thresholdx >= self.swipeThreshold:
				self.leftHandSwipeRight = True

			if thresholdy > thresholdx:
				self.leftHandSwipeLeft = False
				self.leftHandSwipeRight = False

			if self.leftHandSwipeLeft == False and self.leftHandSwipeRight == False:
				if first_item[1] > second_item[1] and thresholdy >= self.swipeThreshold:
					self.leftHandSwipeUp = True

				if first_item[1] < second_item[1] and thresholdy >= self.swipeThreshold:
					self.leftHandSwipeDown = True

	def updateRightHandMovements(self):
		self.rightHandSwipeLeft = False
		self.rightHandSwipeRight = False
		self.rightHandSwipeUp = False
		self.rightHandSwipeDown = False

		#self.updateRightHandWaving()

		if self.rightHansObjId > 10 and self.rightHandWaving == False:
			first_item = next(iter(self.rightHand.items()))
			first_item = first_item[1]
			second_item = self.rightHand[self.rightHansObjId - 1]
			thresholdx = abs(second_item[0] - first_item[0])
			thresholdy = abs(second_item[1] - first_item[1])

			if first_item[0] < second_item[0] and thresholdx >= self.swipeThreshold:
				self.rightHandSwipeLeft = True

			if first_item[0] > second_item[0] and thresholdx >= self.swipeThreshold:
				self.rightHandSwipeRight = True

			if thresholdy > thresholdx:
				self.rightHandSwipeLeft = False
				self.rightHandSwipeRight = False

			if self.rightHandSwipeLeft == False and self.rightHandSwipeRight == False:
				if first_item[1] > second_item[1] and thresholdy >= self.swipeThreshold:
					self.rightHandSwipeUp = True

				if first_item[1] < second_item[1] and thresholdy >= self.swipeThreshold:
					self.rightHandSwipeDown = True

	def updateRightHandWaving(self):
		self.rightHandWaving = False
		count = 0
		increasing = 0
		decreasing = 0
		tlen = len(self.rightHand)
		for point in self.rightHand:
			if count > 0 and count + 3 >= tlen:
				if self.rightHand[count][0] > point[0] > self.rightHand[count + 2][0]:
					decreasing += 1
				if self.rightHand[count][0] < point[0] < self.rightHand[count + 2][0]:
					increasing += 1

		if tlen == 29 and abs(increasing - decreasing) <= 5:
			self.rightHandWaving = True

	def updateLeftHandWaving(self):
		self.leftHandWaving = False
		count = 0
		increasing = 0
		decreasing = 0
		tlen = len(self.leftHand)
		for point in self.leftHand:
			if count > 0 and count + 3 >= tlen:
				if self.leftHand[count][0] > point[0] > self.leftHand[count + 2][0]:
					decreasing += 1
				if self.leftHand[count][0] < point[0] < self.leftHand[count + 2][0]:
					increasing += 1

		if tlen == 29 and abs(increasing - decreasing) <= 5:
			self.leftHandWaving = True

	def update(self, person):

		leftDisp = True
		rightDisp = True
		leftElbow = None
		rightElbow = None
		leftWrist = None
		rightWrist = None

		for keypoint in person['keypoints']:
			if keypoint['part'] == 'leftWrist' and keypoint['score'] > self.confidence_threshold:
				leftWrist = keypoint['position']

			if keypoint['part'] == 'rightWrist' and keypoint['score'] > self.confidence_threshold:
				rightWrist = keypoint['position']

			if keypoint['part'] == 'leftElbow' and keypoint['score'] > self.confidence_threshold:
				leftElbow = keypoint['position']

			if keypoint['part'] == 'rightElbow' and keypoint['score'] > self.confidence_threshold:
				rightElbow = keypoint['position']

		if leftElbow is not None and leftWrist is not None:
			if leftElbow['y'] > leftWrist['y']:
				leftH = leftWrist
				point = (int(leftH['x']), int(leftH['y']))
				self.register(point)
				leftDisp = False
				self.leftDisappeared = 0

		if rightElbow is not None and rightWrist is not None:
			if rightElbow['y'] > rightWrist['y']:
				print(rightElbow['y'])
				print(rightWrist['y'])
				rightH = rightWrist
				point = (int(rightH['x']), int(rightH['y']))
				self.register(point, True)
				rightDisp = False
				self.rightDisappeared = 0



		if leftDisp:
			self.leftDisappeared += 1

		if rightDisp:
			self.rightDisappeared += 1

		if self.leftDisappeared > self.maxDisappeared:
			self.leftHandObjId = 0
			self.leftHand = OrderedDict()
			

		if self.rightDisappeared > self.maxDisappeared:
			self.rightHansObjId = 0
			self.rightHand = OrderedDict()


		if len (self.rightHand) >= self.maxPointsToTrack:
			first_item = next(iter(self.rightHand.items()))
			first_item = first_item[0]
			self.deregister(first_item, True)

		if len (self.leftHand) >= self.maxPointsToTrack:
			first_item = next(iter(self.leftHand.items()))
			first_item = first_item[0]
			self.deregister(first_item)

		self.updateLeftHandMovements()
		self.updateRightHandMovements()

		print(len(self.leftHand))
		print(len(self.rightHand))

		return 0


	def drawPoints(self, person, img, color):
		fontsize = 0.5
		for keypoint in person['keypoints']:
			if keypoint['part'] == 'leftWrist' and keypoint['score'] > self.confidence_threshold:
				leftHand = keypoint['position']
				point = (int(leftHand['x']), int(leftHand['y']))
				text = ""
				if self.leftHandSwipeLeft == True:
					text = "Left hand swipe left"
				if self.leftHandSwipeRight == True:
					text = "Left hand swipe right"
				if self.leftHandSwipeDown == True:
					text = "Left hand swipe down"
				if self.leftHandSwipeUp == True:
					text = "Left hand swipe up"
				if self.leftHandWaving == True:
					text = "Left hand waving"

				if text != "":
					cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, 1, cv2.LINE_AA)

			if keypoint['part'] == 'rightWrist' and keypoint['score'] > self.confidence_threshold:
				rightHand = keypoint['position']
				point = (int(rightHand['x']), int(rightHand['y']))
				text = ""
				if self.rightHandSwipeLeft == True:
					text = "Right hand swipe left"
				if self.rightHandSwipeRight == True:
					text = "Right hand swipe right"
				if self.rightHandSwipeDown == True:
					text = "Right hand swipe down"
				if self.rightHandSwipeUp == True:
					text = "Right hand swipe up"
				if self.rightHandWaving == True:
					text = "Right hand waving"
				if text != "":
					cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, 1, cv2.LINE_AA)
