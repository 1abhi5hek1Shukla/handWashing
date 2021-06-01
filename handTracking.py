import cv2
import mediapipe as mp
import time
import numpy as np

class handDetector(object):
	"""docstring for handDetector"""
	def __init__(self, mode = False, maxHands = 2, detectionConfidence = 0.5, trackConfidence = 0.5):
		self.mode = mode
		self.maxHands = maxHands
		self.detectionConfidence = detectionConfidence
		self.trackConfidence = trackConfidence

		
		self.mp_hands = mp.solutions.hands
		self.mp_drawing = mp.solutions.drawing_utils

		self.hands =  self.mp_hands.Hands(self.mode,
				self.maxHands, self.detectionConfidence,
				self.trackConfidence)

	def findHands(self, img, drawHands = True):

		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(image)
		mask = np.zeros_like(img)
		
		if self.results.multi_hand_landmarks:
			for hand_landmarks in self.results.multi_hand_landmarks:
				if drawHands:
					self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
		return img

	def findPosition(self, img, handNo=0, draw = True):
		lmList = []

		if self.results.multi_hand_landmarks:
			myhand = self.results.multi_hand_landmarks[handNo]

			for _id, lm in enumerate(myhand.landmark):
				h,w,c = img.shape
				cx,cy = int(lm.x * w), int(lm.y * h)
				lmList.append([_id,cx,cy])
				if draw:
					cv2.circle(img, (cx, cy), 7,(255,0,0), cv2.FILLED)

		return lmList

def main():
	cap = cv2.VideoCapture(0)
	detector = handDetector()

	pTime = 0

	while cap.isOpened():
		ret, frame = cap.read()
		
		if not ret:
			continue
		
		cTime = time.time()
		fps = 1 / (cTime - pTime)
		pTime = cTime
		
		frame = cv2.flip(frame, 1)
		frame = detector.findHands(frame)
		lmList = detector.findPosition(frame)
	
		if len(lmList) != 0:
			print(lmList[4])

		cv2.putText(frame,f'FPS:{int(fps)}', (48,78), cv2.FONT_HERSHEY_COMPLEX,
			3, (255,0,0), 3)
		cv2.imshow("Image", frame)

		if cv2.waitKey(1) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()

