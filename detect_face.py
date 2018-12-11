import cv2
import numpy as np
import time
from PIL import Image
import sys

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
def detect_faces(cascade, small_im, normal_im, scaleFactor = 1.609):

	'''
	Переводим изображение в GRAY

	Находим лица на фотографии из каскада

	Ходим на спискам лиц на фото с помощью функции	
	'''
	gray = cv2.cvtColor(small_im, cv2.COLOR_BGR2GRAY)

	faces = cascade.detectMultiScale(
		gray,
		scaleFactor = scaleFactor,
		minNeighbors = 3,
		minSize=(10, 10),
		flags = cv2.CASCADE_SCALE_IMAGE
		)
	i = 0

	for (x, y, h, w) in faces:
		x *= size
		y *= size
		h *= size
		w *= size
		cv2.rectangle(normal_im, (x, y), (x + w, y + h),RED,2)

		
		cv2.rectangle(normal_im, (x -1, y + h), (x + w + 1, y + h + 35), RED, cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(normal_im, 'FACE ' + str(i), (x + 6, y + h - 6 + 35), font, .7, (255, 255, 255), 1)
		i += 1
	return normal_im

if sys.argv.pop() == '--help':
	import hp
	exit(0)

print('[INFO] Welcome! Детектор лица от лучшего программиста', end = '\n')
print('       на этой планете. Чтобы получить мануал, пропишите ', end = '\n')
print('       python detect_face.py --help', end = '\n\n')
print('[INFO] Загрузка камеры...', end = '\n\n')
cap = cv2.VideoCapture(0) # Захват камеры
#time.sleep(2)
print('[INFO] Загрузка каскада...', end = '\n\n')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print('[INFO] Для выхода нажмите {}'.format('<ESC>'), end = '\n\n')

size = 5
cap.set(0,1920)
cap.set(1,1280)
while True:

	ret, im = cap.read() # считать изображение

	small_im = cv2.resize(im, (0, 0), fx=1/size, fy=1/size)

	if ret:

		frame = detect_faces(face_cascade, small_im, im)

		cv2.imshow('DETECT FACES', frame)

		key = cv2.waitKey(10)
		if key == 27:
			break
	#process_this_cap = (process_this_cap + 1)%24
cap.release()
cv2.destroyAllWindows()




