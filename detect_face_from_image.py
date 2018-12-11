import cv2
import numpy as np
import time
from PIL import Image
import sys

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
def detect_faces(cascade, small_im, normal_im, scaleFactor = 1.1):

	'''
	Переводим изображение в GRAY

	Находим лица на фотографии из каскада

	Ходим на спискам лиц на фото с помощью функции	
	'''
	gray = cv2.cvtColor(small_im, cv2.COLOR_BGR2GRAY)

	faces = cascade.detectMultiScale(
		gray,
		scaleFactor = scaleFactor,
		minNeighbors = 10,
		minSize=(5, 5),
		flags = cv2.CASCADE_SCALE_IMAGE
		)
	i = 0

	for (x, y, h, w) in faces:
		cv2.rectangle(normal_im, (x, y), (x + w, y + h),RED,2)

		size = w/190

		cv2.rectangle(normal_im, (x -1, y + h), (x + w + 1, y + h + 35), RED, cv2.FILLED)
		font = cv2.FONT_HERSHEY_TRIPLEX
		cv2.putText(normal_im, 'FACE ' + str(i), (x + 6, y + h - 6 + 35), font, size, (255, 255, 255), 1)
		i += 1
	return normal_im

if '--help' in sys.argv:
	import hp
	exit(0)


try:
	path_to_image = sys.argv[1]
except Exception as e:
	print('[ERROR] Введите имя файла:\n>> python ' + str(sys.argv[0]) + ' images/image.png' )
	exit(1)
print('[INFO] Загрузка каскада...', end = '\n\n')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print('[INFO] Для выхода нажмите <{}>'.format('ESC'), end = '\n\n')
process_this_cap = 1



im = cv2.imread(path_to_image)
if (im.shape[1] > 1300):
	koef = 1300/im.shape[1]
	im  = cv2.resize(im, (0, 0), fx=koef, fy=koef)
print()
#small_im = cv2.resize(im, (0, 0), fx=0.25, fy=0.25)


frame = detect_faces(face_cascade, im, im)


cv2.imshow('DETECTED FACES', frame)

cv2.waitKey()

cv2.destroyAllWindows()




