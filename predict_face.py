'''
TO DO:

+1) нахождения картинок моего личика 
+2) фунция, которая будет вырезать из фотографии мое личико
+3) функция предсказания моего личика
+4) функции, которые будут выводить красиво текст
+5) функция, которая рисует квадратик
 6) Считывание типов лиц с файла
 7)
+++last) Молиться, что все будет работать как надо

'''

import cv2
import numpy as np
import time
from PIL import Image
import sys
import os
import pickle

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
WHITE = (255,255,255)

types = [None, 'Lev', 'Volodya','new','new','new','new','new','new','new']

def prepare(path):
	'''
	Хождение по photos и поиск всех изображений.
	Подготовка данных, которые нашли, для обработки
	'''
	faces = []  # лицо
	labels = [] # тип 1: Lev Lymarenko, 2: ...
	
	for dirname in os.listdir(path):
		if dirname[0] == '_':
			continue
		label = int(dirname)
		for name_image in os.listdir(path + '/' + dirname):
			print('[INFO] Обрабока ' + path + '/' + dirname + '/' + name_image)
			im = cv2.imread(path + '/' + dirname + '/' + name_image)
			faces_detected = detect_one_face(face_cascade, im)      # ищем на картинке лица
			if not faces_detected is None:               # если нашли лицо, кидаем в faces/labeles
				faces.append(faces_detected[0][0])
				labels.append(label)
		print()
	return faces, labels

'''
def detect_one_face(cascade, colored_im, scaleFactor = 1.3):

	gray = cv2.cvtColor(colored_im, cv2.COLOR_BGR2GRAY) # to gray
	faces = cascade.detectMultiScale(
		gray,
		scaleFactor = scaleFactor,
		minNeighbors = 10)             # Главная функция, которая находит все лица на картинке по каскаду
	if len(faces) > 0:
		(x, y, w, h) = faces[0]
		if w > 0 and h > 0:
			return gray[y:(y + w), x:(x + h)], faces[0]      # возврат (картинка, прямоугольник)
		else:
			return None, None	
	else:
		return None, None
'''

def detect_one_face(cascade, colored_im, scaleFactor = 1.3, minNeighbors = 5):
	"""
	Нахождение лиц на фотографии.
	Возвращает массив, в котором находятся рожи и их координаты.
	"""
	gray = cv2.cvtColor(colored_im, cv2.COLOR_BGR2GRAY) # to gray


	# Главная функция, которая находит все лица на картинке по каскаду. Работает
	# ИСКЛЮЧИТЕЛЬНО на магии и подборе значений scaleFactor и minNeighbors
	faces = cascade.detectMultiScale(gray, scaleFactor = scaleFactor, minNeighbors = minNeighbors)              
	ans = []
	if len(faces) > 0:
		for i in faces:
			(x, y, w, h) = i
			ans.append((gray[y:(y + w), x:(x + h)], i))
		return ans      # возврат [(картинка, прямоугольник),(картинка, прямоугольник) ... ] 	
	else:
		return None

def draw_specifications_with_labels(colored_im, name, conf, rect):
	x,y,w,h = rect
	color = GREEN
	# Прямоугольник
	cv2.rectangle(colored_im, (x, y), (x + w, y + h), color, 2)

	# Текст
	font = cv2.FONT_HERSHEY_DUPLEX
	cv2.rectangle(colored_im, (x - 1, y + h), (x + w + 1, y + h + 35), color, cv2.FILLED)
	cv2.putText(colored_im, name + ' [{}]'.format(100 - float(conf) ), (x + 6, y + h - 6 + 35), font, 1, WHITE, 1)
	return colored_im

def draw_specifications_without_labels(colored_im, rect):
	x,y,w,h = rect
	color = RED
	# Прямоугольник
	cv2.rectangle(colored_im, (x, y), (x + w, y + h), color, 2)

	# Текст
	font = cv2.FONT_HERSHEY_DUPLEX
	cv2.rectangle(colored_im, (x - 1, y + h), (x + w + 1, y + h + 35), color, cv2.FILLED)
	cv2.putText(colored_im, 'UNKNOW PERSON', (x + 6, y + h - 6 + 35), font, 0.8, WHITE, 1)
	return colored_im
	
def predict_face(colored_im):
	'''
	Говорит с какой вероятностью
	и к какому лейблу относится
	лица на картинке. Пишет на картинке данные
	и возвращает ее.
	'''

	faces_detected = detect_one_face(face_cascade, colored_im, scaleFactor = scaleFactor, minNeighbors = minNeighbors)
	if faces_detected is None:
		return colored_im
	for face, rect in faces_detected:
		if face is None:
			continue           # непонятный костыль, который почему-то работает
			#return colored_im
		label, conf = face_recognizer.predict(face)
		if conf < 50:
			name = Labels[label]
			colored_im = draw_specifications_with_labels(colored_im,name, str(round(conf,0)), rect)
		else:
			colored_im = draw_specifications_without_labels(colored_im, rect)
	return colored_im

a = sys.argv[1:]
scaleFactor = 1.3
minNeighbors = 5
for i,x in enumerate(a):
	if x == '-s':
		scaleFactor = float(a[i + 1])
	elif x == '-n':
		minNeighbors = int(a[i + 1])



'''
Самая важная часть, из-за которой все работает.
происходит загрузка каскада, который я нашел на гитхабе. 
Затем данные, которые мы фотали редачатся и обрезаются.
Затем проиходит тренеровка данных (МАГИЯ!!!!)
Затем загружаем камеру 
'''
print('[INFO] Загрузка каскада...', end = '\n\n')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

print('[INFO] Изменение данных...', end = '\n\n')
#faces, labels = prepare('photos')

print('[INFO] Загрузка изображений...', end = '\n\n')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.train(faces, np.array(labels))
face_recognizer.read("train/face-trainner.yml")

with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	Labels = {v:k for k,v in og_labels.items()}

print('[INFO] Загрузка камеры...', end = '\n\n')
cap = cv2.VideoCapture(0) # Захват камеры
time.sleep(2)

print('[INFO] Для выхода нажмите {}'.format('<ESC>'), end = '\n\n')

while True:
	ret, im = cap.read() # считать изображение
	#im = cv2.imread('face_two.jpg')
	if ret:
		frame = predict_face(im) #Главная функция по "предсказанию" лица
		cv2.imshow('PREDICT FACES', frame)

		key = cv2.waitKey(10)
		if key == 27:
			break
cap.release()
cv2.destroyAllWindows()