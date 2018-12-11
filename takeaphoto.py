import os
import cv2
import time
import sys

name = int(open('current_number.txt').read())
flag = False
for i,x in enumerate(sys.argv):
	if x == '-n':
		name = int(sys.argv[i + 1])
		flag = True
try:
	name = sys.argv[1]
except:
	print('Ошибка! Введите имя человека.')
	exit(1)

path ='photos/' + str(name)
if not os.path.exists(path):   # создаем директорию
	os.makedirs(path)


print('[INFO] Загрузка камеры...', end = '\n\n')
cap = cv2.VideoCapture(0)     # Захват изображения
time.sleep(.5)

number_of_photo = 0
for filename in os.listdir(path):
	number = ''
	for i in filename:
		if i.isdigit():
			number += i
	if int(number) > number_of_photo:
		number_of_photo = int(number)

print('[INFO] Для снимка нажмите {}. Для выхода нажмите {}'.format('<ПРОБЕЛ>', '<ESC>'), end = '\n\n')
try:
	i = 0
	while True:
		ret, im = cap.read()

		if ret:
			cv2.imshow('[CAMERA]', im)
			key = cv2.waitKey(10)
			if key == 27:
				exit(1)
			if key == ord(' '):
				number_of_photo += 1
				cv2.imwrite(path + '/face' + str(number_of_photo) + '.jpg', im)
				i += 1
				

except:
	print('[INFO] Завершение работы.', end = '\n\n')
	print('[INFO] {} снимка(ов) сохранены в ./'.format(i) + path + '/', end = '\n\n')