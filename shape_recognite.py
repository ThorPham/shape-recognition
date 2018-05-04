import numpy as np
import keras
from matplotlib import pyplot as plt
from keras.utils import to_categorical
import os
from keras.models import load_model
import cv2


model = load_model("model_detection.h5")
index_name={0: 'Apple',1: 'Circle',2: 'Diamond',3: 'Envelope',4: 'Fish',5: 'Moon',6: 'Smiley face',7: 'Square',8: 'Triangle',9: 'Watermelon'}
def predict_shape(path_image):
	im = cv2.imread(path_image)
	im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	im_resize = cv2.resize(im_gray,(28,28),interpolation=cv2.INTER_AREA)
	im_dila = cv2.erode(im_resize,(2,2))
	im_not = cv2.bitwise_not(im_dila)/255
	y_im = im_not.reshape((1,28,28,1))
	predict = model.predict_classes(y_im)
	return index_name[int(predict)]
