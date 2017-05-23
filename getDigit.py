#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:31:23 2017

@author: tunahansalih
"""

from sklearn.externals import joblib
from scipy.misc import imread, imresize
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt

dump_file_address = "/Users/tunahansalih/Work/DigitRecognition/model.joblib.pkl"

model = joblib.load(dump_file_address)

im = imread("/Users/tunahansalih/Desktop/Dataset/test9.jpg","L")
im = imresize(im,(28,28))
imArr, im_hog= hog(im,orientations = 9, pixels_per_cell=(2,2), cells_per_block=(1,1), visualise=True)
plt.figure()
plt.imshow(im)
plt.figure()
plt.imshow(im_hog)
model.predict(imArr.reshape(1,-1))
