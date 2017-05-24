#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 00:12:41 2017

@author: tunahansalih
"""

import glob
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


#Data classes
image_dataset_file = "/Users/tunahansalih/Desktop/extracted_images"
character_classes = ['-',
                     '+',
                     '0',
                     '1',
                     '2',
                     '3',
                     '4',
                     '5',
                     '6',
                     '7',
                     '8',
                     '9',
                     'div',
                     'times']


classes = []
features = []


#Read Images
for char_class in character_classes:
    i = 0
    for filename in glob.glob(image_dataset_file + "/" + char_class + "/*.jpg" ):
        i = i + 1
        print(str(i))
        if i > 3000:
            break
        im = imread(filename, "L")
        im = imresize(im,(28,28))
        classes.append(char_class)
        features.append(im)

#Create HOG of images
i=0;
features_hog = []
for feature in features:
    print(i)
    i=i+1
    fd = hog(feature,orientations = 9, pixels_per_cell=(2,2), cells_per_block=(1,1), visualise=False)
    features_hog.append(fd)


#Create test and train data
features_hog_train, features_hog_test, classes_train, classes_test = train_test_split(features_hog, classes, test_size=0.33, random_state= 42,stratify=classes)


#Multivariate SVN
model = LinearSVC()
model.fit(features_hog_train, classes_train)

joblib.dump(model,"/Users/tunahansalih/Work/DigitRecognition/model.joblib.pkl",compress=9)

print(model.score(features_hog_test,classes_test))
#
#for i in range(0,28):
#
#    im = imread("/Users/tunahansalih/Desktop/Dataset/test" + str(i)+ ".jpg","L")
#    im = imresize(im,(28,28))
#    imArr, im_hog= hog(im,orientations = 9, pixels_per_cell=(2,2), cells_per_block=(1,1), visualise=True)
#    plt.figure()
#    plt.imshow(im_hog)
#    print(model.predict(imArr.reshape(1,-1)))