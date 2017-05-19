#import the modules
from PIL import Image
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt


dataset = datasets.fetch_mldata("MNIST Original")

features = np.array(dataset.data,'int16')
labels = np.array(dataset.target,'int')

list_hog_fd = []
for feature in features:
    feature[feature > 0] = 1 
    fd = hog(feature.reshape((28,28)),orientations = 9, pixels_per_cell=(1,1), cells_per_block=(1,1), visualise=False)
    list_hog_fd.append(fd)
    
hog_features = np.array(list_hog_fd,'float64')

clf = LinearSVC()
clf.fit(hog_features, labels)


for i in range(0,28):
    
    im = Image.open("/Users/tunahansalih/Desktop/Dataset/test" + str(i)+ ".jpg").convert('LA')
    im = np.array(im.resize((28,28)).getdata())[:,0].reshape((28,28))
    im[im<128]=0
    im[im>=128]=1
    imArr, im_hog= hog(im,orientations = 9, pixels_per_cell=(1,1), cells_per_block=(1,1), visualise=True)
    hog_image_rescaled = exposure.rescale_intensity(im_hog, in_range=(0, 0.02))
    plt.figure()
    plt.imshow(im_hog, cmap='binary')
    print(clf.predict(imArr.reshape(1,-1)))
