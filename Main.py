import glob
import cv2 as cv
import numpy as np

bin_n = 16  # Number of bins

samples = []
labels = []


def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


# Get positive samples
for filename in glob.glob("image_xe_hoi/*.png"):
    img = cv.imread(filename, 1)
    img = cv.resize(img, (128, 128))
    hist = hog(img)
    samples.append(hist)
    labels.append(1)

# Get negative samples
for filename in glob.glob("image_bg_xe/*.png"):
    img = cv.imread(filename, 1)
    hist = hog(img)
    samples.append(hist)
    labels.append(0)

# Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)


# Shuffle Samples
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]

# Create SVM classifier
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)  # cv2.ml.SVM_LINEAR
# svm.setDegree(0.0)
svm.setGamma(5.383)
# svm.setCoef0(0.0)
svm.setC(2.67)
# svm.setNu(0.0)
# svm.setP(0.0)
# svm.setClassWeights(None)

# Train
svm.train(samples, cv.ml.ROW_SAMPLE, labels)
svm.save('svm_data.dat')
# Predict
img1 = cv.imread("image(1).png", 1)
svm = cv.ml.SVM_load('svm_data.dat')
Mau=[]
hist = hog(img1)
Mau.append(hist)
Mau=np.float32(Mau)
kqnd = svm.predict(Mau)
print(int(kqnd[1]))
