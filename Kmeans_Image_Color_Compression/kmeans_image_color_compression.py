import numpy as np
import cv2
IMG_PATH = './R3B03380.jpg'
K_INDEX = [4, 8, 16]

img, res, criteria = cv2.imread(IMG_PATH), list(), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001)
imgshape = [img.shape[0] // 2, img.shape[1] // 2, img.shape[2]]
img = np.float32(cv2.resize(img, (imgshape[1], imgshape[0])).reshape((-1, 3)))

for k in K_INDEX:
    tmp = cv2.kmeans(img, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    res.append(np.uint8(tmp[2])[tmp[1].flatten()].reshape(imgshape))

for i, r in enumerate(res):
    cv2.imshow('Clustering with K = {0}'.format(K_INDEX[i]), r)
    cv2.waitKey(0)
cv2.destroyAllWindows()