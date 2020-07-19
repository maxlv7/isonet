import h5py
import time
import numpy as np
import cv2

f = h5py.File("val_rgb.h5",'r')
f1 = h5py.File("val_rgb_label.h5",'r')

# for y in range(1,10000):
#     print(np.array(f1.get(str(y))))
#     time.sleep(0.001)
for x in range(1,1000):

    p1 = np.array(f.get(str(x)))*255
    p1 = p1.astype(np.uint8)
    img = p1.transpose((1,2,0))[:,:,::-1]
    print(img.shape)
    cv2.imshow("t",img)
    cv2.waitKey(0)