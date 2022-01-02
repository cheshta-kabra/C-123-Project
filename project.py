import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
#Setting an HTTPS Context to fetch data from OpenML 
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

x,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M'
'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=9,test_size=2500,train_size=7500)
xtrain_scaled=xtrain/255
xtest_scaled=xtest/255
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrain_scaled,ytrain)
ypred=clf.predict(xtest_scaled)
accuracy=accuracy_score(ytest,ypred)
print(accuracy)
#Starting the camera 
cap=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()
        print(ret)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        roi=gray[upper_left[1],bottom_right[1],upper_left[0],bottom_right[0]]
        im_pil=Image.fromarray(roi)
        image_bw=im_pil.convert('L')
        image_bw_resize=image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resize_invert=PIL.ImageOps.invert(image_bw_resize)
        pixel_filter=20
        minpixel=np.percentile(image_bw_resize_invert,pixel_filter)
        image_bw_resize_invert_scaled=np.clip(image_bw_resize_invert-minpixel,0,225)
        maxpixel=np.nax(image_bw_resize_invert)
        image_bw_resize_invert_scaled=np.asarray(image_bw_resize_invert_scaled)/maxpixel
        testsample=np.array(image_bw_resize_invert_scaled).reshape(1,784)
        test_pred=clf.predict(testsample)
        print('predicted class is',test_pred)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1)& 0xFF==ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()