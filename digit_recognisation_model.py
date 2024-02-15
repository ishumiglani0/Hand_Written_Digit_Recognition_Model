import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from autils import *
# %matplotlib inline

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def get_mnist_data():
    # load dataset
    (X_train, y_train),(X_test,y_test) = mnist.load_data()
    return (X_train, y_train,X_test,y_test)

def train_model(X_train, y_train,X_test,y_test):
    # class myCallback(tf.keras.callbacks.Callback):
    #     def on_epoch_end(self,epoch,logs={}):
    #         print(logs)
    #         if(logs.get('accuracy')>0.99):
    #             print("\nREACHED 99% ACCURACY SO CANCELLING TRAINING")
    #             self.model.stop_training = True

    #     callbacks = myCallback()
    X_train = np.array(X_train).reshape(len(X_train),28*28)
    X_test = np.array(X_test).reshape(len(X_test),28*28)
    y_train = np.array(y_train).reshape(len(y_train),1)
    y_test = np.array(y_test).reshape(len(y_test),1)
    model = Sequential([
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu'),
        Dense(units=10, activation='softmax')  # Output layer with softmax for multi-class classification
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    # model.compile(loss=tf.compat.v1.losses.sparse_softmax_cross_entropy(), optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, 28 * 28))
    print(model.summary())

    model.fit(X_train, y_train, epochs=100, validation_data=(X_train, y_train))
    return model

def predict(model, img):
    # Flatten the input image
    img_flat = img.flatten()
    imgs = np.array([img_flat])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)


startInference = False
def ifClicked(event,x,y,flags,params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

threshold = 130
def on_threshold(x):
    global threshold
    threshold=x

def start_cv(model):
    global threshold
    cap = cv2.VideoCapture(0)
    frame = cv2.namedWindow('background')
    cv2.setMouseCallback('background',ifClicked)
    cv2.createTrackbar('threshold','background',150,255,on_threshold)
    background = np.zeros((480,640),np.uint8)
    frameCount =0 

    while True:
        ret,frame = cap.read()
        if(startInference):
            frameCount +=1
            frame[0:480,0:80] = 0
            frame[0:480,560:680] = 0
            grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            _,thr = cv2.threshold(grayFrame,threshold,255,cv2.THRESH_BINARY_INV)

            resizeFrame = thr[240-75:240+75,320-75:320+75]
            background[240-75:240+75,320-75:320+75] = resizeFrame

            iconImg = cv2.resize(resizeFrame,(28,28))

            res = predict(model,iconImg)

            if frameCount ==5:
                background[0:480,0:80] =0 
                frameCount =0
            
            cv2.putText(background,res,(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
            cv2.rectangle(background,(320-80,240-80),(320+80,240+80),(255,255,255),thickness=3)

            cv2.imshow('background',background)
        else : 
            cv2.imshow('background',frame)
        

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release
    cv2.destroyAllWindows()
    
def main():
    model=None
    try:
        print('loading model')
        model=tf.keras.models.load_model('ishu.sav')
        print('loaded saved model')
        print(model.summary())
    except:
        print("getting mnist data")
        # load dataset
        (X_train, y_train,X_test,y_test) = get_mnist_data()
        print("training model...")
        model = train_model(X_train, y_train,X_test,y_test)
        print("saving model...")
        model.save('ishu.sav')

    print('starting cv...')
    start_cv(model)

if __name__== '__main__':
    main()