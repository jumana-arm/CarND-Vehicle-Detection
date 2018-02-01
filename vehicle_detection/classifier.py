import numpy as np
import pickle
import cv2
import glob
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from moviepy.editor import VideoFileClip
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from vehicle_detection import feature_extraction as feature

def train_hog_features(cars, noncars, hog_params, logger=None):
    """
    Get hog features of images and train using SVC classifier
    :cars: list of car images
    :noncars: list of noncar images
    :hogparams: dictionary of necessary hog parameters; colorspace, orientation, pixel per cell, cell per block, hog channel
    
    :returns:
    svc: trained classifier model
    hog_svc_data: training results: accuracy, predcition time etc
    """
    cspace = hog_params['cspace']
    orient = hog_params['orient']
    pix_per_cell = hog_params['pix_per_cell']
    cell_per_block = hog_params['cell_per_block']
    hog_channel = hog_params['hog_channel']

    hog_svc_data = {}
    t=time.time()
    car_features = feature.extract_hog_features(cars, cspace=cspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    noncar_features = feature.extract_hog_features(noncars, cspace=cspace, orient=orient, 
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)
    t2 = time.time()
    feature_extraction_time = round(t2-t, 2) 
    hog_svc_data['feature_extraction_time'] = feature_extraction_time
    print(feature_extraction_time, 'Seconds to extract HOG features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    training_time = round(t2-t, 2)
    hog_svc_data['training_time'] = training_time
    print(training_time, 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    hog_svc_data['accuracy'] = accuracy
    print('Test Accuracy of SVC = ', accuracy)
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    prediction_time = round(t2-t, 5) 
    hog_svc_data['prediction_time'] = prediction_time
    print(prediction_time, 'Seconds to predict', n_predict,'labels with SVC')
    return svc, hog_svc_data