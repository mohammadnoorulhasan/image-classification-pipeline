import importlib
import glob
import os
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from cv2 import cv2
from PIL import Image as pil_image
from matplotlib import pyplot as plt

import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
import validation
import utils
import preprocess

class PredictClass:

    def __init__(self, project_name, weights_name='latest'):
        # call ValidationClass to get all the paths and config file
        validObj = validation.ValidateClass(project_name)
        self.paths = validObj.paths 
        self.config = validObj.config
        # validate prediction input
        validObj.validate_presence_of_items_in_folder(self.paths['predict_input_path'])
        # preprocessing 
        self.preprocessObj = preprocess.PreprocessClass(project_name)
        # save weights name
        self.weights_path = self.get_weights_path(weights_name)

    def get_weights_path(self, weight_name):
        if weight_name == 'latest':
            # Find latest weight
            list_of_weights = glob.glob(f'{self.paths["model_weights_path"]}/*.h5') # * means all if need specific format then *.csv
            latest_weight = max(list_of_weights, key=os.path.getctime)
            return latest_weight
        else:
            return f"{self.paths['model_weights_path']}/{weight_name}"

    def load_model(self):
        utils.print_head('Load model for prediction...', color='darkcyan')
        self.model = keras.models.load_model(self.weights_path)
        self.classes = np.load(f"{self.paths['class_file_path']}/{self.paths['class_file_name']}")

    def draw_class(self, image, class_name):
        height, width = image.shape[:2]
        bordered = cv2.copyMakeBorder(image, top=0, bottom=50, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[245,222,179])
        cv2.putText(bordered, class_name, (2, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        return bordered

    def predict_class(self):
        utils.print_head('Prediction Initiated...', color='darkcyan')
        images_path = glob.glob(self.paths['predict_input_path'] + '/*.*')

        for image_path in tqdm(images_path):
            try:
                image, tailored_image = self.preprocessObj.operations(image_path)
                pred = self.model.predict(tailored_image)
                result = np.where(pred == np.amax(pred))                
                resultidx = result[1][0]
                class_name = self.classes[resultidx]
                img_with_class = self.draw_class(image, class_name)

                plt.imsave(f'{self.paths["predict_output_path"]}/{class_name}-{os.path.basename(image_path)}', img_with_class)

            except Exception as e:
                utils.print_head(f'Prediction failed for {image_path}', color='red')
                print(e)


def predict(project_name, weight_name='latest'):
    predObj = PredictClass(project_name, weight_name)

    predObj.load_model()
    predObj.predict_class()