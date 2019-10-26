import keras
import glob
import os
import importlib
from tqdm import tqdm
import numpy as np
from cv2 import cv2
import validation
import utils

class TestClass:
    
    def __init__(self, project_name, weights_name='latest'):
        # call ValidationClass to get all the paths and config file
        validObj = validation.ValidateClass(project_name)
        self.paths = validObj.paths 
        self.config = validObj.config
        # get model name from function arg
        self.weights_name = weights_name        
        self.weights_path = self.get_weights_path(self.weights_name)
                         
    def get_weights_path(self, weight_name):
        ''' This function load the weight file (latest/custom) '''
        if weight_name == 'latest':
            utils.print_head("\nUsing latest weight file for testing model....\n", color ='darkcyan')
            # Find latest weights
            list_of_weights = glob.glob(f'{self.paths["model_weights_path"]}/*.h5') # * means all if need specific format then *.csv
            latest_weight = max(list_of_weights, key=os.path.getctime)
            return latest_weight
        else:
            utils.print_head(f"Loading weigth file {weight_name} for testing model....", color='darkcyan')
            return f"{self.paths['model_weights_path']}/{weight_name}"
    

    def load_testing_dataset(self):
        ''' This function loads the testing dataset '''
        utils.print_head('Testing dataset loaded...', 'darkcyan')
        self.test_datagen = utils.load_test_dataset(self.paths['test_dataset_path'], self.config)
    
    def load_model(self):
        utils.print_head('Model loaded for Testing...', 'darkcyan')
        self.model = keras.models.load_model(self.weights_path)
        self.classes = np.load(f"{self.paths['class_file_path']}/{self.paths['class_file_name']}")

    def evaluate_mode(self):
        
        # load model
        self.load_model()
        # load testing dataset
        self.load_testing_dataset()
        steps = self.test_datagen.samples // self.config.batch_size
        if(steps < 1):
            steps = self.test_datagen.samples
        utils.print_head('Evaluate model on testing dataset..', 'darkcyan')

        best_scores = self.model.evaluate_generator(self.test_datagen, steps=steps)
        
        utils.print_head(f"Best Weight's Accuracy: {utils.font_bold(round((best_scores[1]) * 100, 2))}%", color='darkcyan')


def test(name, weights_name='latest'):
    # call test class
    testObj = TestClass(name, weights_name)

    # load test dataset
    testObj.load_testing_dataset()
    # evaluate the model
    testObj.evaluate_mode()