import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
from cv2 import cv2
import glob
import validation
import utils

## Set function operations() to operate preprocessing

class PreprocessClass:
    
    def __init__(self, project_name):
        # call ValidationClass to get all the paths and config file
        validObj = validation.ValidateClass(project_name)
        self.paths = validObj.paths 
        self.config = validObj.config 

    def preprocess(self, image):
        return self.operations(image)
        
    def operations(self, input_image_path):
        ''' This function contains all the preprocessing operations to perform to single input image or folder of images '''
        try:
            input_image = self.load_image(input_image_path)
            input_image = self.convert_bga_to_rgb(input_image)
            tailored_image = self.resize_image(input_image)
            tailored_image = self.reshape_image(tailored_image)
            return input_image, tailored_image
        except Exception as e:
            print(f'Preprocesing Failed for {input_image_path}')
            print(e)

    def get_images_path(self, images_folder_path):
        return glob.glob(images_folder_path + '/*.*')

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def convert_bga_to_rgb(self, input_image):
        return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    def convert_to_grey_scale(self):
        pass
    
    def resize_image(self, input_image):
        return cv2.resize(input_image, (self.config.img_height, self.config.img_width))

    def reshape_image(self, input_image):
        return input_image.reshape(-1, self.config.img_height, self.config.img_width, self.config.channel)

    def convert_image_to_array(self, input_image):
        return image.img_to_array(input_image)

    def expand_image_dims(self, input_image):
        return np.expand_dims(input_image, axis=0)
    
    def remove_noise(self):
        pass

    def segmentation(self, parameter_list):
        pass

    def morphology(self, parameter_list):
        pass

    def save_images(self, image_path, image):
        cv2.imwrite(f'{image_path}', image)

