import glob
import os
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from keras import applications, optimizers
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
import utils
import validation
import model
import prepare_dataset

class TrainClass:
    ''' This class trains the model '''
    def __init__(self, project_name , model_name = 'new'):
        # prepare dataset
        prepare_dataset.prepare_dataset(project_name)
        # call ValidationClass
        validObj = validation.ValidateClass(project_name)
        # get all the paths and config file
        validObj.get_validated_paths()
        # get validated paths & config file
        self.paths = validObj.paths 
        self.config = validObj.config
        # set id to current run 
        self.run_id = self.get_run_id() 
        # get model name from function arg
        self.model_name = model_name
        # create object of model class to operate model.py 
        self.modelObj = model.ModelClass(project_name)       
        
    def load_training_dataset(self):
        ''' This function loads the training dataset '''
        utils.print_head('TRAINING dataset loaded...', 'darkcyan')
        self.train_datagen = utils.load_train_dataset(self.paths['train_dataset_path'], self.config)
        self.val_datagen = utils.load_validation_dataset(self.paths['train_dataset_path'], self.config)
        # Saving classes to use for predictions
        classes = np.array(list(self.train_datagen.class_indices.keys()))
        np.save(f"{self.paths['class_file_path']}/{self.paths['class_file_name']}", classes)

    def load_testing_dataset(self):
        ''' This function loads the testing dataset '''
        utils.print_head('TESTING dataset loaded...', 'darkcyan')
        self.test_datagen = utils.load_test_dataset(self.paths['test_dataset_path'], self.config)
    
    def get_run_id(self):
        ''' This function get the run id to name the weight files '''
        return f"{self.config.model_name}-{self.config.epochs}e{self.config.lr_rate}lr-{time.time()}"

    def resume_last_model(self):
        ''' This function load & resumes the last trainned model'''
        utils.print_head(f"Resuming last trained model....", color='darkcyan')
        #Find latest weights
        list_of_model = glob.glob(f"{self.paths['model_path']}/*.h5") # * means all if need specific format then *.csv
        try:
            latest_model = max(list_of_model, key=os.path.getctime)
            self.model = load_model(f"{latest_model}")
        except:
            utils.print_head(f"No pre-trained model found..!!", color='red')
            utils.print_head(f"Training model from initial epoch......", color='darkcyan')
            self.model = self.modelObj.train_model()
            #self.model = self.get_model_vgg()

    def get_last_model(self, model_name):
        if self.model_name == 'resume':
            self.resume_last_model()
        # try to load the demanded model
        else:
            # check the trained model existance
            check = os.path.exists(f"{model_name}")
            if check == True:
                utils.print_head(f"Resuming given model : {model_name}", color='darkcyan')
                # load the model if model exists
                self.model = load_model(f"{model_name}")
            else:
                utils.print_head(f"Model not found !!", color='red')
                utils.print_head(f"Training model from initial epoch......", color='darkcyan')
                self.resume_last_model()

    def get_best_dynamic_weights_name(self):
        ''' This function names the best weight '''
        utils.print_head(f"Saving Best Weight  :  {self.paths['model_weights_path']}/Best-{self.run_id}.h5", color='darkcyan')
        return f"{self.paths['model_weights_path']}/Best-{self.run_id}.h5"

    def get_dynamic_weights_name(self):
        ''' This function names the weight '''
        return f"{self.paths['model_weights_path']}/{self.run_id}.h5"

    def get_model_callbacks(self):
        ''' This function saves the callbacks (weight file)'''
        # Save all the weights and checkpoints
        checkpoint = ModelCheckpoint(self.get_dynamic_weights_name(),
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=False,
                                    period=1)

        # Save the best model checkpoints[best_checkpoint, checkpoint, tboard]
        best_checkpoint = ModelCheckpoint(self.get_best_dynamic_weights_name(),
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='auto',
                                    period=1)

        tboard = TensorBoard(log_dir=f"{self.paths['model_logs_path']}/{self.run_id}",
                batch_size=self.config.batch_size,
                write_images=True)
    
        return [best_checkpoint, checkpoint, tboard]


    def train_model(self):
        ''' This function trains the model '''

        utils.print_head('Model Training Initiated...', 'darkcyan')
        if self.model_name == 'new':
            self.model = self.modelObj.train_model()
        else:
            # load the model to resume training
            self.get_last_model(f"{self.paths['model_path']}/{self.model_name}")
        
        # fit the model
        self.model.fit_generator(self.train_datagen,
                                 steps_per_epoch=self.train_datagen.samples // self.config.batch_size,
                                 validation_data=self.val_datagen,
                                 validation_steps=self.val_datagen.samples // self.config.batch_size,
                                 epochs=self.config.epochs,
                                 callbacks = self.get_model_callbacks())
        # calculate the score                                 
        latest_scores = tqdm(self.model.evaluate_generator(self.train_datagen,
                                 steps=self.train_datagen.samples // self.config.batch_size))
        print("Score : ",latest_scores)


    def evaluate_mode(self):
        ''' This function evaluates the traininng model '''
        utils.print_head('Evaluating model.....', 'darkcyan')
        # loading test dataset if present
        self.load_testing_dataset()
        
        # counting steps
        steps = self.train_datagen.samples // self.config.batch_size
        if(steps < 1):
            steps = self.train_datagen.samples

        latest_scores = self.model.evaluate_generator(self.test_datagen,
                                 steps=steps)

        self.model.load_weights(f"{self.get_dynamic_weights_name()}")
        
        best_scores = self.model.evaluate_generator(self.test_datagen, steps=steps)

        utils.print_head(f"Latest Weight's Accuracy: {utils.font_bold(round(latest_scores[1] * 100, 2))}%", color='green')
        print(f"Best Weight's Accuracy: {utils.font_bold(round(best_scores[1] * 100, 2))}%")


def train(project_name, model_name = 'new'):
    # This function trains the model
    trainObj = TrainClass(project_name, model_name)

    trainObj.load_training_dataset()
    trainObj.train_model()
    trainObj.evaluate_mode()