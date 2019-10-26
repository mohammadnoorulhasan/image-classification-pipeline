import os
import validation
import utils
import keras
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop, SGD, Adam
from keras import regularizers
from keras.regularizers import l1

class ModelClass :

    ''' This class defines the model for the training '''
    def __init__(self, project_name):
        # call ValidationClass to get all the paths and config file
        validObj = validation.ValidateClass(project_name)
        self.paths = validObj.paths 
        self.config = validObj.config

    def get_selected_model(self, model_to_trigger):
        ''' This function triggers the selected model in config.py '''
        available_models = { 
                    'vgg16' : self.vgg16, 
                    'fer' : self.fer,
                    'customized_model_name' : self.customized_model_name
        } 
        selected_model = available_models.get(model_to_trigger, lambda : utils.print_head("Invalid Model Selection!!", color='red'))
        return selected_model()

    def train_model(self):
        ''' This function enables to compile the selected model for training '''
    
        # get user selected model
        self.model = self.get_selected_model(self.config.selected_model)
        
        utils.print_head(f"Selected Model : {self.config.selected_model}, Epochs : {self.config.epochs}, Batch Size : {self.config.batch_size}, Learning Rate : {self.config.lr_rate}, Optimizer : {self.config.optimizer}", color='purple')
        
        self.model.compile(loss = self.config.loss,
                            optimizer = self.get_optimizer(),
                            metrics=["accuracy"])
        # save the model
        self.model.save(f"{self.paths['model_path']}/{self.config.model_name}--{self.config.selected_model}--{self.config.epochs}_e--{self.config.lr_rate}_lr--{self.config.batch_size}_batch--{self.config.optimizer}.h5")
        return self.model


    ### DEFINED MODELS ###

    def customized_model_name(self):
        ''' This function defines the customized model for training purpose '''
        # write model body only
        # no need to write compile and fit function
        # model is getting compiled in train_model()
        # model is getting fit in train.py
        pass

    def fer(self):
        ''' This function can be used for recoginizing the facial emotions '''
        # utils.print_head(f"Running FER2013 Model..", color='green')
        # Create the model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(self.config.img_width, self.config.img_height, self.config.channel)))
        # model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(7, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        # # model.add(BatchNormalization())
        # model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        # model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Activation("softmax"))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(2, activation='softmax'))
        return model

    def vgg16(self):
        ''' This function loads the VGG16 model '''
        # utils.print_head(f"Running VGG16 Model...", color='green')
        model = applications.VGG16(weights = "imagenet",
                                    include_top=False, 
                                    input_shape = (self.config.img_width, self.config.img_height, self.config.channel))

        # Freeze the layers which you don't want to train. Here I am freezing the first 8 layers.
        for layer in model.layers[:8]:
            layer.trainable = False

        #Adding custom Layers 
        out = model.output
        out = Flatten()(out)
        out = Dense(2048, activation="relu")(out)
        out = Dropout(rate=0.5)(out)
        out = Dense(1024, activation="relu")(out)
        predictions = Dense(self.config.nb_classes, activation="softmax")(out)

        # creating final model
        model_final = Model(input = model.input, output = predictions)
        
        return model_final

    def get_optimizer(self):
        ''' This function sets the optimizer from config file '''
        self.optimizer = self.config.optimizer
        self.options = self.config.options

        if(self.options['name'].lower() == 'adam'):
            lr = self.options['lr']
            #beta_1 = self.options['beta_1']
            #beta_2 = self.options['beta_2']
            #decay = self.options['decay']
            optimizer = optimizers.adam(lr)
            #optimizer = optimizers.adam(lr, beta_1, beta_2, decay)

        elif(self.options['name'].lower() == 'adadelta'):
            lr = self.options['lr']
            rho = self.options['rho']
            epsilon = self.options['epsilon']
            decay = self.options['decay']

            optimizer = optimizers.adadelta(lr, rho, epsilon, decay)

        elif(self.options['name'].lower() == 'sgd'):
            lr = self.options['lr']
            momentum = self.options['momentum']
            decay = self.options['decay']
            nesterov = self.options['nesterov']

            optimizer = optimizers.sgd(lr, momentum, decay, nesterov)
       
        elif(self.options['name'].lower() == 'rmsprop'):
            lr = self.options['lr']
            rho = self.options['rho']
            epsilon = self.options['epsilon']
            decay = self.options['decay']

            optimizer = optimizers.rmsprop(lr, rho, epsilon, decay)

        return optimizer
