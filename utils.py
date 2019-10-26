import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator

# Set color pallete 
color = {
   'purple': '\033[95m',
   'cyan': '\033[96m',
   'darkcyan': '\033[36m',
   'blue': '\033[94m',
   'green': '\033[92m',
   'yellow': '\033[93m',
   'red': '\033[91m',
   'bold': '\033[1m',
   'underline': '\033[4m',
   'end': '\033[0m'
}

def print_head(text, color):
   ''' This function prints the heading on cmd '''
   print(f'{font_bold(font_color(color, text))}')

def font_color(code, text):
   ''' This function gives color to the input text '''
   return f'{color[code]}{text}{color["end"]}'

def font_bold(text):
   ''' This function makes the input text bold '''
   return f'{color["bold"]}{text}{color["end"]}'

def get_img_generator(config):
   '''This function augment the datset '''
   return ImageDataGenerator(rescale=1./255,
   validation_split=config.validation_split,
   rotation_range=config.rotation_range,
   width_shift_range=config.width_shift_range,
   height_shift_range=config.height_shift_range,
   zoom_range=config.zoom_range)

def load_dataset(dataset_path, subset, config):
   ''' This function loads the required dataset '''
   # get image generator
   img_datagen = get_img_generator(config)

   # for training & validation sets
   if subset == 'training' or subset == 'validation':

      datagen = img_datagen.flow_from_directory(
         f"{dataset_path}",
         target_size=(config.img_height, config.img_width),  
         batch_size=config.batch_size,
         class_mode=config.class_mode,
         subset=subset)
         
   # for testing set
   if subset == 'testing':
      datagen = img_datagen.flow_from_directory(
            f"{dataset_path}",
            target_size=(config.img_height, config.img_width),
            batch_size=config.batch_size,
            class_mode=config.class_mode,
            shuffle = False)

   return datagen

def load_train_dataset(path, config):
   ''' This function loads the training dataset '''
   train_dataset = load_dataset(path, "training", config)
   return train_dataset

def load_validation_dataset(path, config):
   ''' This function loads the testing dataset '''
   validation_dataset = load_dataset(path, "validation", config)
   return validation_dataset

def load_test_dataset(path, config):
   ''' This function loads the vallidation dataset '''
   test_dataset = load_dataset(path,"testing", config)
   return test_dataset