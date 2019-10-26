''' DATASET PARAMETERS '''

# training and testing dataset is needed, as data for validation is spliting during traning
train_data_path = "train"
test_data_path = "test"

# select the dataset type then given options : 'csv' /'json' / 'unspilt_data', 'None' (for the saved image dataset)
dataset_type = None

# split ratio to split dataset into train and test
train_test_valid_ratio = (0.8, 0.1, 0.1)

''' DATA AUGMENTATION '''

# randomly rotate images in the range (degrees 0 to 180)
rotation_range = 0

# randomly shift images horizontally/vertically (fraction of total width/height)
width_shift_range = 0
height_shift_range = 0

# randomly flip images
horizontal_flip = False
vertical_flip = False

# Random zoom
zoom_range = 0

''' MODEL PARAMETERS '''

model_name = 'Shapes-Classification'

# Image params
img_height = 48
img_width = 48
channel = 3

# Class selection
class_mode = 'categorical'
nb_classes = 2
validation_split = 0.01

# Training params
epochs = 2
lr_rate = 0.001
batch_size = 32
loss = "categorical_crossentropy"

# Optimizers
optimizer = 'adadelta'

# [ADAM Config]
# options = {
#     'name' : 'adam',
#     'lr': 0.0001,
# }

## [Adadelta Config]
options = {
  'name' : 'adadelta',
  'lr': 0.01,
  'rho': 0.95,
  'epsilon': None, 
  'decay': 0.0
}

selected_model =  'vgg16'
## [RMSProp Config]
# options = {
#   'name' = 'rmsprop',
#    'lr': 0,
#    'rho': 0.9,
#    'epsilon': None, 
#    'decay':0.0
# }

## [SGD Config]
# options = {
#   'name' : 'sgd',
#   'lr' : 0,
#   'momentum' : None,
#   'decay' : None,
#   'nesterov' : None
# }