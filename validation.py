import os
import utils
import importlib

class ValidateClass:

	def __init__(self, project_name):
		# save project name
		self.project_name = project_name
		# check the listed directories
		self.paths = {
				'project_path' : f'projects/{self.project_name}',
				'dataset_path' : f'projects/{self.project_name}/dataset',
				'train_dataset_path' : f'projects/{self.project_name}/dataset/train',
				'test_dataset_path' :  f'projects/{self.project_name}/dataset/test',
				'model_path' : f'projects/{self.project_name}/model/saved models',
				'model_logs_path' : f'projects/{self.project_name}/model/tblogs',
				'model_weights_path' : f'projects/{self.project_name}/model/weights',
				'class_file_path' : f'projects/{self.project_name}/model',
				'class_file_name' : f'classes.npy',
				'predict_folder' : f'projects/{self.project_name}/predict',
				'predict_input_path' : f'projects/{self.project_name}/predict/input',
        	    'predict_output_path' : f'projects/{self.project_name}/predict/output',
				'images_path' : f'projects/{self.project_name}/dataset/images'
				}
		# load config file
		self.config = self.parse_config()


	def get_validated_paths(self):
		# validate config file
		self.validate_config()
		# validate common directories
		self.validate_directories(self.paths)
		# validate dataset
		self.validate_dataset()
		utils.print_head('Validated folders structure & dataset...', 'darkcyan')

	def validate_config(self):
		''' This function validates the config.py file '''
		config_path = f"{self.paths['project_path']}/config.py"
		if self.check_directory_existance(config_path) == False:
			utils.print_head(f"{config_path} file not found !!", color='red')
			exit()
	
	def parse_config(self):
		''' This function loads the config.py file '''
		self.config = importlib.import_module(f"projects.{self.project_name}.config")
		return self.config

	def validate_directories(self, paths):
		''' This function validates the common directories '''
		for key, value in self.paths.items(): 
			# if directory not found
			if self.check_directory_existance(value) == False:
				# create dir
				if os.mkdir(value):
					utils.print_head(f"{key} directory created !!", color='green')

	def validate_dataset(self):
		'''' This function validates the test & train dataset ''' 
		if (len(os.listdir(self.paths['train_dataset_path'])) == 0) or (len(os.listdir(self.paths['test_dataset_path'])) == 0) :
			utils.print_head(f"Train or Test dataset not found !\nKindly check {self.paths['train_dataset_path']} and {self.paths['test_dataset_path']}.", color='red')
			exit()

	def validate_presence_of_items_in_folder(self, path):
		'''' This function validates the non-emptyness of the folder ''' 
		if (len(os.listdir(path)) == 0) or (len(os.listdir(path)) == 0) :
			utils.print_head(f"Data not found !\nKindly check {path}.", color='red')
			exit()

	def check_directory_existance(self, folder_path):
		''' This function checks the existance of a directory'''
		return os.path.exists(folder_path)
