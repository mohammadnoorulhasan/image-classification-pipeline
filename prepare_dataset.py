import urllib.request
from csv import reader
import json
import requests
import uuid
from tqdm import tqdm
import os, os.path
import shutil
import split_folders
import importlib
from cv2 import cv2
import numpy as np
import pandas as pd
import math
import utils
import validation

def prepare_dataset(project_name):
    # call ValidationClass to get all the paths and config file
    validObj = validation.ValidateClass(project_name)
    paths = validObj.paths
    config = validObj.config
    flag = None
    # dataset type 
    dataset_type = config.dataset_type

    ## Set paths
    # final output dataset
    dataset_path = paths['dataset_path']
    # loaded images into a single folder
    images_folder_path = paths['images_path']

    train_data_path  = f'{dataset_path}/{config.train_data_path}'
    test_data_path  = f'{dataset_path}/{config.test_data_path}'

    split_input_path = images_folder_path
    split_output_path = dataset_path

    # check if dataset already converted into images
    flag = check_the_dataset_status(train_data_path, test_data_path)

    # load and prepare dataset
    if((dataset_type == 'unspilt_data') & (flag == False)):
        # split images into train and test 
        split_dataset_into_train_test(split_input_path, split_output_path, config, paths)
    
    elif((dataset_type == 'json') & (flag == False)):
        prepare_json_dataset(project_name, config, dataset_path, paths)
        # split dataset into train and test
        split_dataset_into_train_test(split_input_path, split_output_path, config, paths)
    
    elif((dataset_type == 'csv_urls') & (flag == False)):
        # get images into a single folder from csv
        prepare_csv_url_dataset(project_name, config, dataset_path, paths)
        # split dataset into train and test
        split_dataset_into_train_test(split_input_path, split_output_path, config, paths)

    elif((dataset_type == 'csv_pixels') & (flag == False)):
        # get images into a single folder from csv
        prepare_csv_pixel_dataset(project_name, config, dataset_path, paths)
        # split dataset into train and test
        split_dataset_into_train_test(split_input_path, split_output_path, config, paths)
    
    else:
        return

def check_the_dataset_status(train_data_path, test_data_path):
    ''' This function checks the existance of train and test dataset ''' 
    if (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
        if ([len(os.listdir(train_data_path)) != 0] or [len(os.listdir(test_data_path)) != 0]):
            flag = True
    else:
        flag = False
    return flag

def split_dataset_into_train_test(split_input_path, split_output_path, config, paths):
    
    split_folders.ratio(split_input_path, split_output_path, seed=1337, ratio = config.train_test_valid_ratio)

    # remove images folder which of no use after split
    try:
        for dir in os.listdir(f"{split_output_path}/val/"):
            for image in os.listdir(f"{split_output_path}/val/{dir}/"):
                shutil.move(f"{split_output_path}/val/{dir}/{image}", f"{paths['input_path']}", copy_function = shutil.copytree)
    except:
        pass
    try:
        shutil.rmtree(split_input_path)
        shutil.rmtree(f"{split_output_path}/val")
    except:
        try:
            os.rmdir(split_input_path)
            os.rmdir(f"{split_output_path}/val")
        except:
            pass

def prepare_json_dataset(project_name, config, dataset_path, paths):
    ''' This function loads json dataset, extract images and labels '''
    utils.print_head("Preparing dataset from provided .json file... ", color='darkcyan')
    # create folder to store images if not exists
    if os.path.exists(paths['images_path']) == False:
        os.mkdir(paths['images_path'])
    # load json
    try:
        saved, unable_to_download, already_existed  = 0, 0, 0
        with open(config.json_filename) as json_file:
            # load the json
            data = json.load(json_file)
            for item in tqdm(data):
                # get the url
                url = item[config.image_url_key_name]
                # find image extension
                image_extension = url[url.rfind('.'): ]
                # get the image label
                label = item[config.image_label_key_name]  #.strip('[', ']', '{', '}', "\'", '\"')
                label = f'{paths["images_path"]}/{label}'
                if os.path.exists(label) == False:
                    os.mkdir(label)
                # get the image name
                name = f'{item[config.image_name_key_name].replace("/", " ")}{image_extension}'
                # if image already exists
                if os.path.isfile(name):
                    already_existed += 1
                    continue
                else:
                    try:
                        # save the image
                        urllib.request.urlretrieve(url = url, filename = f'{label}/{name}') 
                        saved += 1
                    except:
                        utils.print_head(f"Unable to download : {name}", color='red')
                        unable_to_download += 1
                        continue
            utils.print_head(f"** Dataset Status **\nTotal images processed : {saved+unable_to_download+already_existed}\nSaved images : {saved}\nUnable to download : {unable_to_download}\nAlready Existing : {already_existed}", color='purple')

    except:
        utils.print_head("JSON data file is not provided!\nCheck `json_filename` in config.py file & `dataset` folder...", color='red')
        exit()

def prepare_csv_url_dataset(project_name, config, dataset_path, paths):
    ''' This function loads csv dataset containing image urls, extract images and labels '''
    utils.print_head("Preparing dataset of provided .csv file having urls & labels... ", color='darkcyan')
    # create folder to store images if not exists
    if os.path.exists(paths['images_path']) == False:
        os.mkdir(paths['images_path'])
    # load csv
    try:
        with open((f'{dataset_path}/{config.csv_filename}'.format(f'{dataset_path}/{config.csv_filename}')), 'r') as csv_file: 
            # save the status 
            saved, unable_to_download, already_existed  = 0, 0, 0
            # process the file
            for csv_row in tqdm(reader(csv_file)):
                if csv_row[config.image_url_column_index] != '' and csv_row[config.image_url_column_index] != config.image_url_column_name:
                    # find image extension
                    url = csv_row[config.image_url_column_index]
                    # find image extension
                    image_extension = url[url.rfind('.'): ]
                    # get the label and store belongings to the respective folders
                    label_column_name = csv_row[config.image_label_column_index].strip("[], '' ")
                    label = f"{paths['images_path']}/{label_column_name}"
                    if os.path.exists(label) == False:
                        os.mkdir(label)
                    # get image name
                    name = f'{csv_row[config.image_name_column_index].replace("/", " ")}{image_extension}'
                    # if image already exists
                    if os.path.isfile(name):
                        already_existed += 1
                        continue
                    else:
                        try:
                            urllib.request.urlretrieve(url = csv_row[config.image_url_column_index], filename = f'{label}/{name}')
                            saved += 1
                        except:
                            utils.print_head(f"Unable to download {name}", color='red')
                            unable_to_download += 1
                            continue
            utils.print_head(f"** Dataset Status **\nTotal images processed : {saved+unable_to_download+already_existed}\nSaved images : {saved}\nUnable to download : {unable_to_download}\nAlready Existing : {already_existed}", color='purple')
    except:
        utils.print_head("CSV data file is not provided!\nCheck `csv_filename` in config.py file & `dataset` folder...", color='red')
        exit()

def prepare_csv_pixel_dataset(project_name, config, dataset_path, paths):
    ''' This function loads csv dataset, extract images and labels '''
    utils.print_head("Preparing dataset of provided .csv file having image pixels & labels... ", color='darkcyan')
    # load csv
    try:
        dataset = pd.read_csv(f'{dataset_path}/{config.csv_filename}')
    except:
        utils.print_head("CSV data file is not provided!\nCheck `csv_filename` in config file & `dataset` folder...", color='red')
        exit()

    # create folder to store images if not exists
    if os.path.exists(paths['images_path']) == False:
        os.mkdir(paths['images_path'])

    ## Labels
    # create folders for the labels found in the dataset
    for folder_name in dataset[config.label_column_name].unique():
        if os.path.exists(f"{paths['images_path']}/{str(folder_name)}") == False:
            os.mkdir(f"{paths['images_path']}/{str(folder_name)}")

    labels = dataset[config.label_column_name].tolist()
    
    ## Images 
    # save all the pixels values to list 'pixels' 
    pixels_of_images = dataset[config.images_column_name].tolist()

    for image_label, image_pixels in zip(labels, pixels_of_images) :

        # face contains pixels value of single image
        image = [int(pixel) for pixel in image_pixels.split(' ')]

        # convert image into array
        try:        
            image = np.asarray(image).reshape(config.img_height, config.img_width)
        except:
            utils.print_head("The size provided for resizing is larger than the image!", color='red')
        
        # save the image as .png file
        image_name = str(uuid.uuid4())
        cv2.imwrite(f"{paths['images_path']}/{str(image_label)}/{image_name}.png", image)       
