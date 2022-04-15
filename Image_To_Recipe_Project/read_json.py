import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from skimage import io
import os


l1 = open('recipe1M_layers/layer1.json')
l2 = open('recipe1M_layers/layer2.json')

layer1 = json.load(l1) #jsons are loaded as a list of dictionaries with keys as specified on the readme
layer2 = json.load(l2) #same here. 'images' is also a list of dictionaries.


l1 = open('recipe1M_layers/layer1.json')
l2 = open('recipe1M_layers/layer2.json')

layer1 = np.array(json.load(l1))
layer2 = np.array(json.load(l2))[0:100000]

#breakpoint()
# filter layer1 for only values that have images associated with them
layer1_ids = np.array([d['id'] for d in layer1])
ids_with_images = np.array([d['id'] for d in layer2])
layer1_has_img_mask = np.array([x in ids_with_images for x in layer1_ids])
layer1 = layer1[layer1_has_img_mask] #layer 1 with ONLY text data that have images associated with it.

type_arr = np.array([d['partition'] for d in layer1]) #list of types (train, test, val)
layer2_ids = np.array([d['id'] for d in layer2])


# TRAINING SET
mask = type_arr == 'train'
txt_data = layer1[mask] #text data that have the type TEST
txt_data_ids = np.array([d['id'] for d in txt_data])

mask1 = np.array([x in txt_data_ids for x in layer2_ids]) #mask for LAYER2 (image one) with the type TEST
imgs = layer2[mask1]
with open('imgs_train.json', 'w') as f:
	json.dump(imgs.tolist(), f, indent=2)
with open('txt_train.json', 'w') as f:
	json.dump(txt_data.tolist(), f, indent=2)

# TESTING SET
mask = type_arr == 'test'
txt_data = layer1[mask] #text data that have the type TEST
txt_data_ids = np.array([d['id'] for d in txt_data])

mask1 = np.array([x in txt_data_ids for x in layer2_ids]) #mask for LAYER2 (image one) with the type TEST
imgs = layer2[mask1]
with open('imgs_test.json', 'w') as f:
	json.dump(imgs.tolist(), f, indent=2)
with open('txt_test.json', 'w') as f:
	json.dump(txt_data.tolist(), f, indent=2)

# VALIDATION SET
mask = type_arr == 'val'
txt_data = layer1[mask] #text data that have the type TEST
txt_data_ids = np.array([d['id'] for d in txt_data])

mask1 = np.array([x in txt_data_ids for x in layer2_ids]) #mask for LAYER2 (image one) with the type TEST
imgs = layer2[mask1]
with open('imgs_val.json', 'w') as f:
	json.dump(imgs.tolist(), f, indent=2)
with open('txt_val.json', 'w') as f:
	json.dump(txt_data.tolist(), f, indent=2)

breakpoint()
