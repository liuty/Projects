import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from skimage import io
from PIL import Image
import os

class OneMRecipeDataset(Dataset):
	def __init__(self, partition):
		if partition == 'train':
			f1 = open('imgs_train.json')
			f2 = open('txt_train.json')
		if partition == 'test':
			f1 = open('imgs_test.json')
			f2 = open('txt_test.json')
		if partition == 'val':
			f1 = open('imgs_val.json')
			f2 = open('txt_val.json')

		self.partition = partition
		self.imgs = json.load(f1)
		self.txts = json.load(f2)

		#APPLY TRANSFORMS
		if self.partition == 'train':
			xform = transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor()])
		if self.partition == 'test' or self.partition == 'val':
			xform = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor()])
		self.xform = xform



	def __len__(self):
		return len(self.imgs)

	#def __getitem__(self, idx):
		#try:
		#	iter(idxs)
		#except:
		#	idxs = [idxs]

		#if torch.is_tensor(idxs):
		#	idxs = idxs.tolist()

		#for idx in idxs: ###
		#img_info = self.imgs[idx]['images'][0]['id']
		#pic = Image.open(self.partition+'/'+img_info[0]+'/'+img_info[1]+'/'+img_info[2]+'/'+img_info[3]+'/'+img_info)

		#pic = self.xform(pic)

		#txt = self.txts[idx]
		#name = txt['title']
		#recipe_ingredients = txt['ingredients']
		#recipe_directions = txt['instructions']
		#return (pic, name, recipe_ingredients, recipe_directions)

	#def __getitem__(self, idxs):
	#	try:
	#		iter(idxs)
	#	except:
	#		idxs = idxs#[idxs]

	#	if torch.is_tensor(idxs):
	#		idxs = idxs.tolist()

	#	pics = torch.Tensor()
	#	img_infos = self.imgs[idxs]

	#	item_ids = np.array([])
	#	for img_info in img_infos:
	#		img_id = img_info['images'][0]['id']

	#		tmp2 = tmp1[0]
	#		img_id = tmp2['id']
	#		pic = Image.open(self.partition+'/'+img_id[0]+'/'+img_id[1]+'/'+img_id[2]+'/'+img_id[3]+'/'+img_id)

	#		pic = self.xform(pic)[None,:]
	#		pics = torch.cat((pics,pic),dim=0)
	#		item_ids = np.append(item_ids, img_info['id'])

	#	return pics, item_ids


	def __getitem__(self, idxs):
		try:
			iter(idxs)
		except:
			idxs = idxs#[idxs]

		if torch.is_tensor(idxs):
			idxs = idxs.tolist()

		img_info = self.imgs[idxs]

		img_id = img_info['images'][0]['id']
		pic = Image.open(self.partition+'/'+img_id[0]+'/'+img_id[1]+'/'+img_id[2]+'/'+img_id[3]+'/'+img_id)
		pic = self.xform(pic)
		return pic, idxs

	def getinfo(self, ndx):
		return self.txts[ndx]


