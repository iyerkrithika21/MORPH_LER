import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import glob 
import pickle
import itertools

def pkload(fname):

    with open(fname, 'rb') as f:

        return pickle.load(f)






def preprocess_image(image, modality="ct", segmentation=None):
    if modality == "ct":
        min_ = -1000
        max_ = 1000
        image = torch.clamp(image, min=min_, max=max_)
    elif modality == "mri":
        min_ = image.min().item()
        max_ = torch.quantile(image, 0.99).item()
        image = torch.clamp(image, min=min_, max=max_)
    else:
        raise ValueError(f"{modality} not recognized. Use 'ct' or 'mri'.")

    # Normalize the image
    image = (image - min_) / (max_ - min_)

    # Apply segmentation mask if provided
    if segmentation is not None:
        image = apply_mask(image, segmentation)
        
    return image

def preprocess(images, modality="ct", segmentations=None):
    if segmentations is not None and len(segmentations) != len(images):
        raise ValueError("The number of segmentations must match the number of images.")

    processed_images = []
    for i, image in enumerate(images):
        segmentation = segmentations[i] if segmentations is not None else None
        processed_image = preprocess_image(image, modality, segmentation)
        processed_images.append(processed_image)
        
    return torch.stack(processed_images)

def apply_mask(image, mask):
    return image * mask


class CustomImageDataset(Dataset):
	def __init__(self, image_dir, transform=None):
		self.image_dir = image_dir
		self.transform = transform
		self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		img_path = os.path.join(self.image_dir, self.image_files[idx])
		name = os.path.basename(img_path)
		image = Image.open(img_path).convert('L')
		
		if self.transform:
			image = self.transform(image)
		return image, name

class ImageSliceDataset(Dataset):
	def __init__(self):
		"""
		Args
		"""
		self.data_dir = "/usr/sci/scratch/IyerNLSSM/RegData/neurite-oasis.2d.v1.0"
		self.transform = transforms.Compose([
			transforms.Resize((160, 160)),  # Resize to 160x160
			transforms.ToTensor(),          # Convert to tensor (C, H, W)
		])

		df = pd.read_csv("/usr/sci/scratch/IyerNLSSM/RegData/neurite-oasis.2d.v1.0/oasis_1.csv")
		
		df.set_index('ID', inplace=True)
		self.metadatasheet = df
		self.folders = [folder for folder in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, folder))]
	
	def __len__(self):
		return len(self.folders)
	
	def __getitem__(self, idx):
		folder = self.folders[idx]
		folder_path = os.path.join(self.data_dir, folder)
		metadata = self.metadatasheet.loc[f'{folder}'.replace("OASIS_","")]

		# Load the 2D slices from the respective files
		slice_norm_path = os.path.join(folder_path, 'slice_norm.nii.gz')

		# Load the nifti files using nibabel (Get the image data as numpy array)
		slice_norm = nib.load(slice_norm_path).get_fdata()
		
		slice_norm = slice_norm[:160,:160,0]
		
		# Convert to PIL images (assuming slices are 2D)
		slice_norm_img = Image.fromarray(slice_norm)
		
		# Resize to 160x160 using transforms (only resizing)
		slice_norm_tensor = self.transform(slice_norm_img).reshape((1,160, -1, 1))
		
		slice_norm_tensor = preprocess(slice_norm_tensor, modality='mri')

		# Add channel dimension (assuming grayscale 2D slices, so we add a "channel" dimension)
		slice_norm_tensor = slice_norm_tensor.reshape((1,160,-1))  

		
		# Return the tensors
		# return {'slice_norm': slice_norm_tensor, 'folder':folder, 'metadata': metadata}
		return slice_norm_tensor, folder, metadata.tolist()


# class OASISBrainDataset(Dataset):
#     '''
# These two datasets are randomly sampling two pairs of images
# '''
class OASISBrainDataset(Dataset):
	def __init__(self):
		self.paths = glob.glob("/usr/sci/scratch/IyerNLSSM/RegData/OASIS_L2R_2021_task03/All/*.pkl")# data_path
		# self.transforms = transforms
		self.pair_indices = list(itertools.combinations(range(len(self.paths)), 2))

	def one_hot(self, img, C):
		out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
		for i in range(C):
			out[i,...] = img == i
		return out

	def __getitem__(self, index):
		path = self.paths[index]
		tar_list = self.paths.copy()
		tar_list.remove(path)
		random.shuffle(tar_list)
		tar_file = tar_list[0]
		x, x_seg = pkload(path)
		y, y_seg = pkload(tar_file)
		x, y = x[None, ...], y[None, ...]
		x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
		# x, x_seg = self.transforms([x, x_seg])
		# y, y_seg = self.transforms([y, y_seg])
		x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
		y = np.ascontiguousarray(y)
		x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
		y_seg = np.ascontiguousarray(y_seg)
		x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
		
		x = preprocess(images = x, modality='mri',segmentations = x_seg)
		y = preprocess(images = y, modality='mri',segmentations = y_seg)
		return x, y, x_seg, y_seg, os.path.basename(path).split(".pkl")[0], os.path.basename(tar_file).split(".pkl")[0]

	def __len__(self):
		return len(self.paths)

	# def __getitem__(self, index):
	# 	idx1, idx2 = self.pair_indices[index]
	# 	path1, path2 = self.paths[idx1], self.paths[idx2]
	# 	x, x_seg = pkload(path1)
	# 	y, y_seg = pkload(path2)
	# 	x, y = x[None, ...], y[None, ...]
	# 	x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
	# 	# x, x_seg = self.transforms([x, x_seg])
	# 	# y, y_seg = self.transforms([y, y_seg])
	# 	x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
	# 	y = np.ascontiguousarray(y)
	# 	x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
	# 	y_seg = np.ascontiguousarray(y_seg)
	# 	x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
	# 	# print(path1)
	# 	# print(path2)
	# 	# print("here")
	# 	x = preprocess(images = x, modality='mri',segmentations = x_seg)
	# 	y = preprocess(images = y, modality='mri',segmentations = y_seg)
				
	# 	return x, y, x_seg, y_seg, os.path.basename(path1).split(".pkl")[0], os.path.basename(path2).split(".pkl")[0]

	# def __len__(self):
	# 	return len(self.pair_indices)
    
    # def __init__(self):
    #     self.paths =  glob.glob("/usr/sci/scratch/IyerNLSSM/RegData/OASIS_L2R_2021_task03/All/*.pkl")
    #     # self.transforms = transforms
    #     # self.transform = transforms.Compose([
	# 	# 	transforms.Resize((160, 160, 160)),  # Resize to 160x160
	# 	# 	transforms.ToTensor(),          # Convert to tensor (C, H, W)
	# 	# ])


def get_torusbmp_data(batch_size):

	# Directory containing your images
	image_directory = "/home/sci/iyerkrithika/SSMFromImages/LERPIA/datasets/torus_bump_300/images2D/"

	# Define the transform to resize the images
	transform = transforms.Compose([
		transforms.Resize((160, 160)),  # Resize to 160x160
		transforms.ToTensor(),           # Convert to tensor
	])

	# Create dataset
	dataset = CustomImageDataset(image_dir=image_directory, transform=transform)

	
	test_split = 0.1 
	val_split = 0.1
	# Calculate number of samples for train/val/test splits
	total_size = len(dataset)
	test_size = int(test_split * total_size)
	val_size = int(val_split * total_size)
	train_size = total_size - val_size - test_size
	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
	
	indices = list(range(total_size))
	random.shuffle(indices)
	dataset = torch.utils.data.Subset(dataset, indices)
	
	train_dataset_2, val_dataset_2, _ = random_split(dataset, [train_size, val_size, test_size])

	# Create DataLoaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, drop_last=True)
	val_loader_2 = DataLoader(val_dataset_2, batch_size=1, shuffle=False)
	

	return train_loader, train_loader_2, val_loader,val_loader_2, test_loader


def get_oasis2D_data(batch_size):

	# Create dataset
	dataset = ImageSliceDataset()

	
	test_split = 0.1 
	val_split = 0.1
	# Calculate number of samples for train/val/test splits
	total_size = len(dataset)
	test_size = int(test_split * total_size)
	val_size = int(val_split * total_size)
	train_size = total_size - val_size - test_size
	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
	
	indices = list(range(total_size))
	random.shuffle(indices)
	dataset = torch.utils.data.Subset(dataset, indices)
	
	train_dataset_2, val_dataset_2, _ = random_split(dataset, [train_size, val_size, test_size])

	# Create DataLoaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, drop_last=True)
	val_loader_2 = DataLoader(val_dataset_2, batch_size=1, shuffle=False)
	

	return train_loader, train_loader_2, val_loader,val_loader_2, test_loader



def get_oasis3D_data(batch_size):

	# Create dataset
	dataset = OASISBrainDataset()

	
	test_split = 0.1 
	val_split = 0.1
	# Calculate number of samples for train/val/test splits
	total_size = len(dataset)
	test_size = int(test_split * total_size)
	val_size = int(val_split * total_size)
	train_size = total_size - val_size - test_size
	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
	
	indices = list(range(total_size))
	random.shuffle(indices)
	dataset = torch.utils.data.Subset(dataset, indices)
	
	train_dataset_2, val_dataset_2, _ = random_split(dataset, [train_size, val_size, test_size])

	# Create DataLoaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

	train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, drop_last=True)
	val_loader_2 = DataLoader(val_dataset_2, batch_size=1, shuffle=False)
	

	return train_loader, train_loader_2, val_loader,val_loader_2, test_loader