import os
import torch
import pandas as pd
from torchvision import datasets, transforms, models
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

DATASET_ROOTS = {"imagenet_val": "YOUR_PATH/ImageNet_val/",
                "broden": "/mnt/data/broden1_224/images/",
                "acdc_train": "/mnt/data/ACDC/training/"}

import torch.nn.functional as F

class ACDCDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_shape=(3,224,224)):  
        self.data_dir = data_dir
        self.transform = transform
        self.target_shape = target_shape  # Define a fixed shape
        self.image_paths = []
        self.label_paths = []
        self.patient_labels = self.load_patient_labels()
    
        for patient_folder in os.listdir(data_dir):
            patient_path = os.path.join(data_dir, patient_folder)
            if os.path.isdir(patient_path):
                for file in os.listdir(patient_path):
                    if "_gt" in file and file.endswith(".nii.gz"):
                        img_file = file.replace("_gt", "")
                        img_path = os.path.join(patient_path, img_file)
                        label_path = os.path.join(patient_path, file)
                        if os.path.exists(img_path):
                            self.image_paths.append(img_path)
                            self.label_paths.append(label_path)
    
    def __len__(self):
        return len(self.image_paths)

    def load_patient_labels(self):
        labels = {}
        for patient_folder in os.listdir(self.data_dir):
            patient_path = os.path.join(self.data_dir, patient_folder)
            cfg_path = os.path.join(patient_path, f"{patient_folder}.cfg")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    for line in f:
                        if "Group" in line:
                            labels[patient_folder] = int(line.split("=")[-1].strip())
        return labels
    
    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pads a 4D image array (C, D, H, W) to match target size.
        
        Args:
            image (np.ndarray): Input image of shape (C, D, H, W)
            
        Returns:
            np.ndarray: Padded image of shape (C, target_D, target_H, target_W)
        """
        c, d, h, w = image.shape
        target_d, target_h, target_w = self.target_shape
        
        # Calculate padding values
        pad_d = max(target_d - d, 0)
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        
        # Calculate padding for each dimension
        pad_d_before = pad_d // 2
        pad_d_after = pad_d - pad_d_before
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before
        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before
        
        # Apply padding
        padded_image = np.pad(
            image,
            (
                (0, 0),  # channels
                (pad_d_before, pad_d_after),  # depth
                (pad_h_before, pad_h_after),  # height
                (pad_w_before, pad_w_after)   # width
            ),
            mode='constant',
            constant_values=0
        )
        
        # Crop if necessary
        if padded_image.shape[1] > target_d:
            start = (padded_image.shape[1] - target_d) // 2
            padded_image = padded_image[:, start:start+target_d, :, :]
        if padded_image.shape[2] > target_h:
            start = (padded_image.shape[2] - target_h) // 2
            padded_image = padded_image[:, :, start:start+target_h, :]
        if padded_image.shape[3] > target_w:
            start = (padded_image.shape[3] - target_w) // 2
            padded_image = padded_image[:, :, :, start:start+target_w]
            
        return padded_image

    def _pad_label(self, label: np.ndarray) -> np.ndarray:
        """
        Pads a 3D label array (D, H, W) to match target size.
        
        Args:
            label (np.ndarray): Input label of shape (D, H, W)
            
        Returns:
            np.ndarray: Padded label of shape (target_D, target_H, target_W)
        """
        d, h, w = label.shape
        target_d, target_h, target_w = self.target_shape
        
        # Calculate padding values
        pad_d = max(target_d - d, 0)
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        
        # Calculate padding for each dimension
        pad_d_before = pad_d // 2
        pad_d_after = pad_d - pad_d_before
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before
        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before
        
        # Apply padding
        padded_label = np.pad(
            label,
            (
                (pad_d_before, pad_d_after),  # depth
                (pad_h_before, pad_h_after),  # height
                (pad_w_before, pad_w_after)   # width
            ),
            mode='constant',
            constant_values=0
        )
        
        # Crop if necessary
        if padded_label.shape[0] > target_d:
            start = (padded_label.shape[0] - target_d) // 2
            padded_label = padded_label[start:start+target_d, :, :]
        if padded_label.shape[1] > target_h:
            start = (padded_label.shape[1] - target_h) // 2
            padded_label = padded_label[:, start:start+target_h, :]
        if padded_label.shape[2] > target_w:
            start = (padded_label.shape[2] - target_w) // 2
            padded_label = padded_label[:, :, start:start+target_w]
            
        return padded_label
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        patient_folder = os.path.basename(os.path.dirname(img_path))
        patient_label = self.patient_labels.get(patient_folder, -1)

        # Load images using nibabel
        img_nib = nib.load(img_path)
        label_nib = nib.load(label_path)
        
        img = img_nib.get_fdata().astype(np.float32)  
        label = label_nib.get_fdata().astype(np.int64)  

        # Normalize image to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        
        # Convert to tensors and add channel dim
        img = torch.tensor(img).unsqueeze(0)  # (1, D, H, W)
        #img = img.repeat(3, 1, 1, 1)  # Repeat to get 3 channels: (3, D, H, W)
        label = torch.tensor(label)
        
        # Apply padding to match target shape
        img = self._pad_image(img)
        label = self._pad_label(label)  # Ensure labels are also the same size
        
        if self.transform:
            img = self.transform(img)
        img = img.squeeze(0)
       
        return img, label#, patient_label

def get_target_model(target_name, device):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if target_name == 'resnet18_places': 
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif "vit_b" in target_name:
        target_name_cap = target_name.replace("vit_b", "ViT_B")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "resnet" in target_name:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    
    target_model.eval()
    return target_model, preprocess

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
    
    elif dataset_name == "acdc_train":
        data = ACDCDataset(DATASET_ROOTS["acdc_train"], transform=None)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    return data


def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")
    
    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')
        
        found = (name+'-s' in broden_scenes['name'].values)
        
        if found:
            id_to_broden_label[i] = name.replace('-', '/')+'-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label
    
def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]
    
    return cifar100_has_superclass, cifar100_doesnt_have_superclass