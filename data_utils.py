import os
import torch
import pandas as pd
from torchvision import datasets, transforms, models
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from PIL import Image

DATASET_ROOTS = {"imagenet_val": "YOUR_PATH/ImageNet_val/",
                "broden": "/mnt/data/broden1_224/images/",
                "acdc_train": "/mnt/data/ACDC/training/",
                "mnm2s": "/mnt/data/MnM2s/MnM2/"}

class ACDCDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_shape=(3,224,224)):  
        self.data_dir = data_dir
        self.transform = transform
        self.target_shape = target_shape  # Define a fixed shape
        self.image_paths = []
        self.label_paths = []
        self.patient_labels = self.load_patient_labels()
        # Create a mapping of text labels to numbers
        self.label_mapping = self.create_label_mapping()
        
        # Collect all NIfTI file paths and their corresponding slice indices
        self.image_slices = []
        for patient_folder in os.listdir(data_dir):
            patient_path = os.path.join(data_dir, patient_folder)
            if os.path.isdir(patient_path):
                for file in os.listdir(patient_path):
                    if "_gt" not in file and file.endswith(".nii.gz") and "4d" not in file:
                        nifti_path = os.path.join(patient_path, file)
                        label_path = os.path.join(patient_path, file)
                        img = nib.load(nifti_path)
                        num_slices = img.shape[2]  # Number of slices along z-axis
                        for slice_idx in range(num_slices):
                            self.image_slices.append((nifti_path, slice_idx))  # Store path and slice index
                            self.label_paths.append(label_path)
                            self.image_paths.append(nifti_path)
    
    def __len__(self):
        return len(self.image_slices)
    
    def load_patient_labels(self):
        labels = {}
        for patient_folder in os.listdir(self.data_dir):
            patient_path = os.path.join(self.data_dir, patient_folder)
            cfg_path = os.path.join(patient_path, "Info.cfg")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    for line in f:
                        if "Group" in line:
                            labels[patient_folder] = line.split(":")[-1].strip()
        return labels
    
    def create_label_mapping(self):
        """ Create a mapping from text labels to numeric values. """
        unique_labels = set(self.patient_labels.values())  # Extract unique labels
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}  # Assign numbers
        #print(f"Label Mapping: {label_mapping}")  # Debugging: Check mapping
        return label_mapping

    def preprocess_slice(self, slice_data):
        """
        Preprocesses a single 2D slice into a tensor of shape (3, 224, 224).
        
        Args:
            slice_data (np.ndarray): A single 2D slice of shape (H, W).
        
        Returns:
            torch.Tensor: Preprocessed tensor of shape (3, 224, 224).
        """
        # Normalize the slice data to range [0, 1]
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-5)
        
        # Convert to PIL Image for resizing
        slice_pil = Image.fromarray((slice_data * 255).astype(np.uint8))  # Convert to uint8 for PIL compatibility
        
        # Resize and pad to make square and match target dimensions (224x224)
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to 224x224
            transforms.ToTensor(),  # Convert back to tensor
        ])
        
        img_resized = transform(slice_pil)  # Shape: (1, H, W)
        
        # Repeat channels to make it (3, H, W)
        img_resized = img_resized.repeat(3, 1, 1)  # Shape: (3, H, W)
        
        return img_resized

    def __getitem__(self, idx):
        """
        Loads a single slice from the dataset.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            torch.Tensor: Tensor of shape (3, 224, 224) for the selected slice.
        """
        #img_path = self.image_paths[idx]
        nifti_path, slice_idx = self.image_slices[idx]
        label_path = self.label_paths[idx]
        patient_folder = os.path.basename(os.path.dirname(nifti_path))
        patient_label = self.patient_labels.get(patient_folder, -1)
        patient_label = self.label_mapping.get(patient_label, -1)  # Default to -1 if not in mapping
        patient_label = torch.tensor(patient_label, dtype=torch.long)
        
        # Load the NIfTI file and extract the specified slice
        img = nib.load(nifti_path)
        data = img.get_fdata()  # Get the image data as a NumPy array
        slice_data = data[:, :, slice_idx]  # Extract the specified slice
        
        # Preprocess the slice
        tensor = self.preprocess_slice(slice_data)
        
        return tensor, patient_label

class MnMsDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_shape=(3, 224, 224)):  
        self.data_dir = data_dir + "dataset/"
        self.labels_csv = data_dir + "dataset_information.csv"
        self.transform = transform
        self.target_shape = target_shape
        self.image_slices = []
        self.labels = self.load_labels()
        
        # Collect all NIfTI file paths and their corresponding slice indices
        for patient_folder in os.listdir(self.data_dir):
            patient_path = os.path.join(self.data_dir, patient_folder)
            if os.path.isdir(patient_path):
                for file in os.listdir(patient_path):
                    if file.endswith(".nii.gz") and "gt" not in file and "CINE" not in file:
                        nifti_path = os.path.join(patient_path, file)
                        img = nib.load(nifti_path)
                        num_slices = img.shape[2]  # Number of slices along z-axis
                        for slice_idx in range(num_slices):
                            self.image_slices.append((nifti_path, slice_idx, patient_folder))

    def __len__(self):
        return len(self.image_slices)
    
    def load_labels(self):
        """
        Load patient labels from a CSV file.
        Assumes the CSV has 'subject_code' for patient ID and 'disease' for label.
        """
        df = pd.read_csv(self.labels_csv)
        labels = {int(row['SUBJECT_CODE']): row['DISEASE'] for _, row in df.iterrows()}
        unique_labels = sorted(set(labels.values()))
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        return labels
    
    def create_label_mapping(self):
        """
        Create a mapping from disease labels to numeric values.
        
        Returns:
            dict: A dictionary mapping each disease label (e.g., 'Healthy') to a unique integer (e.g., 0).
        """
        # Extract the unique disease labels from the loaded labels
        unique_labels = set(self.labels.values())  # Disease labels
        
        # Create a mapping from text labels to numbers
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        return label_mapping


    def preprocess_slice(self, slice_data):
        """Preprocesses a 2D slice into a tensor of shape (3, 224, 224)."""
        if slice_data is None or slice_data.size == 0:
            raise ValueError("Received an empty or None slice.")

        slice_data = np.squeeze(slice_data)  # Ensure it's 2D
        if len(slice_data.shape) != 2:
            raise ValueError(f"Expected 2D slice, got shape {slice_data.shape}")

        slice_data = np.nan_to_num(slice_data, nan=0.0)  # Replace NaNs
        slice_data = slice_data.astype(np.float32)  # Ensure correct type

        min_val, max_val = slice_data.min(), slice_data.max()
        if max_val - min_val < 1e-5:  # Prevent division by zero
            slice_data = np.zeros_like(slice_data)  # Replace with blank image
        else:
            slice_data = (slice_data - min_val) / (max_val - min_val)

        slice_pil = Image.fromarray((slice_data * 255).astype(np.uint8))  

        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        
        img_resized = transform(slice_pil)  # Shape: (1, H, W)
        img_resized = img_resized.repeat(3, 1, 1)  # Convert to (3, 224, 224)

        return img_resized

    def __getitem__(self, idx):
        """
        Loads a single slice from the dataset.
        """
        nifti_path, slice_idx, patient_folder = self.image_slices[idx]
        #print(self.labels)
        patient_label = self.labels[int(patient_folder)]
        patient_label = self.label_mapping.get(patient_label, -1)
        patient_label = torch.tensor(patient_label, dtype=torch.long)
        img = nib.load(nifti_path)
        data = img.get_fdata()
        slice_data = data[:, :, slice_idx]
    
        tensor = self.preprocess_slice(slice_data)
        return tensor, patient_label


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
        if target_name.endswith("_custom"):  # Handle custom ResNet models
            target_name_cap = target_name[:-7]#.replace("resnet", "ResNet")
            custom_path = "models/best_resnet152_MnMs_scratch_img.pth"
            target_model = eval(f"models.{target_name_cap}(weights=None).to(device)")
            target_model.fc = torch.nn.Linear(target_model.fc.in_features, 8).to(device)  # Match the original training setup
            checkpoint = torch.load(custom_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            target_model.load_state_dict(state_dict['model_state_dict'])
            target_model.eval()
            preprocess = get_resnet_imagenet_preprocess()
        else:
            target_name_cap = target_name.replace("resnet", "ResNet")
            weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
            preprocess = weights.transforms()
            target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "densenet" in target_name:
        if target_name.endswith("_custom"):  # Handle custom ResNet models
            target_name_cap = target_name[:-7]#.replace("resnet", "ResNet")
            custom_path = "models/best_densenet161_MnMs_scratch_img.pth"
            target_model = eval(f"models.{target_name_cap}(weights=None).to(device)")
            target_model.classifier = torch.nn.Linear(target_model.classifier.in_features, 8).to(device)  # Match the original training setup
            checkpoint = torch.load(custom_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            target_model.load_state_dict(state_dict['model_state_dict'])
            target_model.eval()
            preprocess = get_resnet_imagenet_preprocess()
        else:
            target_name_cap = target_name.replace("densenet", "DenseNet")
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

    elif dataset_name == "mnm2s":
        data = MnMsDataset(DATASET_ROOTS["mnm2s"], transform=None)
        
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