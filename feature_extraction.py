import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix#, plot_confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.image as mpimg
from data_utils import ACDCDataset, MnMsDataset

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option('display.max_colwidth',3000)

activation_values = None
activations_avg = []

def hook_fn_resnet(module, input, output):
    global activation_values
    global activations_avg
    activation_values = output
    unit_avg={}
    for grad in output:  
        try:
            for i,j in enumerate(grad):
                unit_avg[i+1]=j.mean().item()
            activations_avg.append(unit_avg)
        except AttributeError: 
            print ("None found for Gradient")

target_layer = None

def feature_extraction (train_dir, test_dir=None, num_epochs=50, batch_size=16, learning_rate=1e-4, dataset_name="ACDC", num_classes=5, model_name='resnet152'):
    transform_resnet152 = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match ResNet input size
        transforms.ToTensor(),          # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
        ])

    # Dynamically load the appropriate dataset class based on `dataset_name`
    if dataset_name == "ACDC":
        DatasetClass = ACDCDataset  # Replace with other dataset classes if needed
    elif dataset_name == "MnMs":
        DatasetClass = MnMsDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load datasets
    # Load datasets and handle train-test split if `test_dir` is None
    full_dataset = DatasetClass(train_dir, transform=transform_resnet152)

    if test_dir is None:
        test_split_ratio = 0.3  # Use 30% of the data for testing
        test_size = int(len(full_dataset) * test_split_ratio)
        train_size = len(full_dataset) - test_size
        
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        print(f"Dataset split: {train_size} training samples and {test_size} testing samples.")
    else:
        train_dataset = full_dataset
        test_dataset = DatasetClass(test_dir, transform=transform_resnet152)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    custom_path = "models/best_resnet152_MnMs_scratch_img.pth"
    model = eval(f"models.{model_name}(weights=None).to(device)")
    model.fc = torch.nn.Linear(model.fc.in_features, 8).to(device)  # Match the original training setup
    checkpoint = torch.load(custom_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    if model == 'resnet152':
        target_layer_name = 'layer4'  #avgpool Replace this with the name of your desired layer
    elif model == 'densenet161':
        target_layer_name = 'features'

    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    hook_handle = target_layer.register_forward_hook(hook_fn_resnet)

    image_name_train=[]
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            model(images)
        image_name_train.append(labels[0].item())
        

    d_avg = pd.DataFrame(activations_avg)
    n = pd.DataFrame(image_name_train, columns=['name'])
    dt_avg = d_avg.merge(n,how='inner',left_index=True, right_index=True)
    dt_avg.to_csv(f'dt_avg_{model_name}_{dataset_name}.csv')
    hook_handle.remove()

def get_top_features(model_name, dataset_name):
    dt_avg = pd.read_csv(f'dt_avg_{model_name}_{dataset_name}.csv',index_col=0)

    def top_n_columns(row, n=18):
        return row.nlargest(n).index.tolist()
    
    def top_n_columns_values(row, n=18):
        return row.nlargest(n).tolist()

    dt_avg['top_feature'] = dt_avg.iloc[:, :-2].apply(top_n_columns, axis=1)
    dt_avg['top_feature_values'] = dt_avg.iloc[:, :-3].apply(top_n_columns_values, axis=1)

    df=dt_avg[['name','class','top_feature','top_feature_values']]

    new_df = df['top_feature'].apply(lambda x: pd.Series(x))
    new_df_value = df['top_feature_values'].apply(lambda x: pd.Series(x))

    result_df = pd.concat([df, new_df], axis=1)
    result_df_values = pd.concat([df, new_df_value], axis=1)
    #result_df_inception = pd.concat([df_inception, new_inception], axis=1)

    # Drop the original list column if needed
    result_df.drop('top_feature', axis=1, inplace=True)
    result_df_values.drop('top_feature_values', axis=1, inplace=True)

    result_df.drop('top_feature_values', axis=1, inplace=True)
    result_df_values.drop('top_feature', axis=1, inplace=True)

    df_final_neuron = pd.melt(result_df,id_vars=['name','class']).sort_values('name')
    df_final_values = pd.melt(result_df_values,id_vars=['name','class']).sort_values('name')

    df_final = df_final_neuron.merge(df_final_values, on=['name','variable'],how='inner')[['name','class_x','variable','value_x','value_y']].sort_values(['name','variable']).rename(columns={'class_x':'class','variable':'rank','value_x':'neuron_id','value_y':'act_map'})

    df_final['neuron_id'] = df_final['neuron_id'].astype('Int64')
    df_final['act_map'] = df_final['act_map'].astype('Float64')

    df_final.to_csv(f'df_final_{model_name}_{dataset_name}.csv',index=None)

def connect_clip_features(model_name, dataset_name):
    df_final = pd.read_csv(f'df_final_{model_name}_{dataset_name}.csv')

    def normalisation(df_clip_resnet_l):
        min_values_res = df_clip_resnet_l['similarity'].min()
        max_values_res = df_clip_resnet_l['similarity'].max()
        
        df_clip_resnet_l['similarity_norm'] = (df_clip_resnet_l['similarity'] - min_values_res) / (max_values_res - min_values_res)
        
        return df_clip_resnet_l

    # %%
    def get_clip_results(folder_list):
        data_clip = '/mnt/data/dissect/CLIP-dissect/results'
        df_clip_resnet = pd.read_csv(data_clip+'/{}/descriptions.csv'.format(folder_list[0]))# resnet152_action40_vitb16_act40
        df_clip_inception = pd.read_csv(data_clip+'/{}/descriptions.csv'.format(folder_list[1])) # inception_action40_vitb16_act40/descriptions.csv

        df_clip_resnet_l = df_clip_resnet[df_clip_resnet['layer']=='layer4']
        df_clip_inception_l = df_clip_inception[df_clip_inception['layer']=='Mixed_7c']

        df_clip_resnet_l['unit'] = df_clip_resnet_l['unit'] + 1
        df_clip_inception_l['unit'] = df_clip_inception_l['unit'] + 1

        df_clip_resnet_l, df_clip_inception_l = normalisation(df_clip_resnet_l, df_clip_inception_l)
        return df_clip_resnet_l, df_clip_inception_l

    df_clip_resnet_gpt_act40  = get_clip_results('resnet152_action40_vitb16_act40_gpt4')
