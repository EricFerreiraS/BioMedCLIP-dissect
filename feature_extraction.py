# %% [markdown]
# # CLIP-Dissect applied to the Action40 dataset
# This experiment verifies the CLIP-Dissection approach over a image classification task.
# 
# The experiment comprises:
# - Image classification task (action), based on Action-40 dataset
# - Based on the models Resnet-152 and Inception-V3, both trained on Imagenet
# - Get the most important concepts for each class

# %%
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix#, plot_confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.image as mpimg

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option('display.max_colwidth',3000)
import clip
#from utils import show_image

# %% [markdown]
# ### Creating the dataloader

# %%
# Define data paths
data_dir = "/mnt/data/netdissection/net-dissection-lite/dataset/Stanford40/"
full_dir = os.path.join(data_dir,"JPEGImages")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "test")

# %%
def data_transforms(kind='resnet'):
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    if kind == 'resnet':
        preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    elif kind == 'inception':
        preprocess = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

# Create ImageFolder datasets
train_dataset_resnet = ImageFolder(train_dir, transform=data_transforms())
val_dataset_resnet = ImageFolder(val_dir, transform=data_transforms())

train_dataset_inception = ImageFolder(train_dir, transform=data_transforms('inception'))
val_dataset_inception = ImageFolder(val_dir, transform=data_transforms('inception'))

# Create DataLoaders
batch_size = 32
train_loader_resnet = DataLoader(train_dataset_resnet, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader_resnet = DataLoader(val_dataset_resnet, batch_size=batch_size, shuffle=False, num_workers=4)

train_loader_inception = DataLoader(train_dataset_inception, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader_inception = DataLoader(val_dataset_inception, batch_size=batch_size, shuffle=False, num_workers=4)

# Check the number of classes in the dataset (should be 40)
num_classes = len(train_dataset_inception.classes)
print("Number of classes:", num_classes)

# %% [markdown]
# ### Reading the trained models

# %%
clip_rn50, _ = clip.load('RN50', device=device)

# %%
resnet_152 = models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V2')
resnet_152_num = '152'
inception = models.inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
inception_num = 'v3'

# %%
resnet_50 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
resnet_50_num = '50'

# %% [markdown]
# ### Training models function

# %%
def training_model(model, train_loader, val_loader,model_num,kind='last'):
    model_name = model.__class__.__name__

    #model.eval()

    # Freeze all layers except the last fully connected layer
    if kind=='last':
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad = True
        model.fc = torch.nn.Linear(2048, 40)
    elif kind=='clip':
        for param in model.parameters():
            param.requires_grad = False
        model.ln_final.requires_grad = True
        model.ln_final = torch.nn.Linear(2048, 40)
    else:
        model.fc = torch.nn.Linear(2048, 40)
    

    # Define loss function, optimizer, and learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss()
    if kind=='last':
        #optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9) #only the last layer
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001) #only the last layer
    elif kind=='clip':
        optimizer = torch.optim.Adam(model.ln_final.parameters(), lr=0.001) #only the last layer
    else:
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Move the model to the desired device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if model_name == 'Inception3':
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Calculate and print average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


    # Validation loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy on validation set: {:.2f}%".format(100 * correct / total))
    
    torch.save(model.state_dict(), "{}_action40_{}.pth".format(model_name+str(model_num),kind))
    return model

# %% [markdown]
# ### Training the models

# %%
#freezing the model
resnet_152_last = training_model(resnet_152,train_loader_resnet,val_loader_resnet,resnet_152_num)

# %%
#clip_rn50_last = training_model(clip_rn50, train_loader_resnet, val_loader_resnet, resnet_50_num, kind='clip')

# %%
#resnet50
resnet_50_last = training_model(resnet_50,train_loader_resnet,val_loader_resnet,resnet_50_num)

# %%
resnet_152_optm = training_model(resnet_152,train_loader_resnet,val_loader_resnet,resnet_152_num,'optm')

# %%
#resnet50
resnet_50_optm = training_model(resnet_50,train_loader_resnet,val_loader_resnet,resnet_50_num,'optm')

# %%
#freezing the model
inception_last = training_model(inception,train_loader_inception,inception_num,val_loader_inception)

# %%
inception_optm = training_model(inception,train_loader_inception,val_loader_inception,inception_num,'optm')

# %%
val_loader_resnet = DataLoader(val_dataset_resnet, batch_size=batch_size, shuffle=False, num_workers=4)

true_labels = []
predicted_labels = []

# Put the model in evaluation mode
#resnet.eval()
resnet_152_optm.to('cuda')
resnet_152_optm.eval()

with torch.no_grad():
    for inputs, labels in val_loader_resnet:
        # Forward pass
        outputs = resnet_152_optm(inputs.cuda())

        # Get predicted labels (class with highest probability)
        _, predicted = torch.max(outputs, 1)

        # Append true and predicted labels to the lists
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# %%
cm = confusion_matrix(true_labels, predicted_labels)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
class_labels= val_dataset_resnet.classes
# Create a heatmap of the confusion matrix with percentages
plt.figure(figsize=(30, 30))
sns.heatmap(cm_percentage, annot=True, fmt=".1%", cmap="Blues", 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentages)')
plt.show()

# %% [markdown]
# Inception confusion matrix

# %%
val_loader_inception = DataLoader(val_dataset_inception, batch_size=batch_size, shuffle=False, num_workers=4)

true_labels = []
predicted_labels = []

# Put the model in evaluation mode
inception_optm.eval()

with torch.no_grad():
    for inputs, labels in val_loader_inception:
        # Forward pass
        outputs = inception(inputs.cuda())

        # Get predicted labels (class with highest probability)
        _, predicted = torch.max(outputs, 1)

        # Append true and predicted labels to the lists
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# %%
cm = confusion_matrix(true_labels, predicted_labels)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
class_labels= val_dataset_inception.classes
# Create a heatmap of the confusion matrix with percentages
plt.figure(figsize=(30, 30))
sns.heatmap(cm_percentage, annot=True, fmt=".1%", cmap="Blues", 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentages) - Inception')
plt.show()

# %% [markdown]
# ## Loading the trained models

# %%
# Step 1: Load the pre-trained ResNet model
resnet_152 = models.resnet152(weights=None)
#inception = models.inception_v3(weights=None,init_weights=False)

resnet_152.fc = nn.Linear(resnet_152.fc.in_features, 40)
#inception.fc = nn.Linear(inception.fc.in_features, 40)

resnet_152.load_state_dict(torch.load("ResNet152_action40_optm.pth"))
#inception.load_state_dict(torch.load("Inception3_action40_last.pth"))

# %%
resnet_152.eval()
#inception.eval()

# %% [markdown]
# ### Creating the function to extract the features from the model

# %%
# Step 2: Define a forward hook on the desired layer
target_layer_name_resnet = 'layer4'  #avgpool Replace this with the name of your desired layer
target_layer_name_inception = 'Mixed_7c'

# %%
activation_values_resnet, activation_values_inception = None, None
activations_avg_resnet, activations_avg_inception=[],[]

def hook_fn_resnet(module, input, output):
    global activation_values_resnet
    global activations_avg_resnet
    activation_values_resnet = output
    unit_avg={}
    for grad in output:  
        try:
            for i,j in enumerate(grad):
                unit_avg[i+1]=j.mean().item()
            activations_avg_resnet.append(unit_avg)
        except AttributeError: 
            print ("None found for Gradient")

def hook_fn_inception(module, input, output):
    global activation_values_inception
    global activations_avg_inception
    activation_values_inception = output
    unit_avg={}
    for grad in output:  
        try:
            for i,j in enumerate(grad):
                unit_avg[i+1]=j.mean().item()
            activations_avg_inception.append(unit_avg)
        except AttributeError: 
            print ("None found for Gradient")

target_layer_resnet, target_layer_inception = None, None

for name, module in resnet_152.named_modules():
    if name == target_layer_name_resnet:
        target_layer_resnet = module
        break
'''
for name, module in inception.named_modules():
    if name == target_layer_name_inception:
        target_layer_inception = module
        break
'''

# %%
hook_handle_resnet = target_layer_resnet.register_forward_hook(hook_fn_resnet)
#hook_handle_inception = target_layer_inception.register_forward_hook(hook_fn_inception)

# %% [markdown]
# ### Preprocessing the images and extract the features

# %%
# Step 3: Preprocess the image
def preprocess_image(image_path,kind='resnet'):
    transform = data_transforms(kind)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

# %%
text_file_path = "{}ImageSplits/actions.txt".format(data_dir)
class_names = []
with open(text_file_path, "r") as file:
    class_names = file.read().splitlines()
classes=[]
for c in class_names:
    classes.append(str(c.split('\t')[0]))
classes = classes[1:]

# %%
image_name_train=[]
for c in classes:
    imgs = os.listdir(train_dir+"/{}".format(c))

    for i,img in enumerate(imgs):
        image_tensor_resnet = preprocess_image(train_dir+"/{}/".format(c)+img)
        #image_tensor_inception = preprocess_image(train_dir+"/{}/".format(c)+img,'inception')

        with torch.no_grad():
            output_resnet = resnet_152(image_tensor_resnet)
            #output_inception = inception(image_tensor_inception)
        image_name_train.append(img)

# %%
image_name_val=[]
for c in classes:
    imgs = os.listdir(val_dir+"/{}".format(c))

    for i,img in enumerate(imgs):
        image_tensor_resnet = preprocess_image(val_dir+"/{}/".format(c)+img)

        with torch.no_grad():
            output_resnet = resnet_152(image_tensor_resnet)
            
        image_name_val.append(img)

# %%
#all files
image_name=[]

imgs = os.listdir(full_dir+"")

for i,img in enumerate(imgs):
    image_tensor_resnet = preprocess_image(full_dir+"/{}".format(img))
    image_tensor_inception = preprocess_image(full_dir+"/{}".format(img),'inception')

    with torch.no_grad():
        output_resnet = resnet(image_tensor_resnet)
        output_inception = inception(image_tensor_inception)
    image_name.append(img)

# %% [markdown]
# ### Getting the features for each class

# %%
d_avg_resnet = pd.DataFrame(activations_avg_resnet)
#d_avg_inception = pd.DataFrame(activations_avg_inception)
n = pd.DataFrame(image_name_val, columns=['name'])
dt_avg_resnet = d_avg_resnet.merge(n,how='inner',left_index=True, right_index=True)
#dt_avg_inception = d_avg_inception.merge(n,how='inner',left_index=True, right_index=True)

# %%
dt_avg_resnet['class'] = dt_avg_resnet['name'].apply(lambda x: x.split('.')[0][:-4])
#dt_avg_inception['class'] = dt_avg_inception['name'].apply(lambda x: x.split('.')[0][:-4])

# %%
dt_avg_resnet.to_csv('dt_avg_resnet_optm_test.csv')
#dt_avg_inception.to_csv('dt_avg_inception_last_full.csv')

# %%
dt_avg_resnet_train = pd.read_csv('dt_avg_resnet_optm_train.csv',index_col=0)
dt_avg_resnet_test = pd.read_csv('dt_avg_resnet_optm_test.csv',index_col=0)
#dt_avg_inception = pd.read_csv('dt_avg_inception.csv',index_col=0)

# %%
top_features_per_class_resnet = {}
top_features_per_class_inception = {}
features_columns = list(dt_avg_resnet.columns)[:-2]
# Loop through each group and get the top 10 features for each class

for class_name, group in dt_avg_resnet.groupby('class'):
    mean_features = group[features_columns].mean()  # Calculate the mean of each feature
    top_features = mean_features.nlargest(10)  # Get the top 10 features based on the mean
    top_features_per_class_resnet[class_name] = top_features.index.tolist()

for class_name, group in dt_avg_inception.groupby('class'):
    mean_features = group[features_columns].mean()  # Calculate the mean of each feature
    top_features = mean_features.nlargest(10)  # Get the top 10 features based on the mean
    top_features_per_class_inception[class_name] = top_features.index.tolist()

# %%
df_top_features_resnet = pd.DataFrame.from_dict(top_features_per_class_resnet, orient='index')
df_top_features_inception = pd.DataFrame.from_dict(top_features_per_class_inception, orient='index')

# %%
df_top_features_resnet.to_csv('df_top_features_resnet.csv',index=None)
df_top_features_inception.to_csv('df_top_features_inception.csv',index=None)

# %%
df_top_features_resnet = pd.read_csv('df_top_features_resnet.csv')
df_top_features_inception = pd.read_csv('df_top_features_inception.csv')

# %%
df_top_features_resnet = df_top_features_resnet.transpose()
df_top_features_inception = df_top_features_inception.transpose()

# %%
df_top_features_resnet = df_top_features_resnet.reset_index().set_index('index').stack().reset_index()
df_top_features_inception = df_top_features_inception.reset_index().set_index('index').stack().reset_index()

# %%
df_top_features_resnet[0] = df_top_features_resnet[0].astype('Int64')
df_top_features_inception[0] = df_top_features_inception[0].astype('Int64')

# %% [markdown]
# ### Getting the values for each sample

# %%
dt_avg_resnet.iloc[:, :-2].head()

# %% [markdown]
# can I selected the top 10 from different clusters?

# %%
def top_n_columns(row, n=18):
    return row.nlargest(n).index.tolist()

# %%
def top_n_columns_values(row, n=18):
    return row.nlargest(n).tolist()

# %%
dt_avg_resnet_train['top_feature'] = dt_avg_resnet_train.iloc[:, :-2].apply(top_n_columns, axis=1)
dt_avg_resnet_train['top_feature_values'] = dt_avg_resnet_train.iloc[:, :-3].apply(top_n_columns_values, axis=1)

dt_avg_resnet_test['top_feature'] = dt_avg_resnet_test.iloc[:, :-2].apply(top_n_columns, axis=1)
dt_avg_resnet_test['top_feature_values'] = dt_avg_resnet_test.iloc[:, :-3].apply(top_n_columns_values, axis=1)

# %%
dt_avg_resnet[dt_avg_resnet['name']=='applauding_001.jpg']

# %%
#dt_avg_inception['top_feature'] = dt_avg_inception.iloc[:, :-3].apply(top_n_columns, axis=1)
#dt_avg_inception['top_feature_values'] = dt_avg_inception.iloc[:, :-3].apply(top_n_columns_values, axis=1)

# %%
df_resnet_train=dt_avg_resnet_train[['name','class','top_feature','top_feature_values']]
df_resnet_test=dt_avg_resnet_test[['name','class','top_feature','top_feature_values']]
#df_inception = dt_avg_inception[['name','class','top_feature','top_feature_values']]

# %%
new_resnet_train = df_resnet_train['top_feature'].apply(lambda x: pd.Series(x))
new_resnet_value_train = df_resnet_train['top_feature_values'].apply(lambda x: pd.Series(x))

new_resnet_test = df_resnet_test['top_feature'].apply(lambda x: pd.Series(x))
new_resnet_value_test = df_resnet_test['top_feature_values'].apply(lambda x: pd.Series(x))
#new_inception = df_inception['top_feature'].apply(lambda x: pd.Series(x))

# %%
result_df_resnet_train = pd.concat([df_resnet_train, new_resnet_train], axis=1)
result_df_resnet_values_train = pd.concat([df_resnet_train, new_resnet_value_train], axis=1)
#result_df_inception = pd.concat([df_inception, new_inception], axis=1)

# Drop the original list column if needed
result_df_resnet_train.drop('top_feature', axis=1, inplace=True)
result_df_resnet_values_train.drop('top_feature_values', axis=1, inplace=True)

result_df_resnet_train.drop('top_feature_values', axis=1, inplace=True)
result_df_resnet_values_train.drop('top_feature', axis=1, inplace=True)



result_df_resnet_test = pd.concat([df_resnet_test, new_resnet_test], axis=1)
result_df_resnet_values_test = pd.concat([df_resnet_test, new_resnet_value_test], axis=1)
#result_df_inception = pd.concat([df_inception, new_inception], axis=1)

# Drop the original list column if needed
result_df_resnet_test.drop('top_feature', axis=1, inplace=True)
result_df_resnet_values_test.drop('top_feature_values', axis=1, inplace=True)

result_df_resnet_test.drop('top_feature_values', axis=1, inplace=True)
result_df_resnet_values_test.drop('top_feature', axis=1, inplace=True)

#result_df_inception.drop('top_feature', axis=1, inplace=True)

# %%
result_df_resnet.head()

# %%
df_resnet_final_neuron_train = pd.melt(result_df_resnet_train,id_vars=['name','class']).sort_values('name')
df_resnet_final_values_train = pd.melt(result_df_resnet_values_train,id_vars=['name','class']).sort_values('name')

df_resnet_final_neuron_test = pd.melt(result_df_resnet_test,id_vars=['name','class']).sort_values('name')
df_resnet_final_values_test = pd.melt(result_df_resnet_values_test,id_vars=['name','class']).sort_values('name')

#df_inception_final = pd.melt(result_df_inception,id_vars=['name','class']).sort_values('name')

# %%
df_resnet_final_neuron[df_resnet_final_neuron['name']=='applauding_001.jpg'].sort_values('variable')

# %%
df_resnet_final_values[df_resnet_final_values['name']=='applauding_270.jpg'].sort_values('variable')

# %%
df_resnet_final_train = df_resnet_final_neuron_train.merge(df_resnet_final_values_train, on=['name','variable'],how='inner')[['name','class_x','variable','value_x','value_y']].sort_values(['name','variable']).rename(columns={'class_x':'class','variable':'rank','value_x':'neuron_id','value_y':'act_map'})

df_resnet_final_test = df_resnet_final_neuron_test.merge(df_resnet_final_values_test, on=['name','variable'],how='inner')[['name','class_x','variable','value_x','value_y']].sort_values(['name','variable']).rename(columns={'class_x':'class','variable':'rank','value_x':'neuron_id','value_y':'act_map'})

# %%
df_resnet_final.head()

# %%
df_resnet_final_train['neuron_id'] = df_resnet_final_train['neuron_id'].astype('Int64')
df_resnet_final_train['act_map'] = df_resnet_final_train['act_map'].astype('Float64')

df_resnet_final_test['neuron_id'] = df_resnet_final_test['neuron_id'].astype('Int64')
df_resnet_final_test['act_map'] = df_resnet_final_test['act_map'].astype('Float64')
#df_inception_final['value'] = df_inception_final['value'].astype('Int64')

# %%
df_resnet_final_train.to_csv('df_resnet_final_optm_train.csv',index=None)
df_resnet_final_test.to_csv('df_resnet_final_optm_test.csv',index=None)
#df_inception_final.to_csv('df_inception_final_last_full.csv',index=None)

# %%
df_resnet_final = pd.read_csv('df_resnet_final_last_train.csv')
#df_inception_final = pd.read_csv('df_inception_final_last_full.csv')

# %%
# Unregister the hook after you're done using it
#hook_handle_inception.remove()
hook_handle_resnet.remove()

# %% [markdown]
# ## Adding the information from CLIP-Dissect
# ### CLIP-Dissect runned over the action40 classes' concepts extracted from GPT3.5

# %%
def normalisation(df_clip_resnet_l, df_clip_inception_l):
    min_values_res = df_clip_resnet_l['similarity'].min()
    max_values_res = df_clip_resnet_l['similarity'].max()

    min_values_inc = df_clip_inception_l['similarity'].min()
    max_values_inc = df_clip_inception_l['similarity'].max()
    
    df_clip_resnet_l['similarity_norm'] = (df_clip_resnet_l['similarity'] - min_values_res) / (max_values_res - min_values_res)
    df_clip_inception_l['similarity_norm'] = (df_clip_inception_l['similarity'] - min_values_inc) / (max_values_inc - min_values_inc)
    return df_clip_resnet_l, df_clip_inception_l

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

# %% [markdown]
# ### Normalising

# %%
df_clip_resnet_gpt_act40, df_clip_inception_gpt_act40  = get_clip_results(['resnet152_action40_vitb16_act40_gpt4','inception_action40_vitb16_act40'])

#df_clip_resnet_gpt_20k_act40_img, df_clip_inception_gpt_20k_act40_img  = get_clip_results(['resnet152_imgnetact_vitb16_act&20k','inception_imgnetact_vitb16_act&20k'])

#df_clip_resnet_gpt_img_act40_img, df_clip_inception_gpt_img_act40_img  = get_clip_results(['resnet152_imgnetact_vitb16_act&imgnet','inception_imgnetact_vitb16_act&imgnet'])

#df_clip_resnet_gpt_img_act40, df_clip_inception_gpt_img_act40  = get_clip_results(['resnet152_action40_vitb16_act&imgnet','inception_action40_vitb16_act&imgnet'])

#df_clip_resnet_gpt_img_broden_act40, df_clip_inception_gpt_img_broden_act40  = get_clip_results(['resnet152_imgnetbrodenact_vitb16_act','inception_imgnetbrodenact_vitb16_act'])

#df_clip_resnet_img_broden, df_clip_inception_img_broden  = get_clip_results(['resnet152_imgnetbroden_vitb16_img&broden','inception_imgnetbroden_vitb16_img&broden'])

# %% [markdown]
# ### Merging the dataframes

# %%
#clustered classes
df_merge_resnet = df_top_features_resnet.merge(df_clip_resnet_l, left_on=[0], right_on=['unit'])
df_merge_inception = df_top_features_inception.merge(df_clip_inception_l, left_on=[0], right_on=['unit'])

# %%
#each sample
df_merge_resnet_each = df_resnet_final.merge(df_clip_resnet_l, left_on=['value'], right_on=['unit'])
df_merge_inception_each = df_inception_final.merge(df_clip_inception_l, left_on=['value'], right_on=['unit'])

# %%
df_merge_resnet_each.sort_values('name').head(10)

# %%
df_merge_inception_each.sort_values('name').head(10)

# %%
df_merge_resnet_each.to_csv('resnet152_action40_probs_norm.csv')
df_merge_inception_each.to_csv('inceptionv3_action40_probs_norm.csv')

# %%
df_merge_resnet[df_merge_resnet['level_1']=='applauding']

# %%
#freezed
df_merge_resnet[df_merge_resnet['level_1']=='applauding']

# %%
df_merge_inception[df_merge_inception['level_1']=='applauding']

# %%
#freezed
df_merge_inception[df_merge_inception['level_1']=='applauding']

# %% [markdown]
# - summarisation between the two models, over the normalisated value
# - show the better concepts from CLIP-Dissect (over 70%)
# - looking for the overlaping across the concepts
# 

# %%
def plot_charts(df_clip_resnet_l,df_clip_inception_l):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # Create the first bar chart on the first subplot (ax1)
    ax1.bar(['total concepts', 'unique concepts'],[df_clip_resnet_l.shape[0],len(df_clip_resnet_l['description'].unique())], color='blue')
    ax1.set_xlabel('ResNet-152')
    ax1.set_ylabel('Count')
    ax1.set_title('Total x Unique concepts - ResNet-152')

    # Create the second bar chart on the second subplot (ax2)
    ax2.bar(['total concepts', 'unique concepts'],[df_clip_inception_l.shape[0],len(df_clip_inception_l['description'].unique())], color='green')
    ax2.set_xlabel('Inception_V3')
    ax2.set_ylabel('Count')
    ax2.set_title('Total x Unique concepts - Inception V3')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # Create the first bar chart on the first subplot (ax1)
    ax1.hist(df_clip_resnet_l['similarity_norm'], bins=10, color='blue', alpha=0.7)
    ax1.set_xlabel('ResNet-152')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of similarity_norm - ResNet-152')

    # Create the second bar chart on the second subplot (ax2)
    ax2.hist(df_clip_inception_l['similarity_norm'], bins=10, color='green', alpha=0.7)
    ax2.set_xlabel('Inception_V3')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of similarity_norm - Inception-V3')

    # Adjust the spacing between subplots
    plt.tight_layout()
    # Show the plot
    plt.show()

    print(len(df_clip_inception_l['description'].unique())/df_clip_inception_l.shape[0])

# %% [markdown]
# ### Analysis - Concepts from GPT3 and Action40 Images: V1

# %% [markdown]
# #### CLIP-Dissect
# ![image.png](attachment:image.png)

# %% [markdown]
# #### Getting the top-10 most highlighted neurons from the network
# ![image.png](attachment:image.png)

# %% [markdown]
# $$
# \forall a_{i} \in A_{k}(x_{i})
# $$
# 
# $$
# TopK{x_{i}} = \arg\max_{a, A} a_{i}
# $$
# 
# $$
# \forall c_{j} \in C
# $$
# 
# 
# $$
# TopC{x_{i}} = \arg\max_{a,A;c,C} a_{i,j}
# $$

# %%
plot_charts(df_clip_resnet_gpt_act40, df_clip_inception_gpt_act40)

# %%
#unique concepts recognised by the CLIP-Dissection, using GPT concepts and Action40 images
len(df_clip_resnet_gpt_act40['description'].unique())

# %%
#unique concepts recognised by the CLIP-Dissection, using GPT concepts (superclass and around) and Action40 images
len(df_clip_resnet_gpt_act40['description'].unique())

# %%
df_clip_resnet_gpt_act40['description'].value_counts().reset_index()[df_clip_resnet_gpt_act40['description'].value_counts().reset_index()['count']>10].set_index('description').plot(kind='bar',rot=85)
plt.title('Concepts with more than 30 apperances in the CLIP-Dissection/Resnet')
plt.show()

# %%
df_clip_resnet_gpt_act40['description'].value_counts().reset_index()[df_clip_resnet_gpt_act40['description'].value_counts().reset_index()['count']>30].set_index('description').plot(kind='bar',rot=85)
plt.title('Concepts with more than 30 apperances in the CLIP-Dissection/Resnet')
plt.show()

# %%
df_clip_resnet_gpt_act40['description'].value_counts().reset_index()[df_clip_resnet_gpt_act40['description'].value_counts().reset_index()['count']>15].set_index('description').plot(kind='bar',rot=85)
plt.title('Concepts with more than 15 apperances in the CLIP-Dissection/Resnet')
plt.show()

# %%
l_list=list(df_clip_resnet_gpt_act40[df_clip_resnet_gpt_act40['description']=='a trampoline'].sort_values('similarity',ascending=False)['unit'].values) #a person looking through the viewfinder
l_list[1]

# %%
dt_avg_resnet.sort_values(str(l_list[1]),ascending=False).head(1)['name']

# %%
for img in list(dt_avg_resnet.sort_values(str(l_list[1]),ascending=False).head(1)['name'].values):
    show_image('resnet152','ResNet_action40_last.pth','/mnt/data/netdissection/net-dissection-lite/dataset/Stanford40/JPEGImages/{}'.format(img))
    show_image('resnet152','ResNet_action40_last.pth','/mnt/data/netdissection/net-dissection-lite/dataset/Stanford40/JPEGImages/{}'.format(img),1587)

# %%
df_clip_resnet_gpt_act40[df_clip_resnet_gpt_act40['unit']==1587]

# %%
df_clip_inception_gpt_act40['description'].value_counts().reset_index()[df_clip_inception_gpt_act40['description'].value_counts().reset_index()['count']>30].set_index('description').plot(kind='bar',rot=85)
plt.title('Concepts with more than 30 apperances in the CLIP-Dissection/Inception')
plt.show()

# %%
#unique concepts recognised by the CLIP-Dissection, using GPT concepts and Action40 images
len(df_clip_inception_gpt_act40['description'].unique())

# %%
df_clip_resnet_gpt_act40.sort_values(['similarity'],ascending=False).head(10)

# %%
df_clip_inception_gpt_act40.sort_values(['similarity'],ascending=False).head(10)

# %%
all_concepts = open('/mnt/data/dissect/CLIP-dissect/data/action40_filtered_remove_target_gpt4.txt','r') #action40_concepts
list_concepts = []
for i in all_concepts:
    if len(i) > 0:
        list_concepts.append(i.replace('\n',''))

# %%
set_concepts = set(list_concepts)

# %%
set_resnet = set(df_clip_resnet_gpt_act40['description'])
#set_inception = set(df_clip_inception_gpt_act40['description'])

# %%
print('total concepts: {}'.format(len(set_concepts)))
print('concepts resnet: {}'.format(len(set_resnet)))
#print('concepts inception: {}'.format(len(set_inception)))

# %%
print('total concepts: {}'.format(len(set_concepts)))
print('concepts resnet: {}'.format(len(set_resnet)))
#print('concepts inception: {}'.format(len(set_inception)))

# %%
print('total concepts: {}'.format(len(set_concepts)))
print('concepts resnet: {}'.format(len(set_resnet)))
print('concepts inception: {}'.format(len(set_inception)))

# %%
#concepts not present in the resnet
print(len(set_concepts - set_resnet))
set_concepts - set_resnet

# %%
#concepts not present in the resnet - cubed
print(len(set_concepts - set_resnet))
set_concepts - set_resnet

# %%
#concepts not present in the inception
print(len(set_concepts - set_inception))
set_concepts - set_inception

# %%
#concepts only on resnet
print(len(set_resnet - set_inception))
set_resnet - set_inception

# %%
#concepts only on inception
print(len(set_inception - set_resnet))
set_inception - set_resnet

# %%
#overlapping concepts
print(len(set_resnet.intersection(set_inception)))
set_resnet.intersection(set_inception)

# %%
df_resnet_final.head()

# %%
df_resnet_v1_train = df_resnet_final_train.merge(df_clip_resnet_gpt_act40, left_on=['neuron_id'], right_on=['unit'])
df_resnet_v1_test = df_resnet_final_test.merge(df_clip_resnet_gpt_act40, left_on=['neuron_id'], right_on=['unit'])
#df_inception_v1 = df_inception_final.merge(df_clip_inception_gpt_act40, left_on=['value'], right_on=['unit'])

# %% [markdown]
# ![Alt text](image.png)

# %%
df_resnet_v1[df_resnet_v1['name']=='applauding_001.jpg'].sort_values('act_map',ascending=False).head(18)#[['name','class','description','similarity']]#.sort_values('similarity',ascending=False)

# %%
#cubed
df_resnet_v1[df_resnet_v1['name']=='applauding_001.jpg'].sort_values('act_map',ascending=False).head(10)#[['name','class','description','similarity']]#.sort_values('similarity',ascending=False)

# %%
#more concepts
df_resnet_v1[df_resnet_v1['name']=='applauding_001.jpg'].sort_values('act_map',ascending=False).head(10)#[['name','class','description','similarity']]#.sort_values('similarity',ascending=False)

# %%
for i,n in enumerate(df_resnet_v1[df_resnet_v1['name']=='applauding_001.jpg'].sort_values('act_map',ascending=False).head(10)[['name','neuron_id','description']].values):
    print('Top {} - {}'.format(i+1, n[2]))
    show_image('resnet152','ResNet_action40_last.pth','/mnt/data/netdissection/net-dissection-lite/dataset/Stanford40/JPEGImages/{}'.format(n[0]),n[1])

# %%
#cubed
for i,n in enumerate(df_resnet_v1[df_resnet_v1['name']=='applauding_001.jpg'].sort_values('act_map',ascending=False).head(10)[['name','neuron_id','description']].values):
    print('Top {} - {}'.format(i+1, n[2]))
    show_image('resnet152','ResNet_action40_last.pth','/mnt/data/netdissection/net-dissection-lite/dataset/Stanford40/JPEGImages/{}'.format(n[0]),n[1])

# %%
#more concepts
for i,n in enumerate(df_resnet_v1[df_resnet_v1['name']=='applauding_001.jpg'].sort_values('act_map',ascending=False).head(10)[['name','neuron_id','description']].values):
    print('Top {} - {}'.format(i+1, n[2]))
    show_image('resnet152','ResNet_action40_last.pth','/mnt/data/netdissection/net-dissection-lite/dataset/Stanford40/JPEGImages/{}'.format(n[0]),n[1])

# %%
df_inception_v1[df_inception_v1['name']=='applauding_001.jpg'].head(10)[['name','class','description','similarity']]#.sort_values('similarity',ascending=False)

# %% [markdown]
# ![Alt text](image-2.png)

# %%
df_resnet_v1[df_resnet_v1['name']=='taking_photos_005.jpg'].sort_values('act_map',ascending=False).head(10)#.head(10)[['name','class','description','similarity']]

# %%
for i,n in enumerate(df_resnet_v1[df_resnet_v1['name']=='taking_photos_005.jpg'].sort_values('act_map',ascending=False).head(10)[['name','neuron_id','description']].values):
    print('Top {} - {}'.format(i+1, n[2]))
    show_image('resnet152','ResNet_action40_last.pth','/mnt/data/netdissection/net-dissection-lite/dataset/Stanford40/JPEGImages/{}'.format(n[0]),n[1])

# %%
df_inception_v1[df_inception_v1['name']=='taking_photos_005.jpg'].head(10)[['name','class','description','similarity']]

# %% [markdown]
# ### Rating the concepts for each image and get MAX of the similarity

# %%
df_concept_resnet_max_train = df_resnet_v1_train.groupby(['description'])['similarity'].max().reset_index()
df_resnet_con_v1_train = df_resnet_v1_train.groupby(['name','class'])['description'].value_counts(normalize=True).reset_index()
#df_resnet_con_v1 = df_resnet_v1.groupby([df_resnet_v1['name'].rename('image'),df_resnet_v1['class'],df_resnet_v1['description'].rename('concept')]).aggregate({'description':'count','name':'size'}).reset_index()
df_resnet_con_v1_train = df_resnet_con_v1_train.merge(df_concept_resnet_max_train,how='left',on='description')
df_count_img_train = df_resnet_v1_train.groupby(['class'])['name'].unique().agg([len]).reset_index()
df_resnet_con_v1_train = df_resnet_con_v1_train.merge(df_count_img_train, how='left',on='class')



df_concept_resnet_max_test = df_resnet_v1_test.groupby(['description'])['similarity'].max().reset_index()
df_resnet_con_v1_test = df_resnet_v1_test.groupby(['name','class'])['description'].value_counts(normalize=True).reset_index()
#df_resnet_con_v1 = df_resnet_v1.groupby([df_resnet_v1['name'].rename('image'),df_resnet_v1['class'],df_resnet_v1['description'].rename('concept')]).aggregate({'description':'count','name':'size'}).reset_index()
df_resnet_con_v1_test = df_resnet_con_v1_test.merge(df_concept_resnet_max_test,how='left',on='description')
df_count_img_test = df_resnet_v1_test.groupby(['class'])['name'].unique().agg([len]).reset_index()
df_resnet_con_v1_test = df_resnet_con_v1_test.merge(df_count_img_test, how='left',on='class')

# %%
df_resnet_con_v1.head(18)

# %%
df_resnet_con_v1_train['ratio_weighted'] = df_resnet_con_v1_train['proportion'] * df_resnet_con_v1_train['similarity']

df_resnet_con_v1_test['ratio_weighted'] = df_resnet_con_v1_test['proportion'] * df_resnet_con_v1_test['similarity']

# %%
df_resnet_con_v1[df_resnet_con_v1['name']=='applauding_001.jpg']

# %%
df_resnet_con_v1[df_resnet_con_v1['name']=='taking_photos_005.jpg']

# %%
df_resnet_con_v1[(df_resnet_con_v1['description']=='a container with a spout') & (df_resnet_con_v1['class']=='applauding')]

# %%
a = df_resnet_con_v1.groupby([df_resnet_con_v1['class'],df_resnet_con_v1['description'].rename('concept')]).aggregate({'description':'count','proportion':['sum','mean'],'ratio_weighted':'sum','similarity':'max','len':'max'})
a.columns = ["_".join(x) for x in a.columns.ravel()]
a.reset_index(inplace=True)
a['description_rate_img'] = a['description_count']/a['len_max']
a['concept_similarity_prod'] = (a['description_rate_img']) * a['similarity_max']
#a['proportion_weighted_similarity'] = a['proportion_mean'] * a['similarity_norm_max']
#min_value = a.groupby('class')['ratio_weighted_sum'].min().reset_index()
#min_value.rename(columns={'ratio_weighted_sum':'ratio_weighted_sum_min'},inplace=True)
#max_value = a.groupby('class')['ratio_weighted_sum'].max().reset_index()
#max_value.rename(columns={'ratio_weighted_sum':'ratio_weighted_sum_max'},inplace=True)
#a = a.merge(min_value,on='class',how='left')
#a = a.merge(max_value,on='class',how='left')
#a['ratio_weighted_sum_norm'] = (a['ratio_weighted_sum'] - a['ratio_weighted_sum_min']) / (a['ratio_weighted_sum_max'] - a['ratio_weighted_sum_min'])
#a.drop(columns=['ratio_weighted_sum_min','ratio_weighted_sum_max'],inplace=True)

# %%
a[a['class']=='applauding'].sort_values('concept_similarity_prod', ascending=False).head(18)[['class','concept','description_rate_img','similarity_max','concept_similarity_prod']]

# %%
a[a['class']=='taking_photos'].sort_values('concept_similarity_prod', ascending=False).head(18)[['class','concept','description_rate_img','similarity_max','concept_similarity_prod']]

# %% [markdown]
# - description_count: The number of images that contains the unique concept (for a specific class)
# - proportion_sum: The sum of the proportion of the appearance of a concept for each image (the sum from the all class = 100) | = to description_count * proportion_mean
# - proportion_mean: The mean of the proportion of the appearance of a concept for each image
# - similarity_norm_max = The max similarity value for a specific concept learned from CLIP-Dissect
# - ratio_weighted_sum = The proportion_sum * similarity_norm_max
# - len_max = The total number of images for one class

# %%
df_resnet_v1_train.sort_values(['name','rank'])[['name','class','layer','neuron_id','act_map','description','similarity']].rename(columns={'description':'concept'}).to_csv('concept_list_action40_train_gpt4.csv')

df_resnet_v1_test.sort_values(['name','rank'])[['name','class','layer','neuron_id','act_map','description','similarity']].rename(columns={'description':'concept'}).to_csv('concept_list_action40_test_gpt4.csv')

# %% [markdown]
# ### Rating the concepts for each image and get SUM of the similarity

# %%
def rating_concepts(df_resnet_v1):
    df_resnet_con_v1 = pd.merge(df_resnet_v1.groupby([df_resnet_v1['name'],df_resnet_v1['class'],df_resnet_v1['description'].rename('concept')]).agg({'description':'count','similarity':'sum'}).reset_index(),
            df_resnet_v1.groupby(['name','class']).agg({'description':'count'}),how='left',on=['name','class'],suffixes=('_count','_total'))
    df_count_img = df_resnet_v1.groupby(['class'])['name'].unique().agg([len]).reset_index()
    df_resnet_con_v1['proportion'] = df_resnet_con_v1['description_count']/df_resnet_con_v1['description_total']
    df_resnet_con_v1 = df_resnet_con_v1.merge(df_count_img, how='left',on='class')
    df_resnet_con_v1.rename(columns={'len':'qtd_imgs_class'},inplace=True)
    df_resnet_con_v1['concept_similarity_prod'] = df_resnet_con_v1['proportion'] * df_resnet_con_v1['similarity']

    #df_concept_resnet_max = df_resnet_v1.groupby(['description'])['similarity'].max().reset_index()
    #df_resnet_con_v1 = df_resnet_con_v1.merge(df_concept_resnet_max,how='left',left_on='concept',right_on='description',suffixes=('_sum','_max'))

    # Encode the feature column
    label_encoder = LabelEncoder()
    df_resnet_con_v1['concept_encoded'] = label_encoder.fit_transform(df_resnet_con_v1['concept'])
    # Calculate mutual information for each unique value of the feature
    unique_values = df_resnet_con_v1['concept_encoded'].unique()
    mi_dict={}
    for value in unique_values:
        subset = df_resnet_con_v1[df_resnet_con_v1['concept_encoded'] == value]
        X = subset[['concept_encoded']]
        y = subset['class']
        try:
            mi = mutual_info_classif(X, y, random_state=7)
            mi_dict[label_encoder.inverse_transform([value])[0]] = mi[0]
        except:
            mi_dict[label_encoder.inverse_transform([value])[0]] = 0
    df_mi = pd.DataFrame(mi_dict,index=[0]).transpose().reset_index().rename(columns={'index':'concept',0:'mi'})
    df_mi['mi'] = df_mi['mi'].astype('float64')
    df_resnet_con_v1 = df_resnet_con_v1.merge(df_mi,on='concept',how='left')

    b = df_resnet_con_v1.groupby(['class',df_resnet_con_v1['concept'].rename('concept_name')]).agg({'concept':'count','proportion':'sum','similarity_sum':'mean','qtd_imgs_class':'max'}).reset_index()
    b = b.rename(columns={'concept':'concept_freq'})
    b['concept_rate_class'] = b['concept_freq']/b['qtd_imgs_class']
    b['concept_similarity_prod'] = b['concept_rate_class'] * b['similarity_sum']
    b = b.merge(df_mi,how='left',left_on='concept_name',right_on='concept')
    b['concept_similarity_mi_prod'] = b['concept_similarity_prod'] * b['mi']

    # Define a function to get top N items from each group
    def get_top_n_items(group, n=10):
        df = group.sort_values(by='concept_similarity_prod', ascending=False).head(n)
        return df.sort_values('mi',ascending=False)

    # Apply the function to each group
    c = b.groupby('class', group_keys=False).apply(get_top_n_items, n=10)

    return df_resnet_con_v1, b, c

# %%
df_resnet_con_v1, b_resnet, c_resnet = rating_concepts(df_resnet_v1)
df_inception_con_v1, b_inception, c_inception = rating_concepts(df_inception_v1)

# %%
df_resnet_con_v1[df_resnet_con_v1['name']=='applauding_001.jpg'].sort_values('concept_similarity_prod',ascending=False)[['name','class','concept','proportion','similarity_max','similarity_sum','concept_similarity_prod','mi']]

# %%
df_resnet_con_v1.sort_values(['name','concept_similarity_prod'],ascending=[True,False])[['name','class','concept','proportion','similarity_max','similarity_sum','concept_similarity_prod','mi']].to_csv('concepts_to_rule_learning_all.csv')

# %%
df_resnet_con_v1[df_resnet_con_v1['name']=='taking_photos_005.jpg'].sort_values('concept_similarity_prod',ascending=False)[['name','class','concept','proportion','similarity_max','similarity_sum','concept_similarity_prod','mi']]

# %%
df_resnet_con_v1[df_resnet_con_v1['concept']=='the bowstring pulled back']

# %%
b_resnet[b_resnet['class']=='applauding'].sort_values(['concept_similarity_prod'],ascending=False).head(10)[['class','concept_name','concept_rate_class','similarity_sum','similarity_max','concept_similarity_prod','mi']]

# %%
b_inception[b_inception['class']=='applauding'].sort_values(['concept_similarity_prod'],ascending=False).head(10)[['class','concept_name','concept_rate_class','similarity_sum','similarity_max','concept_similarity_prod','mi']]

# %%
b_resnet[b_resnet['class']=='phoning'].sort_values(['concept_similarity_prod'],ascending=False).head(10)[['class','concept_name','concept_rate_class','similarity_sum','similarity_max','concept_similarity_prod','mi']]

# %%
c_resnet[c_resnet['class']=='applauding'][['class','concept_name','concept_rate_class','similarity_sum','concept_similarity_prod','mi']]

# %%
c_inception[c_inception['class']=='applauding'][['class','concept_name','concept_rate_class','similarity_sum','concept_similarity_prod','mi']]

# %%
c_resnet[c_resnet['class']=='taking_photos'][['class','concept_name','concept_rate_class','similarity_sum','concept_similarity_prod','mi']]

# %%
c_resnet[c_resnet['class']=='phoning'][['class','concept_name','concept_rate_class','similarity_sum','concept_similarity_prod','mi']]

# %%
b[b['class']=='applauding'].sort_values(['concept_similarity_mi_prod'],ascending=False).head(10)[['class','concept_name','concept_rate_class','similarity_sum','similarity_max','concept_similarity_prod','mi','concept_similarity_mi_prod']]

# %%
b[b['class']=='taking_photos'].sort_values(['concept_similarity_prod'],ascending=False).head(10)[['class','concept_name','concept_rate_class','similarity_sum','similarity_max','concept_similarity_prod','mi']]

# %%
b[b['class']=='taking_photos'].sort_values(['concept_similarity_mi_prod'],ascending=False).head(10)[['class','concept_name','concept_rate_class','similarity_sum','similarity_max','concept_similarity_prod','mi','concept_similarity_mi_prod']]

# %% [markdown]
# 

# %%
df_concept_inception_max = df_inception_v1.groupby(['description'])['similarity_norm'].max().reset_index()
df_inception_con_v1 = df_inception_v1.groupby(['name','class'])['description'].value_counts(normalize=True).reset_index()
df_inception_con_v1 = df_inception_con_v1.merge(df_concept_inception_max,how='left',on='description')

# %%
def get_concept_ratio(df):
    a = df.groupby('class')['description'].value_counts(normalize=False).reset_index(name='freq')#.div(df.groupby('class')['description'].nunique()).reset_index(name='ratio_unique')
    b = df.groupby('class')['description'].value_counts(normalize=True).reset_index(name='ratio')
    da = pd.DataFrame(a).sort_values(['class'])
    db = pd.DataFrame(b).sort_values(['class'])
    d = da.merge(db, how='inner', on=['class','description'])
    return d

# %% [markdown]
# $$
# ConceptFreq(t_{i,class})=\sum_{n=1}^{N_{img}} t_{i}\
# $$
# 
# $$
# ConceptFreqRatio(t_{i,class})=\sum_{n=1}^{N_{img}}\frac{t_{i}}{t}\
# $$

# %%
def get_unit_ratio(df):
    a = df.groupby('class')['unit'].value_counts(normalize=False).reset_index(name='freq')#.div(df.groupby('class')['description'].nunique()).reset_index(name='ratio_unique')
    b = df.groupby('class')['unit'].value_counts(normalize=True).reset_index(name='ratio')
    da = pd.DataFrame(a).sort_values(['class'])
    db = pd.DataFrame(b).sort_values(['class'])
    d = da.merge(db, how='inner', on=['class','unit'])
    return d

# %% [markdown]
# $$
# ConceptFreq(k_{i,class})=\sum_{n=1}^{N_{img}} k_{i}\
# $$
# 
# $$
# ConceptFreqRatio(k_{i,class})=\sum_{n=1}^{N_{img}}\frac{k_{i}}{k}\
# $$

# %%
df_resnet_v1_c_ratio = get_concept_ratio(df_resnet_v1)
df_inception_v1_c_ratio = get_concept_ratio(df_inception_v1)

df_resnet_v1_u_ratio = get_unit_ratio(df_resnet_v1)
df_inception_v1_u_ratio = get_unit_ratio(df_inception_v1)

# %% [markdown]
# ### Dealing with the ratio based on the concept

# %%
df_resnet_v1_c_ratio[df_resnet_v1_c_ratio['class']=='applauding'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_resnet_concept_sim_mean = df_clip_resnet_gpt_act40.groupby(['description'])['similarity_norm'].mean().reset_index()

# %% [markdown]
# $$
# \bar{t_{i}}=\frac{\sum_{i=1}^{T}t_{i}}{T}=\frac{\sum_{i=1}^{T}ConceptFreqRatio(t_{i})}{T}\
# $$

# %%
df_resnet_concept_sim_mean[df_resnet_concept_sim_mean['description']=='two people with their arms raised']

# %%
df_resnet_v1_c_ratio = df_resnet_v1_c_ratio.merge(df_resnet_concept_sim_mean, how='inner', on='description')

# %%
df_resnet_v1_c_ratio['ratio_freq_weighted'] = df_resnet_v1_c_ratio['ratio'] * df_resnet_v1_c_ratio['similarity_norm']

# %%
df_resnet_v1_c_ratio[df_resnet_v1_c_ratio['class']=='applauding'].sort_values(['ratio_freq_weighted'],ascending=False).head(10)

# %% [markdown]
# ### Dealing with the ratio based on the neuron

# %%
df_resnet_v1_u_ratio[df_resnet_v1_u_ratio['class']=='applauding'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_resnet_v1_u_ratio = df_resnet_v1_u_ratio.merge(df_clip_resnet_gpt_act40,how='inner',on='unit')

# %%
df_resnet_v1_u_ratio['ratio_freq_weighted'] = df_resnet_v1_u_ratio['ratio'] * df_resnet_v1_u_ratio['similarity_norm']

# %%
df_resnet_v1_u_ratio[df_resnet_v1_u_ratio['class']=='applauding'].sort_values(['ratio_freq_weighted'],ascending=False).drop_duplicates('description').head(10)

# %%
#try the sum the same concept
df_resnet_v1_u_ratio[df_resnet_v1_u_ratio['class']=='applauding'].groupby(['description'])['ratio_freq_weighted'].mean().reset_index().sort_values(['ratio_freq_weighted'],ascending=False).head(10)

# %% [markdown]
# ### Dealing directly to the activation_map

# %% [markdown]
# $$
# A_{k}(x_{i})*sim(t_{m},q_{k};P)\
# $$

# %%
dt_avg_resnet.head()

# %%
df_clip_resnet_gpt_act40.head()

# %%
dt_avg_resnet_ = dt_avg_resnet.iloc[:,:-3].transpose().reset_index()
dt_avg_resnet_['index']=dt_avg_resnet_['index'].astype('int64')

result_merge = dt_avg_resnet_.merge(df_clip_resnet_gpt_act40,right_on='unit',left_on='index')

# %%
result_merge.head()

# %%
result_merge = result_merge.iloc[:,1:-5].mul(result_merge['similarity_norm'], axis=0)
result_merge.index = [i+1 for i in range(len(result_merge))]
result_merge = result_merge.transpose()
dt_avg_resnet_merged = result_merge.merge(dt_avg_resnet[['name','class']],right_index=True,left_index=True)
dt_avg_resnet_merged['top_feature'] = dt_avg_resnet_merged.iloc[:, :-2].apply(top_n_columns, axis=1)

# %%
dt_avg_resnet_merged.head()

# %%
df_resnet_weighted=dt_avg_resnet_merged[['name','class','top_feature']]
new_resnet_w = df_resnet_weighted['top_feature'].apply(lambda x: pd.Series(x))
result_df_resnet_weighted = pd.concat([df_resnet_weighted, new_resnet_w], axis=1)
result_df_resnet_weighted.drop('top_feature', axis=1, inplace=True)
df_resnet_final_w = pd.melt(result_df_resnet_weighted,id_vars=['name','class']).sort_values('name')
df_resnet_final_w['value'] = df_resnet_final_w['value'].astype('Int64')
df_resnet_v1_w = df_resnet_final_w.merge(df_clip_resnet_gpt_act40, left_on=['value'], right_on=['unit'])

# %%
df_resnet_v1_w.sort_values(['name']).head(10)

# %%
df_resnet_v1_w_c_ratio = get_concept_ratio(df_resnet_v1_w)
df_resnet_v1_w_u_ratio = get_unit_ratio(df_resnet_v1_w)

# %%
df_resnet_v1_w_c_ratio[df_resnet_v1_w_c_ratio['class']=='applauding'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_resnet_v1_w_c_ratio_ = df_resnet_v1_w_c_ratio.merge(df_resnet_concept_sim_mean, on='description',how='left')

# %%
df_resnet_v1_w_c_ratio_[df_resnet_v1_w_c_ratio_['class']=='applauding'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_resnet_v1_w_u_ratio[df_resnet_v1_w_u_ratio['class']=='applauding'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_resnet_v1_w_u_ratio = df_resnet_v1_w_u_ratio.merge(df_clip_resnet_gpt_act40,how='inner',on='unit')
#df_resnet_v1_w_u_ratio['ratio_weighted'] = df_resnet_v1_w_u_ratio['ratio'] * df_resnet_v1_w_u_ratio['similarity_norm']

# %%
df_resnet_v1_w_u_ratio[df_resnet_v1_w_u_ratio['class']=='applauding'].sort_values(['similarity_norm'],ascending=False).drop_duplicates('description').head(10)

# %%
df_resnet_v1_w_u_ratio[df_resnet_v1_w_u_ratio['class']=='applauding'].sort_values(['similarity_norm'],ascending=False).head(10)

# %% [markdown]
# So what?

# %%


# %% [markdown]
# # Looking after

# %%
df_clip_resnet_gpt_act40[df_clip_resnet_gpt_act40['description']=='the person is wearing a helmet']

# %%
df_resnet_v1[(df_resnet_v1['class']=='riding_a_bike') & (df_resnet_v1['description']=='the person is wearing a helmet')]['unit'].value_counts()

# %%
df_inception_v1_ration[df_inception_v1_ration['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %% [markdown]
# ### Analysis - Concepts from GPT3 and 20k; Action40+Imagenet Images: V2

# %%
plot_charts(df_clip_resnet_gpt_20k_act40_img, df_clip_inception_gpt_20k_act40_img)

# %%
df_clip_resnet_gpt_20k_act40_img.sort_values(['similarity'],ascending=False).head(10)

# %%
df_clip_inception_gpt_20k_act40_img.sort_values(['similarity'],ascending=False).head(10)

# %%
df_resnet_v2 = df_resnet_final.merge(df_clip_resnet_gpt_20k_act40_img, left_on=['value'], right_on=['unit'])
df_inception_v2 = df_inception_final.merge(df_clip_inception_gpt_20k_act40_img, left_on=['value'], right_on=['unit'])

# %%
df_resnet_v2_ratio = get_concept_ratio(df_resnet_v2)
df_inception_v2_ration = get_concept_ratio(df_inception_v2)

# %%
df_resnet_v2_ratio[df_resnet_v2_ratio['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_inception_v2_ration[df_inception_v2_ration['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_clip_resnet_gpt_20k_act40_img[df_clip_resnet_gpt_20k_act40_img['description']=='bicycles']

# %% [markdown]
# ### Analysis - Concepts from GPT3 and Imagenet; Action40+Imagenet Images: V3

# %%
plot_charts(df_clip_resnet_gpt_img_act40_img, df_clip_inception_gpt_img_act40_img)

# %%
df_clip_resnet_gpt_img_act40_img.sort_values(['similarity'],ascending=False).head(10)

# %%
df_clip_inception_gpt_img_act40_img.sort_values(['similarity'],ascending=False).head(10)

# %%
df_resnet_v3 = df_resnet_final.merge(df_clip_resnet_gpt_img_act40_img, left_on=['value'], right_on=['unit'])
df_inception_v3 = df_inception_final.merge(df_clip_inception_gpt_img_act40_img, left_on=['value'], right_on=['unit'])

# %%
df_resnet_v3_ratio = get_concept_ratio(df_resnet_v3)
df_inception_v3_ration = get_concept_ratio(df_inception_v3)

# %%
df_resnet_v3_ratio[df_resnet_v3_ratio['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_inception_v3_ration[df_inception_v3_ration['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %% [markdown]
# ### Analysis - Concepts from GPT3+Imagenet and Action40 Images: V4

# %%
plot_charts(df_clip_resnet_gpt_img_act40,df_clip_inception_gpt_img_act40)

# %%
df_clip_resnet_gpt_img_act40.sort_values(['similarity'],ascending=False).head(10)

# %%
df_clip_inception_gpt_img_act40.sort_values(['similarity'],ascending=False).head(10)

# %%
df_resnet_v4 = df_resnet_final.merge(df_clip_resnet_gpt_img_act40, left_on=['value'], right_on=['unit'])
df_inception_v4 = df_inception_final.merge(df_clip_inception_gpt_img_act40, left_on=['value'], right_on=['unit'])

# %%
df_resnet_v4_ratio = get_concept_ratio(df_resnet_v4)
df_inception_v4_ration = get_concept_ratio(df_inception_v4)

# %%
df_resnet_v4_ratio[df_resnet_v4_ratio['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_inception_v4_ration[df_inception_v4_ration['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %% [markdown]
# ### Analysis - Concepts from GPT3 and Action40+Imagenet+Broden Images: V5

# %%
plot_charts(df_clip_resnet_gpt_img_broden_act40, df_clip_inception_gpt_img_broden_act40)

# %%
df_clip_resnet_gpt_img_broden_act40.sort_values(['similarity'],ascending=False).head(10)

# %%
df_clip_inception_gpt_img_broden_act40.sort_values(['similarity'],ascending=False).head(10)

# %%
df_resnet_v5 = df_resnet_final.merge(df_clip_resnet_gpt_img_broden_act40, left_on=['value'], right_on=['unit'])
df_inception_v5 = df_inception_final.merge(df_clip_inception_gpt_img_broden_act40, left_on=['value'], right_on=['unit'])

# %%
df_resnet_v5_ratio = get_concept_ratio(df_resnet_v5)
df_inception_v5_ration = get_concept_ratio(df_inception_v5)

# %%
df_resnet_v5_ratio[df_resnet_v5_ratio['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_inception_v5_ration[df_inception_v5_ration['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %% [markdown]
# ### Analysis - Concepts from Imagenet+Broden and Imagenet+Broden Images: V6

# %%
plot_charts(df_clip_resnet_img_broden,df_clip_inception_img_broden)

# %%
df_clip_resnet_img_broden.sort_values(['similarity'],ascending=False).head(10)

# %%
df_clip_inception_img_broden.sort_values(['similarity'],ascending=False).head(10)

# %%
df_resnet_v6 = df_resnet_final.merge(df_clip_resnet_img_broden, left_on=['value'], right_on=['unit'])
df_inception_v6 = df_inception_final.merge(df_clip_inception_img_broden, left_on=['value'], right_on=['unit'])

# %%
df_resnet_v6_ratio = get_concept_ratio(df_resnet_v6)
df_inception_v6_ration = get_concept_ratio(df_inception_v6)

# %%
df_resnet_v6_ratio[df_resnet_v6_ratio['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)

# %%
df_inception_v6_ration[df_inception_v6_ration['class']=='riding_a_bike'].sort_values(['ratio'],ascending=False).head(10)


