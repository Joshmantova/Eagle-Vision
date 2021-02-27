import json #to work with my json files
from io import BytesIO #to convert the s3 images to play nice with PIL
from PIL import Image #to work with images
import os

import boto3 #s3 interactions
from botocore import UNSIGNED #contact public s3 buckets anonymously
from botocore.client import Config #contact public s3 buckets anonymously

import torch #deep learning framework of choice
from torchvision import transforms #helpful transformations to work with images in torch

import streamlit as st #easy library to create data apps

@st.cache()
def predict(img, index_to_label_dict, model, device='cpu'):
    #transforming input image according to ImageNet paper
    #The ResNet was initially trained on ImageNet dataset
    #Need to transform it like they did
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    #Transforming the image to play nice with torch
    img_t = transform(img)
    img_t = torch.unsqueeze(img_t, 0)
    img_t = img_t.to(device)

    model.eval() #putting model in eval mode
    output_tensor = model(img_t) #predicting
    prob_tensor = torch.nn.Softmax(dim=1)(output_tensor) #converting output tensor to probabilities
    top_3 = torch.topk(prob_tensor, 3, dim=1) #getting the top k probabilities and indices
    probabilites = top_3.values.detach().numpy().flatten() #converting to friendly numpy array
    indices = top_3.indices.detach().numpy().flatten() #converting to friendly numpy array
    formatted_predictions = []
    # going through each prediction and formatting them
    for pred_prob, pred_idx in zip(probabilites, indices):
        predicted_label = index_to_label_dict[pred_idx].title()
        predicted_prob = pred_prob * 100
        formatted_predictions.append((predicted_label, f"{predicted_prob:.3f}%"))
    return formatted_predictions

@st.cache()
def load_model(path='../models/trained_model_resnet50.pt', device='cpu'):
    #retrieves trained model
    return torch.load(path, map_location=device)

@st.cache()
def load_index_to_label_dict(path='index_to_class_label.json'):
    #retrives and formats index to class label lookup dictionary
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f) #loads keys in as strings, need to convert to ints
    index_to_class_label_dict = {int(k): v for k, v in index_to_class_label_dict.items()} #converting keys to ints
    return index_to_class_label_dict

@st.cache()
def load_file_from_s3(key, bucket_name='bird-classification-bucket'):
    #retrieves files from S3 anonymously 
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_file_raw = s3.get_object(Bucket=bucket_name,
                            Key=key)
    s3_file = s3_file_raw['Body'].read()
    return s3_file

@st.cache()
def load_all_image_files(path='all_image_files.json'):
    #retrieves json document outlining S3 file structure
    with open(path, 'r') as f:
        return json.load(f)

@st.cache()
def load_list_of_images_available(all_image_files, image_files_dtype, bird_species):
    #retrieves list of available images given the current selections
    species_dict = all_image_files.get(image_files_dtype)
    list_of_files = species_dict.get(bird_species)
    return list_of_files

if __name__ == '__main__':
    model = load_model()
    index_to_class_label_dict = load_index_to_label_dict()
    all_image_files = load_all_image_files()
    types_of_birds = sorted(list(all_image_files['test'].keys())) #alphabetically sorting all the species

    file = st.file_uploader('Upload An Image')
    dtype_file_structure_mapping = {
        'All Images': 'consolidated', 'Images Used To Train The Model': 'train', 
        'Images Used To Tune The Model': 'valid', 'Images The Model Has Never Seen': 'test'
        } #Needed to map the interpretable displayed option to split type in order to index available images

    if not file: #if there's no file uploaded, display preset images to choose from
        dataset_type = st.sidebar.selectbox("Data Portion Type", list(dtype_file_structure_mapping.keys()))
        #getting the split type from user selection to index list of available images
        image_files_dtype = dtype_file_structure_mapping[dataset_type]

        bird_species = st.sidebar.selectbox("Bird Type", types_of_birds)
        available_images = load_list_of_images_available(all_image_files, image_files_dtype, bird_species)
        image_name = st.sidebar.selectbox("Image Name", available_images)
        #S3 file structure is a little strange so this is necessary
        #consolidated has a nested and redundant folder
        if image_files_dtype == 'consolidated':
            s3_key_prefix = 'consolidated/consolidated'
        else:
            s3_key_prefix = image_files_dtype
        key_path = os.path.join(s3_key_prefix, bird_species, image_name)
        s3_file = load_file_from_s3(key=key_path)
        
        img = Image.open(BytesIO(s3_file)) #open the image from S3

    else: #if there is a file uploaded, just open it
        img = Image.open(file)

    #with whatever file was opened, use it to predict
    prediction = predict(img, index_to_class_label_dict, model)
    st.image(img) #show the image
    for idx, p in enumerate(prediction, start=1): #display formatted prediction and confidence level
        st.write(f"Top {idx} prediction: {p[0]}, Confidence level: {p[1]}")
