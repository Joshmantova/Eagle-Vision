import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json
import numpy as np
import boto3
from io import BytesIO
import time

@st.cache()
def predict(img, index_to_label_dict, device='cpu'):
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

    img_t = transform(img)
    img_t = torch.unsqueeze(img_t, 0)
    img_t = img_t.to(device)

    model.eval() #putting model in eval mode
    output_tensor = model(img_t) #predicting
    prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)
    top_3 = torch.topk(prob_tensor, 3, dim=1)
    probabilites = top_3.values.detach().numpy().flatten()
    indices = top_3.indices.detach().numpy().flatten()
    out = []
    for pred_prob, pred_idx in zip(probabilites, indices):
        predicted_label = index_to_label_dict[pred_idx].title()
        predicted_prob = pred_prob * 100
        out.append((predicted_label, f"{predicted_prob:.3f}%"))
    return out

@st.cache()
def load_model(path='trained_model_resnet50.pt', device='cpu'):
    return torch.load(path, map_location=device)

@st.cache()
def load_index_to_label_dict(path='index_to_class_label.json'):
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f) #loads keys in as strings, need to convert to ints
    index_to_class_label_dict = {int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict

@st.cache()
def load_file_from_s3(key, bucket_name='bird-classification-bucket'):
    s3 = boto3.client('s3')
    s3_file_raw = s3.get_object(Bucket=bucket_name,
                            Key=key)
    s3_file = s3_file_raw['Body'].read()
    return s3_file

model = load_model()
index_to_class_label_dict = load_index_to_label_dict()

image_choices = {
    'Albatross': 'train/ALBATROSS/001.jpg',
    'Blue Grouse': 'train/BLUE GROUSE/001.jpg'
}

file = st.file_uploader('Upload image')
if not file:
    choice = st.sidebar.selectbox('Select example bird: ', list(image_choices.keys()))
    # s3 = boto3.client('s3')
    # s3_file = s3.get_object(Bucket='bird-classification-bucket', Key=image_choices[choice])['Body'].read()
    s3_file = load_file_from_s3(key=image_choices[choice])
    img = Image.open(BytesIO(s3_file))

else:
    img = Image.open(file)

prediction = predict(img, index_to_class_label_dict)
st.image(img)
displayed_prediction = str()
for idx, p in enumerate(prediction, start=1):
    st.write(f"Top {idx} prediction: {p[0]}, Confidence level: {p[1]}")
