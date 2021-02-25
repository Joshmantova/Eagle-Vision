import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json
import numpy as np

def predict(img):
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
    batch_t = torch.unsqueeze(img_t, 0)

    model.eval()
    out = model(batch_t)
    _, index_out = torch.max(out, 1)
    index = index_out.item()
    label = index_to_label_dict[index]
    return label.title()

with open('../downloads/model/class_label_to_prediction_index.json', 'rb') as f:
    label_to_index_dict = json.load(f)

index_to_label_dict = {v: k for k, v in label_to_index_dict.items()}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('../downloads/trained_model_resnet50.pt', map_location=device)

image_choices = {
    'Bald Eagle': '../downloads/model/Petrusich-Dont-Mess-with-the-Birds.jpg',
    'African Crowned Crane': '../downloads/model/001.jpg'
}

choice = st.sidebar.selectbox('Select example bird: ', list(image_choices.keys()))
# img = Image.open(image_choices[choice])
# img = Image.open('../downloads/model/Petrusich-Dont-Mess-with-the-Birds.jpg')

file = st.file_uploader('Upload image')
if not file:
    img = Image.open(image_choices[choice])

else:
    img = Image.open(file)

prediction = predict(img)
st.image(img)
st.write(prediction)
