import json
from io import BytesIO
from PIL import Image
import os

import boto3
from botocore import UNSIGNED  # contact public s3 buckets anonymously
from botocore.client import Config  # contact public s3 buckets anonymously

import streamlit as st
import pandas as pd
import numpy as np

from resnet_model import ResnetModel


@st.cache()
def load_model(path: str = 'models/trained_model_resnet50.pt') -> ResnetModel:
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    model = ResnetModel(path_to_pretrained_model=path)
    return model


@st.cache()
def load_index_to_label_dict(
        path: str = 'src/index_to_class_label.json'
        ) -> dict:
    """Retrieves and formats the
    index to class label
    lookup dictionary needed to
    make sense of the predictions.
    When loaded in, the keys are strings, this also
    processes those keys to integers."""
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {
        int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict


def load_files_from_s3(
        keys: list,
        bucket_name: str = 'bird-classification-bucket'
        ) -> list:
    """Retrieves files anonymously from my public S3 bucket"""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_files = []
    for key in keys:
        s3_file_raw = s3.get_object(Bucket=bucket_name, Key=key)
        s3_file_cleaned = s3_file_raw['Body'].read()
        s3_file_image = Image.open(BytesIO(s3_file_cleaned))
        s3_files.append(s3_file_image)
    return s3_files


@st.cache()
def load_s3_file_structure(path: str = 'src/all_image_files.json') -> dict:
    """Retrieves JSON document outining the S3 file structure"""
    with open(path, 'r') as f:
        return json.load(f)


@st.cache()
def load_list_of_images_available(
        all_image_files: dict,
        image_files_dtype: str,
        bird_species: str
        ) -> list:
    """Retrieves list of available images given the current selections"""
    species_dict = all_image_files.get(image_files_dtype)
    list_of_files = species_dict.get(bird_species)
    return list_of_files


@st.cache()
def predict(
        img: Image.Image,
        index_to_label_dict: dict,
        model,
        k: int
        ) -> list:
    """Transforming input image according to ImageNet paper
    The Resnet was initially trained on ImageNet dataset
    and because of the use of transfer learning, I froze all
    weights and only learned weights on the final layer.
    The weights of the first layer are still what was
    used in the ImageNet paper and we need to process
    the new images just like they did.

    This function transforms the image accordingly,
    puts it to the necessary device (cpu by default here),
    feeds the image through the model getting the output tensor,
    converts that output tensor to probabilities using Softmax,
    and then extracts and formats the top k predictions."""
    formatted_predictions = model.predict_proba(img, k, index_to_label_dict)
    return formatted_predictions


if __name__ == '__main__':
    model = load_model()
    index_to_class_label_dict = load_index_to_label_dict()
    all_image_files = load_s3_file_structure()
    types_of_birds = sorted(list(all_image_files['test'].keys()))
    types_of_birds = [bird.title() for bird in types_of_birds]

    st.title('Welcome To Project Eagle Vision!')
    instructions = """
        Either upload your own image or select from
        the sidebar to get a preconfigured image.
        The image you select or upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')
    dtype_file_structure_mapping = {
        'All Images': 'consolidated',
        'Images Used To Train The Model': 'train',
        'Images Used To Tune The Model': 'valid',
        'Images The Model Has Never Seen': 'test'
    }
    data_split_names = list(dtype_file_structure_mapping.keys())

    if file:  # if user uploaded file
        img = Image.open(file)
        prediction = predict(img, index_to_class_label_dict, model, k=5)
        top_prediction = prediction[0][0]
        available_images = all_image_files.get(
            'train').get(top_prediction.upper())
        examples_of_species = np.random.choice(available_images, size=3)
        files_to_get_from_s3 = []

        for im_name in examples_of_species:
            path = os.path.join('train', top_prediction.upper(), im_name)
            files_to_get_from_s3.append(path)
        images_from_s3 = load_files_from_s3(keys=files_to_get_from_s3)

    else:
        dataset_type = st.sidebar.selectbox(
            "Data Portion Type", data_split_names)
        image_files_subset = dtype_file_structure_mapping[dataset_type]

        selected_species = st.sidebar.selectbox("Bird Type", types_of_birds)
        available_images = load_list_of_images_available(
            all_image_files, image_files_subset, selected_species.upper())
        image_name = st.sidebar.selectbox("Image Name", available_images)
        if image_files_subset == 'consolidated':
            s3_key_prefix = 'consolidated/consolidated'
        else:
            s3_key_prefix = image_files_subset
        key_path = os.path.join(
            s3_key_prefix, selected_species.upper(), image_name)
        files_to_get_from_s3 = [key_path]
        examples_of_species = np.random.choice(available_images, size=3)

        for im in examples_of_species:
            path = os.path.join(s3_key_prefix, selected_species.upper(), im)
            files_to_get_from_s3.append(path)
        images_from_s3 = load_files_from_s3(keys=files_to_get_from_s3)
        img = images_from_s3.pop(0)
        prediction = predict(img, index_to_class_label_dict, model, 5)

    st.title("Here is the image you've selected")
    resized_image = img.resize((336, 336))
    st.image(resized_image)
    st.title("Here are the five most likely bird species")
    df = pd.DataFrame(data=np.zeros((5, 2)),
                      columns=['Species', 'Confidence Level'],
                      index=np.linspace(1, 5, 5, dtype=int))

    for idx, p in enumerate(prediction):
        link = 'https://en.wikipedia.org/wiki/' + \
            p[0].lower().replace(' ', '_')
        df.iloc[idx,
                0] = f'<a href="{link}" target="_blank">{p[0].title()}</a>'
        df.iloc[idx, 1] = p[1]
    st.write(df.to_html(escape=False), unsafe_allow_html=True)
    st.title(f"Here are three other images of the {prediction[0][0]}")

    st.image(images_from_s3)
    # st.title('How it works:')
