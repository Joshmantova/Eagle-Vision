import json
from PIL import Image
import boto3
from botocore import UNSIGNED  # contact public s3 buckets anonymously
from botocore.client import Config  # contact public s3 buckets anonymously
from io import BytesIO

from src.resnet_model import ResnetModel


def load_files_from_s3(
        keys,
        bucket_name='bird-classification-bucket'
        ):
    """Retrieves files anonymously from my public S3 bucket"""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3_files = []
    for key in keys:
        s3_file_raw = s3.get_object(Bucket=bucket_name, Key=key)
        s3_file_cleaned = s3_file_raw['Body'].read()
        s3_file_image = Image.open(BytesIO(s3_file_cleaned))
        s3_files.append(s3_file_image)
    return s3_files


def test_s3_connection():
    s3_files = load_files_from_s3(["train/AFRICAN CROWNED CRANE/001.jpg"])
    assert s3_files


def test_predict():
    model = ResnetModel(
        path_to_pretrained_model='models/trained_model_resnet50.pt')
    with open('src/index_to_class_label.json', 'rb') as f:
        index_to_class_labels = json.load(f)
    index_to_class_labels = {
        int(k): v for k, v in index_to_class_labels.items()}
    img = load_files_from_s3(["train/AFRICAN CROWNED CRANE/001.jpg"])[0]
    formatted_prediction = model.predict_proba(
        img,
        3,
        index_to_class_labels,
        show=False
        )
    assert formatted_prediction[0][0] == "African Crowned Crane"
