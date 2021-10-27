import json
from PIL import Image
from src.resnet_model import ResnetModel

def test_predict():
    model = ResnetModel(
        path_to_pretrained_model='../models/trained_model_resnet50.pt')
    with open('index_to_class_label.json', 'rb') as f:
        index_to_class_labels = json.load(f)
    index_to_class_labels = {int(k): v for k, v in index_to_class_labels.items()}
    img = Image.open(
        '/Users/josh-mantovani/Downloads/archive/train/AFRICAN CROWNED CRANE/001.jpg')
    print(model.predict_proba(img, 3, index_to_class_labels, show=False))
    assert True