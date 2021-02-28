import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
from torch import (nn,
                    cuda,
                    optim,
                    device)
from torchvision import (models,
                        transforms,
                        datasets)
from torch.utils.data import DataLoader
from PIL import Image
import boto3

class Resnet_Model:

    def __init__(self, path_to_pretrained_model=None, map_location='CPU', num_classes=250):
        self.device = torch.device('cuda:0' if cuda.is_available() else 'cpu')
        if path_to_pretrained_model:
            self.model = torch.load(path_to_pretrained_model, map_location=map_location)
        else:
            self.model = self._setup_resnet(num_classes=250)

        self.train_transform, self.val_transform, self.test_transform = self._setup_transform()

    def feed_forward(self, model, inputs):
        return model(inputs)

    def fit(self, train_loader, val_loader, num_epochs=10, criterion=None, 
            optimizer=None, batch_size=16, early_stop_min_increase=0.003, 
            early_stop_patience=10, lr=0.0001):
        start = time.time()
        model = self.model
        best_model = self.model.state_dict()
        best_acc = 0
        train_loss_over_time = []
        val_loss_over_time = []
        train_acc_over_time = []
        val_acc_over_time = []
        epochs_no_improve = 0
        early_stop = False
        phases = ['train', 'val']

        if not optimizer:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if not criterion:
            criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            print(f"Epoch number: {epoch + 1} / {num_epochs}")

            for phase in phases:
                if phase == 'train':
                    data_loader = train_loader
                    model.train()
                else:
                    data_loader = val_loader
                    model.eval()

                running_loss = 0
                running_corrects = 0

                for inputs, labels in data_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.feed_forward(model, inputs)
                        _, pred = torch.max(outputs, dim=1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(pred == labels.data)

                if phase == 'train':
                    epoch_loss = running_loss / len(train_loader.dataset)
                    train_loss_over_time.append(epoch_loss)
                    epoch_acc = running_corrects.double() / len(train_loader.dataset)
                    train_acc_over_time.append(epoch_acc)

                else:
                    epoch_loss = running_loss / len(val_loader.dataset)
                    val_loss_over_time.append(epoch_loss)
                    epoch_acc = running_corrects.double() / len(val_loader.dataset)
                    val_acc_over_time.append(epoch_acc)

                print(f"{phase} loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}")

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = model.state_dict()
                    torch.save(model, 'trained_model_resnet50_checkpoint.pt')
                    epochs_no_improve = 0

                elif phase == 'val' and (epoch_acc - best_acc) < early_stop_min_increase:
                    epochs_no_improve += 1
                    print(f"Number of epochs without improvement has increased to {epochs_no_improve}")

            if epochs_no_improve >= early_stop_patience:
                early_stop = True
                print('Early stopping!')
                break

            print('-' * 60)
            total_time = (time.time() - start) / 60
            print(f"Training completed. Time taken: {total_time:.3f} min\nBest accuracy: {best_acc:.3f}")
            model.load_state_dict(best_model)
            loss = {'train': train_loss_over_time, 'val': val_loss_over_time}
            acc = {'train': train_acc_over_time, 'val': val_acc_over_time}
            
            return model, loss, acc

    def predict_proba(self, k):
        pass

    def _setup_resnet(self, num_classes):
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.require_grad = False

        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.30),
                                )
        model.to(self.device)
        return model

    def _setup_transform(self):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds)
        ])

        return (train_transform, val_transform, test_transform)


#NOTE: this script is basically what was used to train. I did not train using
#this exact script - it was trained on AWS using a sagemaker notebook
#This script has been extracted from that notebook for transparancy purposes.

def fit(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, batch_size=16):
    start = time.time()
    best_model = model.state_dict()
    best_acc = 0
    train_loss_over_time = []
    val_loss_over_time = []
    train_acc_over_time = []
    val_acc_over_time = []
    early_stop_min_increase = 0.0001
    early_stop_patience = 10
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        print(f"Epoch number: {epoch+1}/{num_epochs}")
        
        for phase in ['train', 'val']:
            if phase=='train':
                data_loader = train_loader
                model.train()
            else:
                data_loader = val_loader
                model.eval()
                
            running_loss = 0
            running_corrects = 0
            
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)
                    
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred==labels.data)
                        
            if phase=='train':
                epoch_loss = running_loss/len(train_loader.dataset)
                train_loss_over_time.append(epoch_loss)
                epoch_acc = running_corrects.double()/len(train_loader.dataset)
                train_acc_over_time.append(epoch_acc)

            else:
                epoch_loss = running_loss/len(val_loader.dataset)
                val_loss_over_time.append(epoch_loss)
                epoch_acc = running_corrects.double()/len(val_loader.dataset)
                val_acc_over_time.append(epoch_acc)

            print(f"{phase} loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}")
                
            if phase=='val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = model.state_dict()
                torch.save(model, 'trained_model_resnet50_checkpoint.pt') #checkpoint
                epochs_no_improve = 0
            elif phase == 'val' and epoch_acc < best_acc:
                epochs_no_improve += 1
                print(f"Number of epochs without improvement has increased to {epochs_no_improve}")
        if epochs_no_improve >= early_stop_patience:
            early_stop = True
        if early_stop:
            print('Early stopping!')
            break
        print('-'*60)
    total_time = (time.time() - start)/60
    print(f"Training completed. Time taken: {total_time:.3f} min\nBest accuracy: {best_acc:.3f}")
    model.load_state_dict(best_model)
    loss = {'train': train_loss_over_time,
           'val': val_loss_over_time}
    acc = {'train': train_acc_over_time,
          'val': val_acc_over_time}
    return model, loss, acc


def evaluate(test_loader, model, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    preds = list()
    labels_list = list()
    
    for inputs, labels in test_loader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs, dim=1)
            preds.append(pred)
            labels_list.append(labels)
            
        test_loss += loss.item()*inputs.size(0)
        correct = pred.eq(labels.data.view_as(pred))
        accuracy = torch.mean(correct.type(torch.FloatTensor))
        test_acc += accuracy.item() * inputs.size(0)
        
    test_loss = test_loss/len(test_loader.dataset)
    test_acc = test_acc / len(test_loader.dataset)
        
    print(f"Test loss: {test_loss:.4f}\nTest acc: {test_acc:.4f}")
    
    return preds, labels_list

def predict(image_filepath, model, index_to_class_labels, show=True):
    img = Image.open(image_filepath)
    if show:
        display(img)
    img_t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    ])
    img = img_t(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    model.eval()
    output_tensor = model(img)
#     _, pred = torch.max(outputs, dim=1)
    prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)
    top_5 = torch.topk(prob_tensor, 3, dim=1)
    out = []
    for pred_prob, pred_idx in zip(top_5.values.detach().numpy().flatten(), top_5.indices.detach().numpy().flatten()):
        predicted_label = sorted_list_of_dirs[pred_idx]
        predicted_prob = pred_prob*100
        out.append((predicted_label, str(round(predicted_prob, 3))+'%'))
    return out

if __name__ == '__main__':
        
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.require_grad = False
        
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.30),
                            nn.Linear(1024, 250))

    device = torch.device('cuda:0' if cuda.is_available() else 'cpu')

    model.to(device)
    print(device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder('train')
    val_data = datasets.ImageFolder('valid')
    test_data = datasets.ImageFolder('test')

    train_data.transform = train_transform
    val_data.transform = val_transform
    test_data.transform = test_transform

    batch_size=16

    train_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True)
    test_loader = DataLoader(test_data,
                            batch_size=batch_size,
                            shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                        lr=0.0001)

    history, loss, acc = fit(train_loader, val_loader, model, criterion, optimizer,
                            num_epochs=100, batch_size=batch_size)
    predictions, labels = evaluate(val_loader, model, criterion)

    torch.save(mode, 'trained_model_resnet50.pt')