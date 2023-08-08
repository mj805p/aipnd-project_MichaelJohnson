import argparse

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import helper
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import random
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['vgg_type'] == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif checkpoint['vgg_type'] == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)

    return model, checkpoint['class_to_idx']

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    #Resize image
    pil_image = pil_image.resize((224, 224))
    
    #Crop image
    # According to documentation, "left, upper" is calculated as (256-224)/2 = 16 pixels
    # According to documentation, "right, lower" is calculated as "left + width" and "upper + height" or 224+16 = 240 pixels
    pil_image = pil_image.crop((16, 16, 240, 240))
    
    #Encode color channels
    np_image = np.array(pil_image)/255
    
    #Normalize
    mean = np.array([0.485, 0.456, 0.406])
    st_dev = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/st_dev
    
    #Reorder Dimensions
    np_image = np_image.transpose(2, 0, 1)
    
    #Convert from np to Tensor
    tensor_image = torch.from_numpy(np_image)
    tensor_image = tensor_image.type(torch.FloatTensor)
   
    return tensor_image

def predict(image_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    model.to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.exp(model(image))
        
    
    top_probs, top_classes = probs.topk(5, dim=1)

    return top_probs, top_classes

if __name__ == '__main__':
    #Set up command line inputs ie. python train.py data_dir --gpu
    parser = argparse.ArgumentParser(
        description='predict classes')

    parser.add_argument('image_path', type = str, default = "./flowers/test/1/image_06743.jpg", help = 'directory to images')
    parser.add_argument('checkpoint', type = str, default = 'checkpoint.pth', help = 'checkpoint')
    parser.add_argument('--top_k', type = int, default = 3, help = 'top classes')
    parser.add_argument('--category_names', default = 'cat_to_name.json', help = 'json leading to category names')
    parser.add_argument('--gpu', type = str, default="cuda", help = 'if GPU is available', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    model, checkpoint = load_checkpoint(args.checkpoint)
    tensor = process_image(args.image_path)
    top_probs, top_classes = predict(args.image_path, model, args.top_k, args.gpu)
    print("Flower Name:" + str(top_classes))
    print("Probability:" + str(top_probs))
    
    