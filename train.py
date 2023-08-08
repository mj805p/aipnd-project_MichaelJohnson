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

def train_network(args):
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(args.data_dir + '/train', transform=data_transforms["train"])
    valid_data = datasets.ImageFolder(args.data_dir + '/valid', transform=data_transforms["valid"])
    test_data = datasets.ImageFolder(args.data_dir + '/test', transform=data_transforms["test"])

    image_datasets = {"train":train_data, "valid":valid_data, "test":test_data}
    
    train_dataloaders = torch.utils.data.DataLoader(image_datasets["train"], batch_size=32, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=32, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(image_datasets["test"], batch_size=32, shuffle=True)                                                               

    dataloaders = {"train": train_dataloaders, "valid": valid_dataloaders, "test": test_dataloaders}

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    
    ## IF statement based on arch selected
    if args.model_arch == "vgg11":
        model = torchvision.models.vgg11(pretrained=True)
    elif args.model_arch == "vgg13":
        model = torchvision.models.vgg13(pretrained=True)
    elif args.model_arch == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)
    elif args.model_arch == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)

    in_features = 25088  # Inputs based on vgg
    out_features = len(cat_to_name)  # Outputs related to # of possible outcomes

    # Verify Inputs/Outputs
    print("Inputs:" + str(in_features))
    print("Outputs:" + str(out_features))

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(in_features, args.hidden_layer),
                                     nn.ReLU(),
                                     nn.Dropout(args.dropout),
                                     nn.Linear(args.hidden_layer, 1000),
                                     nn.ReLU(),
                                     nn.Dropout(args.dropout),
                                     nn.Linear(1000, out_features),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    device = torch.device('cuda')
    model.to(device)
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward Pass
            logps = model.forward(inputs)
            # Calculate Loss
            loss = criterion(logps, labels)
            # Backward pass
            loss.backward()
            # update weights
            optimizer.step()

            running_loss += loss.item()
            
            print("Epoch:" + str(epoch))
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for images, labels in dataloaders['test']:
                        log_ps = model(images)
                        test_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                        train_losses.append(running_loss/len(dataloaders['train']))
                        test_losses.append(test_loss/len(dataloaders['test']))

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders['train'])),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['test'])),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))
                running_loss = 0
                model.train()

    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'input_size': 25088,
                  'output_size': len(cat_to_name),
                  'arch': 'vgg16',
                  'learning rate': args.learning_rate,
                  'batch_size': 32,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'epochs:': args.epochs,
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, args.save_dir)
    

if __name__ == '__main__':
    #Set up command line inputs ie. python train.py data_dir --gpu
    parser = argparse.ArgumentParser(
        description='parser for train arguments')

    parser.add_argument('data_dir', type = str, default = "./flowers/", help = 'directory of classes/images')
    parser.add_argument('--save_dir', type = str, default = "./checkpoint.pth", help = 'directory of where saved checkpoints will be stored')
    parser.add_argument('--arch', type = str, default = "vgg16", help = 'model architecture', choices=['vgg11', 'vg13', 'vgg16', 'vgg19'])
    parser.add_argument('--learning_rate', type = float, default = 0.003, help = 'Learning Rate')
    parser.add_argument('--hidden_layer', type = int, default = 4096, help = 'hidden_layers')
    parser.add_argument('--epochs', type = int, default = 3, help = 'epochs')
    parser.add_argument('--dropout', type = float, default = 0.2, help = 'dropout value')
    parser.add_argument('--gpu', type = str, default="cuda", help = 'if GPU is available', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    train_network(args)
