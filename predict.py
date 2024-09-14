import argparse
import torch
from PIL import Image
import json
from torchvision import models
from torch import nn
from collections import OrderedDict
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image).numpy()
    return image

def predict(image_path, model, topk=5, gpu=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    top_p = top_p.cpu().numpy()[0]
    top_class = top_class.cpu().numpy()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[cls] for cls in top_class]
    return top_p, top_classes

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, args.top_k, args.gpu)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(cls)] for cls in classes]
    else:
        class_names = classes

    print('Probabilities:', probs)
    print('Classes:', class_names)

if __name__ == '__main__':
    main()
