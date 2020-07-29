import argparse
import sys
import numpy as np
import random
from os import listdir
import json

import torch
from torchvision import datasets
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image

from common import model_dicts, common_transform


def _get_model_from_checkpoint(checkpoint_path):
    model_info = torch.load('model_checkpoint.pth')
    model_to_use = model_dicts[model_info['model_used']]
    model = model_to_use.get("model")

    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    return model


def _process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image = Image.open(image)
    tensor_img = common_transform(image)
    np_img = np.array(tensor_img)
    return np_img.transpose((0, 1, 2))


def _predict(image_path, checkpoint_path, gpu, top_k):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model = _get_model_from_checkpoint(checkpoint_path)
    with torch.no_grad():
        image = torch.from_numpy(_process_image(image_path))
        image.unsqueeze_(0)
        image = image.float()
        model.to(device)
        
        outputs = model(image.to(device))
                
        probability, classes = torch.exp(outputs).topk(top_k)
        return probability[0].tolist(), classes[0].add(1).tolist()


def _predict_and_show(image_path, checkpoint_path, top_k, category_names, gpu):
    cat_to_name = {}

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    print("Image submitted:", image_path)
    probability, classes = _predict(image_path, checkpoint_path, gpu, top_k)
    print(f'Top {top_k} Predictions')
    class_outputs = [ f"{cat_to_name[str(c)]} ({str(c)}) - {round(p * 100, 3)}%" for c, p in zip(classes, probability)]
    for output in class_outputs:
        print(output)



def _get_arguments(sys_args):
    parser = argparse.ArgumentParser(
        description="Testing NN"
    )

    parser.add_argument("image_path")
    parser.add_argument("checkpoint_path")

    parser.add_argument(
        "--top_k",
        default=1,
        help="Number of predictions to return",
        type=int
    )

    parser.add_argument(
        "--category_names",
        default="./cat_to_name.json",
        help="file that contains the category mappings",
    )

    parser.add_argument("--gpu", action="store_true",
                    help="Run on GPU")

    return parser.parse_args(sys_args)


def cmd_line_entry():
    args = _get_arguments(sys.argv[1:])
    _predict_and_show(args.image_path, args.checkpoint_path, args.top_k,
                        args.category_names, args.gpu)


if __name__ == "__main__":
    cmd_line_entry()
