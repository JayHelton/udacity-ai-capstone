import argparse
import numpy as np
import random
from os import listdir
import json
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image

from common import model_dicts, common_transform


def _get_model_from_checkpoint(checkpoint_path):
    model_info = torch.load('model_checkpoint.pth')
    model_to_use = model_dicts[model_info['model']]

    model_to_use.classifier = model_info['classifier']
    model_to_use.load_state_dict(model_info['state_dict'])
    return model_to_use


def _process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    image = Image.open(image)
    tensor_img = common_transform(image)
    np_img = np.array(tensor_img)
    return np_img.transpose((0, 1, 2))


def _predict(image_path, checkpoint_path, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model = _get_model_from_checkpoint(checkpoint_path)
    with torch.no_grad():
        image = torch.from_numpy(_process_image(image_path))
        image.unsqueeze_(0)
        image = image.float()
        model.to(device)
        
        outputs = model(image.to(device))
                
        probability, classes = torch.exp(outputs).topk(topk)
        return probability[0].tolist(), classes[0].add(1).tolist()


def _predict_and_show(image_path, checkpoint_path, top_k, category_names, gpu):
    cat_to_name = {}

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    print("Image submitted:", image_path)
    probability, classes = _predict(image_path, checkpoint_path, gpu)

    classes_labels = [ f"{cat_to_name[str(c)]} ({str(c)})" for c in classes]

    image = Image.open(image_path)
    fig, ax = plt.subplots(2,1)

    ax[0].imshow(image);

    y_positions = np.arange(len(classes_labels))

    ax[1].barh(y_positions,probability,color='blue')

    ax[1].set_yticks(y_positions)
    ax[1].set_yticklabels(classes_labels)

    ax[1].invert_yaxis()  

    ax[1].set_xlabel('Accuracy (%)')
    ax[0].set_title(f'Top {top_k} Predictions')


def _get_arguments(sys_args):
    parser = argparse.ArgumentParser(
        description="Submitting measurement sets to the Submissions API."
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