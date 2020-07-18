from torchvision import models, transforms

model_dicts = {
    "vgg19": {
        "model": models.vgg19(pretrained=True),
        "input": 25088,
        "hidden": 4096
    },
    "densenet121": {
        "model":  models.densenet121(pretrained=True),
        "input": 1024,
        "hidden": 500
    }
}

common_transform = transforms.Compose([
                            transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])