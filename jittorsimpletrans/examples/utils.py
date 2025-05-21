import jittor as jt
from jittor import transform

def get_transforms():
    """Get basic transforms for image datasets"""
    train_transform = transform.Compose([
        transform.ToTensor(),
    ])
    test_transform = transform.Compose([
        transform.ToTensor(),
    ])
    return train_transform, test_transform
