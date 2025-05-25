"""
Author: Annam.ai IIT Ropar  
Team Name: AgroMinds AI  
Team Members: Amirtha K, Tharun Babu C 
Leaderboard Rank: 41  
"""

# This module contains all the image preprocessing functions used during training and validation.

from torchvision import transforms

def get_transforms(phase="train"):
    """
    Returns the set of transformations to be applied on the images
    depending on the phase: 'train', 'val', or 'test'.
    """

    if phase == "train":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:  # val or test
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform
