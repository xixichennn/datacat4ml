import os
import cv2
import torch
import albumentations as A

import config as CFG

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and captions must have the same length; so if there are multiple captions for each image, 
        the image_filenames mush have repetitive file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        ) 
        self.transforms = transforms
    
    def __getitem(self, idx):
        """
        returns a dictionary containing the image tensor and the caption.
        """
        item = {
            key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item
    
    def __len__(self):
        """ Returns the total number of items in the dataset, which is the length of the captions list """
        return len(self.captions)

def get_transforms(mode="train"):
    """Yu: I don't understand why the current implementation doesn't differentiate between modes"""
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )