from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as NF
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import Cityscapes
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.transforms import InterpolationMode, v2


def preprocess(img):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    preprocess = weights.transforms()
    img = preprocess(img)
    return img


def train_preprocess(img, mask):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    preprocess = weights.transforms()
    img = preprocess(img)

    mask_transform = partial(MaskTransform, resize_size=520)
    mask = mask_transform()(mask)

    joint_augmentations, image_augmentations = get_augmentations()

    # Apply the same joint augmentations to both image and mask
    img, mask = joint_augmentations(img, mask)

    # Apply image augmentations only to the image
    img = image_augmentations(img)

    return img, mask


def val_preprocess(img, mask):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    preprocess = weights.transforms()
    img = preprocess(img)

    mask_transform = partial(MaskTransform, resize_size=520)
    mask = mask_transform()(mask)
    return img, mask


def postprocess(prediction, shape=(520, 1040)):
    # Check the number of dimensions in the input tensor
    if prediction.dim() == 4:
        # If the input tensor has a batch dimension, perform softmax and argmax over the second dimension
        prediction = NF.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, dim=1)
    elif prediction.dim() == 3:
        # If the input tensor does not have a batch dimension, perform softmax and argmax over the first dimension
        prediction = NF.softmax(prediction, dim=0)
        prediction = torch.argmax(prediction, dim=0)
    else:
        raise ValueError(
            f"Input tensor must have 3 or 4 dimensions. Got {prediction.dim()}."
        )

    # Convert the tensor to a floating point type
    prediction = prediction.float()

    # Add an extra dimension for the number of channels if it's not present
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(0)

    # Resize prediction to original image size
    prediction = F.resize(
        prediction, size=shape, interpolation=InterpolationMode.NEAREST
    )

    # Remove the extra dimension
    prediction = prediction.squeeze(0)

    # Convert the tensor to a numpy array
    prediction = prediction.cpu().detach().numpy()

    return prediction


def get_augmentations(spatial_dims=(520, 1040)):
    """
    Get the data augmentations

    Returns:
    transform: Compose
    target_transform: Compose
    """
    joint_augmentations = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(spatial_dims, scale=(0.8, 1.0)),
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.Identity(),
        ]
    )

    image_augmentations = v2.Compose(
        [
            v2.ColorJitter(
                brightness=(0.5, 2.5),
                contrast=(0.5, 3),
                saturation=(0.5, 2.5),
                hue=0.15,
            ),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomApply([v2.ElasticTransform(alpha=75, sigma=5)], p=0.01),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=(9, 17), sigma=(3, 16))], p=0.05
            ),
            v2.RandomAdjustSharpness(sharpness_factor=10, p=0.05),
            v2.Identity(),
        ]
    )

    return joint_augmentations, image_augmentations


def get_data_loader(args, batch_size, num_workers, validation_size=0.1):

    # Load the datasets
    train_dataset = Cityscapes(
        args.data_path,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=train_preprocess,
    )

    val_dataset = Cityscapes(
        args.data_path,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=val_preprocess,
    )

    # Get indices for training and validation sets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    # split_idx = int(np.floor(validation_size * num_train))

    # train_idx, valid_idx = indices[:split_idx], indices[split_idx:]
    # assert len(train_idx) != 0 and len(valid_idx) != 0

    train_indices, val_indices = train_test_split(
        indices, test_size=validation_size, random_state=42
    )
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # Define the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


class MaskTransform(nn.Module):
    def __init__(
        self,
        *,
        resize_size: Optional[int],
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
    ) -> None:
        super().__init__()
        self.resize_size = [resize_size] if resize_size is not None else None
        self.interpolation = interpolation

    def forward(self, mask: Tensor) -> Tensor:
        if isinstance(self.resize_size, list):
            mask = F.resize(mask, self.resize_size, interpolation=self.interpolation)
        if not isinstance(mask, Tensor):
            mask = F.pil_to_tensor(mask)
        return mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, H, W)`` and single ``(H, W)`` mask ``torch.Tensor`` objects. "
            f"The masks are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. "
            "Finally the values are converted to long datatype."
        )
