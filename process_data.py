from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, dim=1)
    elif prediction.dim() == 3:
        # If the input tensor does not have a batch dimension, perform softmax and argmax over the first dimension
        prediction = F.softmax(prediction, dim=0)
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
    prediction = TF.resize(
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
        ]
    )

    image_augmentations = v2.Compose(
        [
            v2.ColorJitter(
                brightness=(0.25, 2.5),
                contrast=(0.25, 3),
                saturation=(0.5, 2.5),
                hue=0.20,
            ),
            v2.RandomApply([v2.ElasticTransform(alpha=75, sigma=5)], p=0.01),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=(9, 17), sigma=(3, 16))], p=0.05
            ),
            v2.RandomAdjustSharpness(sharpness_factor=10, p=0.05),
        ]
    )

    return joint_augmentations, image_augmentations


def get_data_loader(args, batch_size, num_workers, validation_size=0.1):

    # Load the dataset without any transformations
    dataset = Cityscapes(
        args.data_path,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=None,
    )

    # Define the size of the validation set
    val_size = int(validation_size * len(dataset))
    train_size = len(dataset) - val_size

    # Split the dataset into training and validation subsets
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    # Apply the transformations to the subsets separately
    train_subset = TransformedSubset(
        train_subset, transform=train_preprocess
    )  # your training transformations
    val_subset = TransformedSubset(val_subset, transform=val_preprocess)

    # Define the data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_subset,
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
            mask = TF.resize(mask, self.resize_size, interpolation=self.interpolation)
        if not isinstance(mask, Tensor):
            mask = TF.pil_to_tensor(mask)
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


class TransformedSubset(Subset):
    def __init__(self, subset, transform=None):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        if self.transform:
            x, y = self.transform(x, y)
        return x, y
