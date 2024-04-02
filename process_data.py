import random
from functools import partial
from typing import Optional

import torch
import torchvision.transforms.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Cityscapes
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.transforms import InterpolationMode, v2


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


class JointTransform:
    def __init__(
        self,
        image_transform=None,
        target_transform=None,
        joint_augmentations=None,
        image_augmentations=None,
    ):
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.image_augmentations = image_augmentations
        self.joint_augmentations = joint_augmentations

    def __call__(self, image, mask):
        # Apply image transformations
        image = self.image_transform(image)

        # Apply mask transformations
        mask = self.target_transform(mask)

        # Synchronize augmentations
        if self.joint_augmentations:
            seed = random.randint(0, 2**8)
            torch.manual_seed(seed)
            random.seed(seed)
            image = self.joint_augmentations(image)

            torch.manual_seed(seed)
            mask = self.joint_augmentations(mask)

        if self.image_augmentations:
            image = self.image_augmentations(image)

        return image, mask


def preprocess(img):
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    preprocess = weights.transforms()
    img = preprocess(img)
    joint_augmentations, image_augmentations = get_augmentations()
    img = joint_augmentations(img)
    img = image_augmentations(img)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return img


def postprocess(prediction, shape=(256, 256)):
    # Apply softmax to get probabilities
    prediction = torch.nn.functional.softmax(prediction, dim=1)

    # Get the class with the highest probability for each pixel
    prediction = torch.argmax(prediction, dim=1)

    # Convert the tensor to a floating point type
    prediction = prediction.float()

    # Add an extra dimension for the number of channels
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


def preprocess_mask(mask):
    mask_transform = partial(MaskTransform, resize_size=520)
    mask = mask_transform()(mask)
    joint_augmentations, _ = get_augmentations()
    mask = joint_augmentations(mask)
    return mask


def joint_preprocess(img, mask):
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


def get_transforms(spatial_dims=(256, 256)):
    """
    Get the data transforms

    Returns:
    transform: Compose
    target_transform: Compose
    """
    transform = v2.Compose(
        [
            v2.Resize(spatial_dims),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    target_transform = v2.Compose(
        [
            v2.Resize(spatial_dims, interpolation=InterpolationMode.NEAREST),
            v2.ToImage(),
        ]
    )

    return transform, target_transform


def get_augmentations(spatial_dims=(256, 256)):
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
            # v2.RandomApply([v2.ElasticTransform(alpha=75, sigma=5)], p=0.01),
            # v2.RandomApply(
            #     [v2.GaussianBlur(kernel_size=(9, 17), sigma=(3, 16))], p=0.05
            # ),
            # v2.RandomAdjustSharpness(sharpness_factor=10, p=0.1),
        ]
    )

    return joint_augmentations, image_augmentations


def get_data_loader(args, batch_size, num_workers, validation_size=0.1):
    # Define the transform for the images
    transform, target_transform = get_transforms(spatial_dims=(256, 256))
    joint_augmentations, image_augmentations = get_augmentations(
        spatial_dims=(256, 256)
    )
    joint_transform = JointTransform(
        transform, target_transform, joint_augmentations, image_augmentations
    )

    # Load the dataset
    dataset = Cityscapes(
        args.data_path,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=joint_transform,
    )

    # Define the size of the validation set
    val_size = int(validation_size * len(dataset))
    train_size = len(dataset) - val_size

    # Split the dataset
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    # Define the data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
