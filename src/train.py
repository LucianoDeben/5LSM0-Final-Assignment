"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from src.model import Model
import torch
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
from torch import nn

DATA_PATH = "./Cityscapes"

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # data loading
    #dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic')
    dataset = Cityscapes(DATA_PATH, split='train', mode='fine', target_type='semantic')

    # visualize example images
        
    # visualize the image
    import matplotlib.pyplot as plt
    plt.imshow(x.permute(1, 2, 0))
    plt.show()
    

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move your model to the device
    model = Model().to(device)

    # define optimizer and loss function (don't forget to ignore class index 255)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss(ignore_index=255)

    # training/validation loop
    for epoch in range(10):
        for i, (x, y) in enumerate(dataset):
            # forward pass
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print some statistics
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
                
    # save model
    torch.save(model.state_dict(), "model.pth")

    # visualize some results



if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
