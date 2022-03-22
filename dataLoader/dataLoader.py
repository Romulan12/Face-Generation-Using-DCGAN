# necessary imports
import torch
from torchvision import datasets
from torchvision import transforms

def get_dataloader(batch_size, image_size, data_dir):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])
    
    image_data = datasets.ImageFolder(data_dir, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=True)
    
    return dataloader

