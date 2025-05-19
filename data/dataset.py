import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(name="MNIST", batch_size=32):
    if name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        input_dim = 28 * 28

    elif name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        input_dim = 28 * 28

    elif name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        input_dim = 32 * 32 * 3  # CIFAR10 has 3 color channels

    else:
        raise ValueError(f"Dataset '{name}' not supported yet.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, input_dim
