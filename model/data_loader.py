from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

def fetch_dataloader(types, params):
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            # create data loaders.
            if split  == 'train':
                dl = DataLoader(training_data, batch_size=params.batch_size)
            elif split == 'test':
                dl = DataLoader(test_data,batch_size=params.batch_size)

            dataloaders[split] = dl

    return dataloaders