import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


class PBTDataLoader(data.DataLoader):
    def __init__(self, dataset_name, phase, is_training, batch_size, num_workers):

        self.phase = phase
        self.batch_size_ = batch_size

        if dataset_name == 'CIFAR10':
            if is_training:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        else:
            raise NotImplementedError('Unsupported dataset: {}'.format(dataset_name))

        super(PBTDataLoader, self).__init__(dataset,
                                            batch_size=batch_size,
                                            shuffle=is_training,
                                            num_workers=num_workers)
