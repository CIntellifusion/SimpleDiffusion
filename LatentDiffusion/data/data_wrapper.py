
### data
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./", batch_size=64,num_workers=63):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # This method is intended for dataset downloading and preparation
        # We will download the MNIST dataset here (only called on 1 GPU in distributed training)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None,transform=None):
        # This method is called on every GPU in the distributed setup and should split the data
        if transform is None : 
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)) # 
            ])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = MNIST(self.data_dir, train=True, transform=transform)
            self.val_dataset = MNIST(self.data_dir, train=False, transform=transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=True)

class CelebDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64,num_workers=63,imsize=32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imsize = args.imsize 
    def split_dataset(self, dataset, split_ratio=0.2):
        """
        Divides the dataset into training and validation sets.

        Args:
        - dataset (datasets.Dataset): The dataset to be divided
        - split_ratio (float): The proportion of the validation set, default is 0.2

        Returns:
        - train_dataset (datasets.Dataset): The divided training set
        - val_dataset (datasets.Dataset): The divided validation set
        """
        num_val_samples = int(len(dataset) * split_ratio)

        val_dataset = dataset.shuffle(seed=42).select(range(num_val_samples))
        train_dataset = dataset.shuffle(seed=42).select(range(num_val_samples, len(dataset)))

        return train_dataset, val_dataset

    def prepare_data(self):
        self.dataset = load_dataset('nielsr/CelebA-faces')

    def setup(self, stage=None, transform=None):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = self.split_dataset(self.dataset['train'], split_ratio=0.2)

    @staticmethod
    def collate_fn(batch):
        # for example in batch:
        #     image = example['image']
        #     image.save("/home/haoyu/research/simplemodels/cache/test.jpg")
        transform = transforms.Compose([
            transforms.Resize((imsize,imsize)),  
            transforms.ToTensor(),           
            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # Normalize images
            # transforms.Lambda(lambda x: (x - 0.5) * 2) # unconment 
        ])
        transformed_batch = torch.stack([transform(example['image']) for example in batch])
        # print("transformerd",transformed_batch.mean(),transformed_batch.min(),transformed_batch.max())
        return transformed_batch,None

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, 
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          pin_memory=True)
