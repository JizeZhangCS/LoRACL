from PIL import Image
import torchvision.datasets as datasets


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class MemoryDataset(object):
    def __init__(self, traindir, transform=None):
        self.transform = transform
        self.root = traindir
        self.imgfolder = datasets.ImageFolder(traindir, transform)
        self.imgs = []
        for pth, label in self.imgfolder.imgs:
            self.imgs.append((pil_loader(pth), label))
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        
        img, label = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MultipleDataset(object):
    def __init__(self, traindir, transform=None):
        self.transform = transform
        self.root_lst = traindir.split(",")
        self.imgs = []

        for root in self.root_lst:
            for img_data in datasets.ImageFolder(root, transform).imgs:
                self.imgs.append(img_data)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        
        pth, label = self.imgs[index]
        img = pil_loader(pth)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
    

class IdxDataset(object):
    def __init__(self, traindir, transform=None):
        self.transform = transform
        self.root = traindir
        self.imgs = []
        for img_data in datasets.ImageFolder(traindir, transform).imgs:
            self.imgs.append(img_data)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        
        pth, label = self.imgs[index]
        img = pil_loader(pth)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.imgs)