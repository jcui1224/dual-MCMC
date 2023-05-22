import torchvision
import torchvision.transforms as transforms
import PIL
import torch as t
import numpy as np
import lmdb
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import io
import string
from collections.abc import Iterable
import pickle
from torchvision.datasets.utils import verify_str_arg, iterable_to_str

class LSUNClass(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        import lmdb
        super(LSUNClass, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        # cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        # av begin
        # We only modified the location of cache_file.
        cache_file = os.path.join(self.root, '_cache_')
        # av end
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

class LSUN(VisionDataset):
    """
    `LSUN <https://www.yf.io/p/lsun>`_ dataset.

    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, classes='train', transform=None, target_transform=None):
        super(LSUN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                root=root + '/' + c + '_lmdb',
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes):
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower']
        dset_opts = ['train', 'val', 'test']

        try:
            verify_str_arg(classes, "classes", dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = ("Expected type str or Iterable for argument classes, "
                       "but got type {}.")
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr = ("Expected type str for elements in argument classes, "
                          "but got type {}.")
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = '_'.join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class",
                                        iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)

class LMDBDataset(t.utils.data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root, 'train.lmdb')
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB')
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.num_samples(self.name, self.train)

    def num_samples(self, dataset, train):
        if dataset == 'celeba256':
            return 27000 if train else 3000
        elif dataset == 'celeba64':
            return 162770 if train else 19867
        # elif dataset == 'celeba64':
        #     return 50000 if train else 19867
        elif dataset == 'imagenet-oord':
            return 1281147 if train else 50000
        elif dataset == 'ffhq':
            return 63000 if train else 7000
        else:
            raise NotImplementedError('dataset %s is unknown' % dataset)

class SingleImagesFolderMTDataset(t.utils.data.Dataset):
    def __init__(self, root, cache, transform=None, workers=8, split_size=200, protocol=None, train=True):
        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.transform = transform if not transform is None else lambda x: x
            self.images = []

            def split_seq(seq, size):
                newseq = []
                splitsize = 1.0 / size * len(seq)
                for i in range(size):
                    newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                return newseq

            def map(path_imgs):
                imgs_0 = [self.transform(np.array(PIL.Image.open(os.path.join(root, p_i)))) for p_i in
                          path_imgs]
                imgs_1 = [self.compress(img) for img in imgs_0]

                print('.')
                return imgs_1

            path_imgs = os.listdir(root)
            if train:
                path_imgs = path_imgs[:50000]
            else:
                path_imgs = path_imgs[-19867:]
            n_splits = len(path_imgs) // split_size
            path_imgs_splits = split_seq(path_imgs, n_splits)

            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(workers)
            results = pool.map(map, path_imgs_splits)
            pool.close()
            pool.join()

            for r in results:
                self.images.extend(r)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.images, f, protocol=protocol)

        print('Total number of images {}'.format(len(self.images)))

    def __getitem__(self, item):
        x = self.decompress(self.images[item])
        # y = torch.tensor([0])
        return x

    def __len__(self):
        return len(self.images)

    @staticmethod
    def compress(img):
        return img

    @staticmethod
    def decompress(output):
        return output

def get_dataset(args):

    img_size = args.img_size

    if args.dataset == 'cifar10':
        data_dir = args.data_dir
        print(data_dir)

        transform = transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        ds_train = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=True, transform=transform)
        ds_val = torchvision.datasets.CIFAR10(data_dir + 'cifar10', download=True, train=False, transform=transform)
        input_shape = [3, img_size, img_size]
        return ds_train, ds_val, input_shape

    if args.dataset == 'celeba64':
        num_classes = 40
        data_dir = args.data_dir

        class CropCelebA64(object):
            """ This class applies cropping for CelebA64. This is a simplified implementation of:
            https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
            """

            def __call__(self, pic):
                new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
                return new_pic

            def __repr__(self):
                return self.__class__.__name__ + '()'

        train_transform = transforms.Compose([
            CropCelebA64(),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        valid_transform = transforms.Compose([
            CropCelebA64(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = LMDBDataset(root=data_dir, name='celeba64', train=True, transform=train_transform,
                                 is_encoded=True)
        valid_data = LMDBDataset(root=data_dir, name='celeba64', train=False, transform=valid_transform,
                                 is_encoded=True)
        input_shape = [3, img_size, img_size]
        return train_data, valid_data, input_shape

    if args.dataset == 'celeba256':
        data_dir = args.data_dir

        num_classes = 1
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = LMDBDataset(root=data_dir, name='celeba256', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=data_dir,  name='celeba256', train=False, transform=valid_transform)
        input_shape = [3, img_size, img_size]
        return train_data, valid_data, input_shape

    if args.dataset == 'imagenet32':
        data_dir = args.data_dir
        num_classes = 1
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = LMDBDataset(root=data_dir, name='imagenet-oord', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=data_dir,  name='imagenet-oord', train=False, transform=valid_transform)
        input_shape = [3, img_size, img_size]
        return train_data, valid_data, input_shape

    if args.dataset == 'lsun_church_64':
        data_dir = args.data_dir
        num_classes = 1
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_data = LSUN(root=data_dir, classes=['church_outdoor_train'], transform=train_transform)
        valid_data = LSUN(root=data_dir, classes=['church_outdoor_val'], transform=valid_transform)
        return train_data, valid_data, [3, img_size, img_size]

