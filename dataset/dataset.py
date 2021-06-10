import torch
import numpy as np
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_images(images, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # Plot img
        ax.imshow(images[i, :, :, :], vmin=-1, vmax=1, interpolation='spline16')

        # Show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def get_train_valid_loader(data_dir, batch_size, augment, random_seed, val_split=0.1, shuffle=True, show_sample=False,
                           num_workers=4, pin_memory=False, download_train=True, download_val=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - val_split: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    - len(train_idx): size of training set.
    - len(valid_idx): size of validation set.
    """
    error_msg = "[!] val_split should be in the range [0, 1]."
    assert ((val_split >= 0) and (val_split <= 1)), error_msg

    # Normalize transforms
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # Define transform
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # If augment is enabled, augment transform. Else, don't augment.
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize, ])

    # Load the dataset
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download_train, transform=train_transform, )
    valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download_val, transform=valid_transform, )

    # Splitting data set and indices into training and validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Get training set iterator
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory, )

    # Get validation set iterator
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory, )

    # Visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(train_dataset, batch_size=9, shuffle=shuffle,
                                                    num_workers=num_workers, pin_memory=pin_memory, )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        x = images.numpy().transpose([0, 2, 3, 1])
        plot_images(x, labels)

    return train_loader, len(train_idx), valid_loader, len(valid_idx)


def get_test_loader(data_dir, batch_size, shuffle=True, num_workers=4, pin_memory=False, download_test=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    - len(dataset): size of test set.
    """

    # Normalize transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Get test data set
    dataset = datasets.CIFAR10(root=data_dir, train=False, download=download_test, transform=transform, )

    # Get test set iterator
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              pin_memory=pin_memory, )

    return data_loader, len(dataset)


def show_some_data(data_loaders):
    # Get a batch of training data
    inputs, classes = next(iter(data_loaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[label_names[x] for x in classes])


def prepare_datasets(data_dir, train_val_batch_size, test_batch_size, val_split=0.1, num_workers=4, download_train=True,
                     download_val=True, download_test=True):
    """
        Utility function for preparing data sets for testing, training, and validation
        using utility functions: get_train_valid_loader and get_test_loader.
        Params
        ------
        - data_dir: path directory to the dataset.
        - train_val_batch_size: how many samples per batch to load during training.
        - test_batch_size: how many samples per batch to load during testing.
        - val_split: percentage split of the training set used for
          the validation set. Should be a float in the range [0, 1].
        - num_workers: number of subprocesses to use when loading the dataset.
        - download_val: whether validation data should be downloaded when not available in library
        - download_train: whether training data should be downloaded when not available in library
        - download_test: whether testing data should be downloaded when not available in library
        Returns
        -------
        - data_loaders: dictionary of training set, validation set, and testing set iterators.
        - dataset_sizes: dictionary of training set, validation set, and testing set sizes.
        - len(label_names): number of total classes
        """

    # Get training set and validation set iterators using get_train_valid_loader function
    train_loader, train_size, valid_loader, val_split = get_train_valid_loader(data_dir=data_dir,
                                                                                batch_size=train_val_batch_size,
                                                                                augment=True,
                                                                                random_seed=10,
                                                                                val_split=val_split,
                                                                                shuffle=True,
                                                                                show_sample=False,
                                                                                num_workers=num_workers,
                                                                                pin_memory=False,
                                                                                download_train=download_train,
                                                                                download_val=download_val)

    # Get testing set iterators using get_test_loader
    test_loader, test_size = get_test_loader(data_dir=data_dir,
                                             batch_size=test_batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             download_test=download_test)

    # Make iterator dictionary of training, validation, and testing set iterators
    data_loaders = {'train': train_loader,
                    'val': valid_loader,
                    'test': test_loader}

    # Make iterator dictionary of training, validation, and testing set sizes
    dataset_sizes = {'train': train_size,
                     'val': val_split,
                     'test': test_size}

    # If you wanna show some data uncomment below
    # WARNING: batch_size number of images are show! (make sure batch_size is not large to display nice grid of images!)
    # show_some_data(data_loaders)

    return data_loaders, dataset_sizes, len(label_names)
