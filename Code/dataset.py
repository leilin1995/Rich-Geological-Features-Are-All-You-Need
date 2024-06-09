"""
__author__: Lei Lin
__project__: dataset.py
__time__: 2024/3/25 
__email__: leilin1117@outlook.com
"""
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import torch
import math

class StructureDataset(Dataset):
    def __init__(self, file_dir, phase, raw_internal_path='seismic',
                 label_internal_path='label', patch_size=(192, 192, 192), data_norm="Normalize", random_crop=False,
                 transform=True, norm01=False):
        """
        Generate seismic structure dataset
        Args:
            file_dir: path to H5 file containing raw data and labels.
            phase: 'train' for training, 'val' for validation, 'test' for testing
            raw_internal_path: H5 internal path to the raw dataset
            label_internal_path: H5 internal path to the label dataset
            patch_size: Data size after random crop, patch size must smaller than raw data
            data_norm: Whether to normalize input data,[None,Normalize,Standard]
            random_crop: crop the raw data to the patch_size, a data argumentation method
            transform: Whether flip, rotate
            norm01: True: to 0,1 False: to -1,1, efficient when data_norm = Normalize
        """
        super(StructureDataset, self).__init__()
        assert phase in ['train', 'val', 'test']
        assert data_norm in [None, "Normalize", "Standard"]
        assert os.path.exists(file_dir), f"path '{file_dir}' does not exists."
        self.phase = phase
        self.file_dir = file_dir
        self.file_names = os.listdir(file_dir)
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path
        self.patch_size = patch_size
        self.data_norm = data_norm
        self.random_crop = random_crop
        self.transform = transform
        self.norm01 = norm01
        self.to_tensor = ToTensor(expand_dims=True)


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        file_name = self.file_names[item]
        file_path = os.path.join(self.file_dir, file_name)
        if self.phase != "test":
            # 打开HDF5文件
            with h5py.File(file_path, 'r') as file:
                # 读取'seismic'路径下的数据
                seis_data = file[self.raw_internal_path][:]
                # 读取'label'路径下的数据
                label_data = file[self.label_internal_path][:]
            self.seis_data = seis_data
            self.data_shape = self.seis_data.shape
            self.label_data = label_data
            if self.label_data.ndim == 4:
                self.label_data = np.argmax(self.label_data, axis=0)
            assert self.seis_data.ndim == 3 and self.label_data.ndim == 3, "Input data shape must 3 dim, not 4"
            if self.random_crop:
                assert all(
                    x > y for x, y in zip(self.data_shape, self.patch_size)), "Patch size must smaller than raw data."
                self.seis_data, self.label_data = self._random_crop()
            if self.data_norm is not None:
                if self.data_norm == "Normalize":
                    data_normalizer = Normalize(norm01=self.norm01)
                    self.seis_data = data_normalizer(self.seis_data)
                else:
                    data_normalizer = Standardize()
                    self.seis_data = data_normalizer(self.seis_data)
            if self.transform:
                rot = np.random.randint(0, 4)
                flip = np.random.randint(0, 3)
                self.seis_data = self._transform(self.seis_data, rot, flip)
                self.label_data = self._transform(self.label_data, rot, flip)
            return self.to_tensor(self.seis_data), self.to_tensor(self.label_data)

        else:
            with h5py.File(file_path, 'r') as file:
                # 读取'seismic'路径下的数据
                seis_data = file[self.raw_internal_path][:]
            if self.data_norm is not None:
                if self.data_norm == "Normalize":
                    data_normalizer = Normalize(norm01=self.norm01)
                    seis_data = data_normalizer(seis_data)
                else:
                    data_normalizer = Standardize()
                    seis_data = data_normalizer(seis_data)
            return self.to_tensor(seis_data), file_name

    def _random_crop(self):
        # 随机选择裁剪的起始点
        start_x = np.random.randint(0, self.data_shape[0] - self.patch_size[0])
        start_y = np.random.randint(0, self.data_shape[1] - self.patch_size[1])
        start_z = np.random.randint(0, self.data_shape[2] - self.patch_size[2])

        # 裁剪数据体
        cropped_seis_data = self.seis_data[start_x:start_x + self.patch_size[0],
                            start_y:start_y + self.patch_size[1],
                            start_z:start_z + self.patch_size[2]]
        cropped_label_data = self.label_data[start_x:start_x + self.patch_size[0],
                             start_y:start_y + self.patch_size[1],
                             start_z:start_z + self.patch_size[2]]

        return cropped_seis_data, cropped_label_data

    @staticmethod
    def _transform(data, rot=0, flip=0):
        # rotation
        data = np.rot90(data, k=rot)
        # flip
        if flip == 0:
            return data
        elif flip == 1:
            data = np.flip(data, axis=0)
            return data
        else:
            data = np.flip(data, axis=1)
            return data


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
        dtype (np.dtype): the desired output data type
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value=None, max_value=None, norm01=False, eps=1e-10, **kwargs):
        if min_value is not None and max_value is not None:
            assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.norm01 = norm01
        self.eps = eps

    def __call__(self, m):
        if self.min_value is None:
            min_value = np.min(m)
        else:
            min_value = self.min_value

        if self.max_value is None:
            max_value = np.max(m)
        else:
            max_value = self.max_value

        norm_0_1 = (m - min_value) / (max_value - min_value + self.eps)

        if self.norm01 is True:
            return np.clip(norm_0_1, 0, 1)
        else:
            return np.clip(2 * norm_0_1 - 1, -1, 1)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank: self.total_size: self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


