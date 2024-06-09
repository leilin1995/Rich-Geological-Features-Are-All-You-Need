"""
__author__: Lei Lin
__project__: predict.py
__time__: 2024/3/27 
__email__: leilin1117@outlook.com
"""

import os
import torch
import torch.nn as nn
import time
from models import UNet3D
from torch.utils.data import DataLoader
from dataset import StructureDataset
import argparse
import shutil
import h5py
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Unet3D for fault segmentation in prediction")
parser.add_argument("--input_dir", default="../Dataset/Test", type=str,
                    help="directory to to the file that need to be predicted")
parser.add_argument("--save_dir", default="../Results/Unet3D/predictTest", type=str,
                    help="directory to save the files in evaluation")
parser.add_argument("--pred_internal_path", default="predict", type=str,
                    help="Internal path of .hdf5 of predicted data")
parser.add_argument("--label_internal_path", default="label", type=str,
                    help="Internal path of .hdf5 of target label data")
parser.add_argument("--classes", default=2, type=int,
                    help="number of classes")
parser.add_argument("--threshold", default=0.5, type=float,
                    help="threshold of label segmentation,target pred > threshold,background label < threshold")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--final_sigmoid", default=True, type=bool,
                    help="apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax (in this case softmax)")
parser.add_argument("--layer_order", default="cbrd", type=str,
                    help="determines the order of operators in a single layer (cbrd - Conv3D+Batchnorm+ReLU+Dropout)")
parser.add_argument("--trained_model", default="../Results/Unet3D/model.pt", type=str,
                    help="well-trained checkpoint path")
parser.add_argument("--data_norm", default="Normalize", type=str,
                    help="Data standardization methods. null, Normalize, Standard")
parser.add_argument("--norm01", default=True, type=bool,
                    help="When data_norm == Normalize,normalize data to 0-1 or -1-1")
parser.add_argument("--raw_internal_path", default="seismic", type=str,
                    help="Internal path of .hdf5 of input seismic data")
parser.add_argument("--device", default="gpu", type=str,
                    help="device to use for prediction")


def save_pred(file_path, data, internal_path="predict"):
    # append dataset
    with h5py.File(file_path, "w") as f:
        if internal_path in f:
            del f[internal_path]
        f.create_dataset(internal_path, data=data)


def main():
    args = parser.parse_args()
    input_dir = args.input_dir
    save_dir = args.save_dir
    print(f"Predicting '{input_dir}'...")
    if not os.path.exists(save_dir):
        print(f"Creating '{save_dir}'...")
        os.makedirs(save_dir)
    start = time.time()
    # Create model
    model = UNet3D(in_channels=args.in_channels,
                   out_channels=args.out_channels,
                   final_sigmoid=args.final_sigmoid,
                   layer_order=args.layer_order)
    model_path = args.trained_model

    print(f'Loading model from {model_path}...')
    model_dict = torch.load(model_path)["state_dict"]
    model.load_state_dict(model_dict)

    # Create test loader
    print(f'Create test dataset...')
    test_dataset = StructureDataset(file_dir=input_dir, phase="test",
                                    raw_internal_path=args.raw_internal_path,
                                    data_norm=args.data_norm,
                                    transform=False,
                                    norm01=args.norm01)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    # use DataParallel if more than 1 GPU available
    if torch.cuda.device_count() > 1 and not args.device == 'cpu':
        model = nn.DataParallel(model)
        print(f'Using {torch.cuda.device_count()} GPUs for prediction')
    if torch.cuda.is_available() and not args.device == 'cpu':
        model = model.cuda()
    # Sets the module in evaluation mode explicitly
    # It is necessary for batchnorm/dropout layers if present as well as final Sigmoid/Softmax to be applied
    model.eval()
    with torch.no_grad():
        for input, batch_files_name in test_loader:
            # send batch to gpu
            if torch.cuda.is_available() and not args.device == 'cpu':
                input = input.cuda(non_blocking=True)
            prediction = model(input)
            for output, file_name in zip(prediction, batch_files_name):
                # output shape: C,H,W,D，Probability of each channel
                # output type: gpu.tensor --> cpu.numpy
                output = output.cpu().numpy()
                save_path = os.path.join(save_dir, file_name)
                save_pred(file_path=save_path, data=output,
                          internal_path=args.pred_internal_path)
                print(f'Saving predictions to: {save_path}')
    print(f'Finished inference in {time.time() - start:.2f} seconds')


if __name__ == "__main__":
    main()
