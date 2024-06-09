"""
__author__: Lei Lin
__project__: data_split_merge.py
__time__: 2024/5/3 
__email__: leilin1117@outlook.com
"""
import numpy as np
import h5py
import os
import segyio


class SegyIO():
    def __init__(self, filepath):
        self.filepath = filepath

    def read(self):
        with segyio.open(self.filepath, strict=True, ignore_geometry=False) as f:
            inlines_dim = len(f.ilines)
            crosslines_dim = len(f.xlines)
            timeslice_dim = len(f.samples)
            dim = (inlines_dim, crosslines_dim, timeslice_dim)
            data = f.trace.raw[:].reshape(dim)
        return np.array(data)


def calculate_split_number_single(num, overlap, sub_dim=192):
    num_remain = num % (sub_dim - overlap)  # 求余
    if num_remain <= overlap:
        cube_num = num // (sub_dim - overlap)  # 求整
    else:
        cube_num = num // (sub_dim - overlap) + 1
    return cube_num


def calculate_split_number_cube(data_dims, overlap, sub_dim):
    crossline_num, inline_num, timeline_num = data_dims
    cubenum_crossline = calculate_split_number_single(crossline_num, overlap, sub_dim)
    cubenum_inline = calculate_split_number_single(inline_num, overlap, sub_dim)
    cubenum_timeline = calculate_split_number_single(timeline_num, overlap, sub_dim)
    return cubenum_crossline, cubenum_inline, cubenum_timeline


def split_data_to_hdf5(data, output_dir, internal_path="seismic", overlap=44, sub_dim=192):
    print(f"Starting data split with parameters:")
    print(f"Output directory: {output_dir}")
    print(f"Internal path: {internal_path}")
    print(f"Overlap: {overlap}")
    print(f"Sub-dimension: {sub_dim, sub_dim, sub_dim}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dims = data.shape
    print(f"Input data dimensions: {data_dims}")
    cubenum_crossline, cubenum_inline, cubenum_timeline = calculate_split_number_cube(data_dims, overlap, sub_dim)
    print(f"Number of blocks (crossline, inline, timeline): {cubenum_crossline}, {cubenum_inline}, {cubenum_timeline}")
    step = sub_dim - overlap
    for i in range(cubenum_crossline):
        for j in range(cubenum_inline):
            for k in range(cubenum_timeline):
                subdata = data[i * step:sub_dim + i * step,
                          j * step:sub_dim + j * step,
                          k * step:sub_dim + k * step
                          ]
                subdata_dims = subdata.shape
                # 补足边缘不足的部分
                if subdata_dims != (sub_dim, sub_dim, sub_dim):
                    padding = ((0, sub_dim - subdata_dims[0]),
                               (0, sub_dim - subdata_dims[1]),
                               (0, sub_dim - subdata_dims[2]))
                    subdata = np.pad(subdata, padding, mode='constant', constant_values=0)
                file_path = os.path.join(output_dir, f'block_XLine{i}_ILine{j}_TLine{k}.hdf5')
                with h5py.File(file_path, 'w') as h5_file:
                    h5_file.create_dataset(internal_path, data=subdata)
                print(f"Saved block to {file_path}, block dimensions: {subdata.shape}")


def merge_data_from_hdf5(input_dir, data_dims, internal_path="seismic", trim=22, overlap=44, sub_dim=192):
    print(f"Starting data merge with parameters:")
    print(f"Input directory: {input_dir}")
    print(f"Data dimensions: {data_dims}")
    print(f"Internal path: {internal_path}")
    print(f"Trim: {trim}")
    print(f"Overlap: {overlap}")
    print(f"Sub-dimension: {sub_dim}")

    cubenum_crossline, cubenum_inline, cubenum_timeline = calculate_split_number_cube(data_dims, overlap, sub_dim)
    print(f"Number of blocks (crossline, inline, timeline): {cubenum_crossline}, {cubenum_inline}, {cubenum_timeline}")
    step = sub_dim - overlap
    complete_data = np.zeros((sub_dim + (cubenum_crossline - 1) * step, sub_dim + (cubenum_inline - 1) * step,
                              sub_dim + (cubenum_timeline - 1) * step))
    for i in range(cubenum_crossline):
        for j in range(cubenum_inline):
            for k in range(cubenum_timeline):
                filepath = os.path.join(input_dir, f'block_XLine{i}_ILine{j}_TLine{k}.hdf5')

                with h5py.File(filepath, 'r') as h5_file:
                    subdata = h5_file[internal_path][:]
                    if len(subdata.shape) > 3:
                        if subdata.shape[0] == 1:
                            subdata = np.squeeze(subdata, axis=0)
                        elif subdata.shape[0] == 2:
                            subdata = subdata[1,]  # channel 0: background,channel 1: target
                        else:
                            raise Exception(
                                "Dimension error! The desired number of channels for binary segmentation is 1 or 2.")
                    # 使用any()检查是否存在NaN
                    has_nan = np.any(np.isnan(subdata))
                    print(has_nan)  # 结果同上
                z_trim_start = trim if k != 0 else 0
                y_trim_start = trim if j != 0 else 0
                x_trim_start = trim if i != 0 else 0

                z_trim_end = trim if k != (cubenum_timeline - 1) else 0
                y_trim_end = trim if j != (cubenum_inline - 1) else 0
                x_trim_end = trim if i != (cubenum_crossline - 1) else 0

                x_start = i * step + x_trim_start
                x_end = sub_dim + i * step - x_trim_end

                y_start = j * step + y_trim_start
                y_end = sub_dim + j * step - y_trim_end

                z_start = k * step + z_trim_start
                z_end = sub_dim + k * step - z_trim_end
                complete_data[x_start:x_end, y_start:y_end, z_start:z_end] = subdata[x_trim_start:sub_dim - x_trim_end,
                                                                             y_trim_start:sub_dim - y_trim_end,
                                                                             z_trim_start:sub_dim - z_trim_end]
                print(
                    f"Inserted block from {filepath} into position {(z_start, y_start, x_start)} in the full dataset.")

    return complete_data[:data_dims[0], :data_dims[1], :data_dims[2]]
