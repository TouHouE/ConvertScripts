"""
This package contains 4 available functions:
1. reconstruct_centerline
2. reconstruct_mask
3. matprint
4. convert_points_LPS2RAS
Author: Hsu Shu-Yu
Date: 2024-03-19
"""

import os
import numpy as np
import pydicom as pyd
import nibabel as nib
import struct
from typing import Tuple, Union, List
TAG_FOCUS_LIST = (0x07a1, 0x1050)
TAG_FOCUS_NAME = (0x07a1, 0x1051)
tmp_path = r'D:\CCTA Result\0001\CT\20121215'


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def reconstruct_centerline(control_points, tangential_vectors, normal_vectors, num_points=100):
    r"""
    Reconstruct spline points using Hermite interpolation.

    Parameters:
        control_points (list of tuples): List of control points as tuples (x, y, z).
        tangential_vectors (list of tuples): List of tangential vectors as tuples (tx, ty, tz).
        normal_vectors (list of tuples): List of normal vectors as tuples (nx, ny, nz).
        num_points (int): Number of points to interpolate between each pair of control points.

    Returns:
        np.array: Array of spline points with shape (num_segments * num_points, 3).
    """
    control_points = np.array(control_points)
    tangential_vectors = np.array(tangential_vectors)
    normal_vectors = np.array(normal_vectors)

    num_segments = len(control_points) - 1
    u = np.linspace(0, 1, num_points)

    # Compute Hermite basis functions
    def hermite_basis(u):
        return np.array([
            2 * u ** 3 - 3 * u ** 2 + 1,
            -2 * u ** 3 + 3 * u ** 2,
            u ** 3 - 2 * u ** 2 + u,
            u ** 3 - u ** 2
        ])

    # Reshape control points and tangential vectors for broadcasting
    P0 = control_points[:-1, np.newaxis]
    P1 = control_points[1:, np.newaxis]
    T0 = tangential_vectors[:-1, np.newaxis]
    T1 = tangential_vectors[1:, np.newaxis]
    N0 = normal_vectors[:-1, np.newaxis]
    N1 = normal_vectors[1:, np.newaxis]

    # Compute spline points
    H = hermite_basis(u)
    Q = H[0, :, np.newaxis] * P0 + H[1, :, np.newaxis] * P1 + H[2, :, np.newaxis] * T0 + H[3, :, np.newaxis] * T1

    # Project onto planes defined by control points and normal vectors
    N = N0 * (1 - u[:, np.newaxis]) + N1 * u[:, np.newaxis]
    Q -= np.sum((Q - P0) * N, axis=-1, keepdims=True) * N

    return Q.reshape(-1, 3)


def reconstruct_mask(isp_obj: str | pyd.FileDataset, return_comment=True
                     ) -> np.ndarray | Tuple[np.ndarray, str]:
    """
    Reconstruct a 3D-mask from a path of ISP dicom file, or an ISP object
    :param isp_obj: a path or read from pydicom
    :param return_comment: Require mask label name
    :return:
    """
    if isinstance(isp_obj, str):
        ds = pyd.dcmread(isp_obj)
    else:
        ds = isp_obj
    try:
        imageComment = ds.get_item(key=(0x0020, 0x4000)).value.decode('ISO_IR 100', 'strict')
    except AttributeError as ae:
        imageComment = ds.get_item(key=(0x0020, 0x4000)).value
    dim_cand = ds.get_item(key=(0x07a1, 0x1007)).value
    if isinstance(dim_cand, list):
        voxel_dim = dim_cand
    else:
        voxel_dim = struct.unpack("H" * 3, dim_cand)
    sliceCount = voxel_dim[2]
    dim_array = np.array(ds.get_item(key=(0x07a1, 0x1008)).value.decode("utf-8").split("\\")).reshape(-1, 3).astype(
        float)
    # (origin, X axis, Y axis, Z axis)
    volumeDimension = \
        {"origin": dim_array[0, :], \
         "x": {"start": dim_array[0, 0], "end": dim_array[1, 0],
               "spacing": (dim_array[1, 0] - dim_array[0, 0]) / voxel_dim[0]}, \
         "y": {"start": dim_array[0, 1], "end": dim_array[2, 1],
               "spacing": (dim_array[2, 1] - dim_array[0, 1]) / voxel_dim[1]}, \
         "z": {"start": dim_array[0, 2], "end": dim_array[3, 2],
               "spacing": (dim_array[3, 2] - dim_array[0, 2]) / voxel_dim[2]}}

    # x07a11007 : 512\512\280
    # x07a11008 : "-59.99969\-98.12469\1814\139.68031\-98.12469\1814\-59.99969\101.55531\1814\-59.99969\-98.12469\1954"
    # x07a11009 : data of length 585470 for VR OW too long to show

    dd = ds.get_item(key=(0x07a1, 0x1009)).value

    pt = 0

    intS = struct.unpack("H" * 512 * sliceCount, dd[pt:(pt + 512 * sliceCount * 2)])
    pt += 512 * sliceCount * 2
    lineArray = np.array(intS).reshape((sliceCount, 512))

    bytemask = []
    for i in range(0, sliceCount):
        bytemask.append([])
        for j in range(0, 512):
            bytemask[i].append(struct.unpack("H" * lineArray[i][j], dd[pt:(pt + lineArray[i][j] * 2)]))
            pt += lineArray[i][j] * 2

    pointList = []
    for i in range(0, sliceCount):
        for j in range(0, 512):
            for k in bytemask[i][j]:
                pointList.append((i, j, (k & 0b0000000111111111), (k & 0b1111111000000000)))
                # (y, z, x, filling)
    pointArray = np.array(pointList)

    sslice = 100
    ssliceArray = np.zeros((voxel_dim[0], voxel_dim[1], sliceCount))
    fillLast = 0
    lastPoint = 0
    for point in pointArray:
        # aabbccdd eeffgghh
        # aa = fill 1 or not
        # bbcc continue filling
        # ddeeffgghh: location
        if fillLast == 1:
            ssliceArray[point[1], lastPoint:point[2], point[0]] = 1
            ssliceArray[point[1], point[2]:lastPoint, point[0]] = 1
        fillLast = 0
        # pixCount =
        # pixCount = (point[3]>>10)
        pixCount = (point[3] >> 10) & 0b011111
        # pixCount = (point[3]>>10) & 0b001111 wrong
        firstPix = (point[3] >> 15)
        # ssliceArray[point[1]:(point[1]+pixCount),point[2]] =1 worng
        ssliceArray[point[1], point[2]:(point[2] + pixCount + 1), point[0]] = 2.4
        ssliceArray[point[1], point[2], point[0]] = 3
        if firstPix == 1:
            fillLast = 1
            lastPoint = point[2] + 1

    maskArray = np.zeros_like(ssliceArray, dtype=np.int16)
    maskArray[np.where(ssliceArray > 0)] = 1

    if return_comment:
        return maskArray, imageComment
    return maskArray


def legal_plaque(x: pyd.Dataset):
    if not isinstance(x, pyd.Dataset): #ilegal type
        return False

    attr = x.get((0x07a1, 0x1051))
    if attr is not None:
        # print(attr.value)
        return attr.value == 'PLAQUE'
        # print('Nothing: ', x)
    return False


def reconstruct_plaque(ds, host_ct, return_plaque_name: bool = True, verbose: bool = False) -> list[np.ndarray] | list[list[np.ndarray, str]]:

    tarmar_list = ds.get((0x07a1, 0x1050))
    if tarmar_list is None:
        if verbose:
            print(f'No tarmar finding.', end='|')
        # print(tarmar_list)
        return []
    legal_plaque_list = list(filter(lambda x: legal_plaque(x), tarmar_list.value))

    if verbose:
        print(f'Number of Plaque in ISP: {len(legal_plaque_list)}')
    cp_list = np.array(ds[(0x07a1, 0x1012)].value).reshape((-1, 3, 3))
    plaque_list = []
    for plaque_idx, pcont in enumerate(legal_plaque_list): # For N plaque
        pshape = pcont[(0x07a1, 0x1052)].value
        fully = pcont[(0x07a1, 0x1053)].value
        segment1 = fully[:0xE0]
        segment2_len = struct.unpack("I", fully[0xE0: 0xE0 + 4])[0]
        segment2 = fully[0xE0+4:0xE0+4+segment2_len*4]
        segment3_len = struct.unpack("I", fully[0xE0+4+segment2_len*4: 0xE0+4+segment2_len*4 + 4])[0]
        segment3 = fully[0xE0+4+segment2_len*4+4:]
        volume_lps_point_list = []
        plaque_name_length = segment3_len
        plaque_name = segment3.decode("UTF-16").strip('\x00')
        if verbose:
            print(f"Plaqu#{plaque_idx} {plaque_name}: segment2_len:{segment2_len}, segment3_len:{segment3_len}/{len(segment3)}", end='')
        if (plaque_name == ''):  # unamed candidate plaque automatically generate by ISP
            if verbose:
                print('-> That is unnamed plaque. We ignore that')
            continue
        if verbose:
            print('\n')
        for i in range(0, segment2_len*4, 4): # For a plaque
            xy = struct.unpack("H", segment2[i: i + 2])[0]
            ncp = struct.unpack("<H", segment2[i + 2: i + 4])[0]
            left_ncp = ncp >> 8
            right_ncp = ncp & 0x00FF

            ncp = left_ncp + right_ncp
            lps_point, normal_vector, tangent_vector = cp_list[ncp]
            normal_vector /= np.sum(normal_vector ** 2) ** .5
            tangent_vector /= np.sum(tangent_vector ** 2) ** .5
            # Original Setting: normal_vector (cross) tangent
            orthogonal_vector = np.cross(normal_vector, tangent_vector)
            x = xy >> 8
            y = xy & 0x00FF
            dx, dy = x - 128, y - 128
            mask_lps = -dy * (orthogonal_vector / 10) - dx * (tangent_vector / 10) + lps_point
            volume_lps_point_list.append(mask_lps)

        volume_lps_point_list = np.array(volume_lps_point_list)
        ras_vol = volume_lps_point_list.copy()
        ras_vol[:, :2] *= -1
        volume_ijk = np.zeros_like(host_ct.get_fdata(), dtype=np.short)
        rasp_vol = np.pad(ras_vol, ((0, 0), (0, 1)), mode='constant', constant_values=1)
        ijkp_vol = (np.linalg.inv(host_ct.affine) @ rasp_vol.T).T

        for ijkp in ijkp_vol:
            i, j, k, _ = ijkp.astype(np.int32)
            volume_ijk[i, j, k] = 1
        pack = [volume_ijk]
        if return_plaque_name:
            pack.append(f'{ds.ImageComments}_{plaque_name}')

        plaque_list.append(pack)
        # plaque_nii = nib.Nifti1Image(volume_ijk, host_ct.affine)
        # nib.save(plaque_nii, f'./static/mask_{ds.ImageComments}_plaque-{plaque_idx}-{plaque_name}-Af.nii.gz')
    return plaque_list

# This one was abandoned
def reconstruct_3d_mask(tag_data, volume_shape):
    mask = np.zeros(volume_shape, dtype=np.uint8)  # Initialize 3D mask

    # Initialize variables to keep track of current line and slice
    current_slice = 0
    current_line = 0
    current_x = 0

    # Iterate through the tag data
    i = 0
    while i < len(tag_data):
        # Parse the first 2 bytes to get the line length
        line_length = int.from_bytes(tag_data[i:i + 2], byteorder='little')
        i += 2

        # Move to the next line if needed
        if current_x == volume_shape[2]:
            current_line += 1
            current_x = 0

        # Update current slice if needed
        if current_line == volume_shape[1]:
            current_slice += 1
            current_line = 0

        # Parse the second part of the tag data for the current line
        while i < len(tag_data) and current_x < line_length:
            # Parse each control point (2 bytes)
            control_point = tag_data[i:i + 2]
            i += 2

            # Extract information from the control point
            fill_flag = control_point[0] >> 6
            fill_length = ((control_point[0] & 0b00111100) << 2) + (control_point[1] >> 6)
            x_coordinate = ((control_point[1] & 0b00111111) << 4) + ((control_point[0] & 0b00000011) << 8)

            # Handle negative x coordinate
            if x_coordinate >= volume_shape[2]:
                x_coordinate -= volume_shape[2]

            # Fill mask data based on control point
            if fill_flag == 1:
                # Fill pixels until next control point
                if x_coordinate + fill_length <= volume_shape[2]:
                    mask[current_slice, current_line, x_coordinate:x_coordinate + fill_length] = 1
                else:
                    mask[current_slice, current_line, x_coordinate:] = 1
                    remaining_length = fill_length - (volume_shape[2] - x_coordinate)
                    mask[current_slice, current_line, :remaining_length] = 1

            # Move to the next control point
            current_x = (x_coordinate + fill_length) % volume_shape[2]

            # Check if the control point indicates the end of the line
            if fill_flag == 1:
                break

    return mask


def convert_points_pack_LPS2RAS(points_pack: np.ndarray) -> np.ndarray:
    """
    :banded this
    Converts the coordinate system from LPS to RAS,
    :param points_pack: shape with (N, 3, 3)
    :return:
    """
    points_pack[:, 0:2] *= -1
    return points_pack


def convert_points_LPS2RAS(points: np.ndarray) -> np.ndarray:
    """
    Converts the coordinate system from LPS to RAS
    :param points: shape with (N, 3)
    :return:
    """
    points[:, :2] *= -1
    return points

