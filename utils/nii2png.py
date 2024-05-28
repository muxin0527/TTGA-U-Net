import os
import numpy as np
import nibabel as nb
from time import time
import PIL.Image as Image
from glob import glob
from natsort import natsorted


image_path = '/home/data/LiverVessel/final/images/'
label_path = '/home/data/LiverVessel/final/labels/'
slice_path = '/home/data/LiverVessel/resample/slice_png/'
save_image_path = '/home/data/Program/LiverVessel_Seg/dataset/Origin/images/'
save_label_path = '/home/data/Program/LiverVessel_Seg/dataset/Origin/labels/'


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def nii2png(file_name, save_path):
    t0 = time()
    # shutil.rmtree(slice_path)
    # shutil.rmtree(median_png_path)

    print('******NIFTI to PNG******')
    # create_dir(slice_path)
    # create_dir(median_png_path)

    nii = nb.load(file_name)
    affine = nii.affine
    resx = nii.header['pixdim'][1]
    data = nii.get_fdata()

    # Clip the specific slice of selected range
    data = data[:, :, :]
    data = np.array(data)
    # data = np.clip(data, 0, 150)

    png_size = 256
    min = np.min(data)
    max = np.max(data)
    data = ((data - min) / (max - min)) * 255
    shape = data.shape
    ref = np.round(affine / resx)

    if ref[0, 0] == 0:
        data = np.transpose(data, (1, 0, 2))
        if ref[0, 1] == 1:
            data = np.flip(data, 0)
        if ref[1, 0] == 1:
            data = np.flip(data, 1)
    if ref[1, 0] == 0:
        if ref[0, 0] == 1:
            data = np.flip(data, 0)
        if ref[1, 1] == 1:
            data = np.flip(data, 1)
    data = np.transpose(data, (1, 0, 2))

    for i in range(shape[2]):
        array = data[:, :, i]
        img = Image.fromarray(array).convert('L').resize((png_size, png_size))
        if i >= 0 and i < 10:
            index = '0' + str(i)
        else:
            index = str(i)
        img.save(os.path.join(save_path, file_name.split('/')[-1].split('_')[-1].split('.')[0] + '_' + index + '.png'))
    t1 = time()
    print('nii to png time:', t1 - t0)
    print('*********************')

    # Return slice number
    return 0


nii2png(image_path, save_path)
