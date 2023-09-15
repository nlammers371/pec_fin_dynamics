import napari
import numpy as np
from aicsimageio.readers.czi_reader import CziReader
from aicsimageio import AICSImage
import os

# set parameters
file_name = "zf_bact2-tdTom_fin_48hpf_timeseries02_2023_05_25__20_23_50_462_fin.ome.tiff"
# file_name = "2022_12_15 HCR Hand2 Tbx5a Fgf10a_1.czi"
read_root = 'D:/Nick/20230525/processed_data/'
# read_root = "E:/Nick/Dropbox (Cole Trapnell's Lab)/Nick/pecFin/HCR_Data/raw/2022_12_15 HCR Hand2 Tbx5a Fgf10a/"
read_path = os.path.join(read_root, file_name)

#############
# Main image
#############

# load in raw czi file
print('Reading image metadata...')
imObject = AICSImage(read_path)
print('Loading image...')
imData = np.squeeze(imObject.get_image_data("TCZYX", T=1))
print(imData.shape)


res_raw = imObject.physical_pixel_sizes
scale_vec = np.asarray(res_raw)

viewer = napari.view_image(imData, scale=scale_vec)
# labels_layer = viewer.add_labels(label_data[0], name='segmentation', scale=scale_vec)

viewer.theme = "dark"

if __name__ == '__main__':
    napari.run()