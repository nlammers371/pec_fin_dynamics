import napari
import numpy as np
from skimage.measure import label, regionprops, regionprops_table
from aicsimageio import AICSImage
import os
import torch

# set parameters
file_name = "zf_bact2-tdTom_fin_48hpf_timeseries02_2023_05_25__20_23_50_462.czi"
read_root = 'D:/Nick/20230525/'
read_path = os.path.join(read_root, file_name)

#############
# Main image
#############

# load in raw czi file
imObject = AICSImage(read_path)
# imData = np.squeeze(imObject.data)
#
# #############
# # Labels
# #############
#
# # read the image data
# reader_lb = Reader(parse_url(readPathLabels))
#
# # nodes may include images, labels etc
# nodes_lb = list(reader_lb())
#
# # first node will be the image pixel data
# label_node = nodes_lb[1]
# label_data = label_node.data
#
# # extract key image attributes
# omero_attrs = image_node.root.zarr.root_attrs['omero']
# channel_metadata = omero_attrs['channels']  # list of channels and relevant info
# multiscale_attrs = image_node.root.zarr.root_attrs['multiscales']
# axis_names = multiscale_attrs[0]['axes']
# dataset_info = multiscale_attrs[0]['datasets']  # list containing scale factors for each axis

# extract useful info
# scale_vec = multiscale_attrs[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
# channel_names = [channel_metadata[i]["label"] for i in range(len(channel_metadata))]
# #colormaps = [channel_metadata[i]["color"] for i in range(len(channel_metadata))]
# colormaps = ["red", "blue", "green", "gray"]
#
# viewer = napari.view_image(imData, channel_axis=0, name=channel_names, colormap=colormaps, scale=scale_vec)
# labels_layer = viewer.add_labels(label_data[0], name='segmentation', scale=scale_vec)
#
# viewer.theme = "dark"

if __name__ == '__main__':
    napari.run()