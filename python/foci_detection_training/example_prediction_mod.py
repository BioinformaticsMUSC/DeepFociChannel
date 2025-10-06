import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tifffile import TiffWriter
from tifffile import imread, imwrite 
from scipy.ndimage import zoom
import sys

sys.path.insert(0, "../utils")
from utils.predict_by_parts import predict_by_parts
from norm_percentile_nocrop import norm_percentile_nocrop, thresholding
from utils.mat2gray import mat2gray
from evaluate_detections import detect


folder_name_to_evaluate = '/scratch/lutsky/deepfoci/DeepFociChannel/python/utils/tiffs/20b_control_53bp1R_nestinY_3'

detection_channel = 1 # red 0, green 1, red and green 2


resized_img_size = [505, 681, 48] #image is resized to this size

normalization_percentile = 0.025#image is normalized into this percentile range

crop_size = [96, 96]

model = torch.load('./detection_model.pt', weights_only=False)


device = torch.device("cuda:0")
model = model.to(device)

filename_sample = folder_name_to_evaluate.split("/")[-1]

img_filename = folder_name_to_evaluate + '/' + filename_sample + "_marker0.tiff"


img = []
img.append(np.transpose(imread(img_filename)[0,...],(1, 2, 0)))
img.append(np.transpose(imread(img_filename.replace('marker0', 'marker2'))[0, ...], (1,2,0)))

img = np.stack(img,axis=3)

img_orig_size = img.shape[:3]
factor =  np.array(resized_img_size) / np.array(img_orig_size)


tmp_size = resized_img_size.copy()
tmp_size.append(img.shape[3])
img_resized = np.zeros(tmp_size,dtype=np.float32)


for channel in range(img.shape[3]):
    
    
    data_one_channel = img[...,channel]
    print(data_one_channel.shape)
    print(factor.shape) 
    print("data_one_channel.ndim =", data_one_channel.ndim)  # should be 3
    print("len(factor) =", len(np.array(factor)))
    print("img_orig_size =", img_orig_size)                  # tuple of 3
    print("resized_img_size =", resized_img_size)            # if this shows 4 elems, that’s the issue
    data_one_channel = zoom(data_one_channel,factor,order=1)
    data_one_channel = norm_percentile_nocrop(data_one_channel,normalization_percentile);
    #if channel == 0:
    #    data_one_channel, threshold = thresholding(data_one_channel, method="isodata")
    #    print("Threshold is: ", threshold)
    img_resized[...,channel] = data_one_channel
    
img = img_resized


img = img.astype(np.float32)
img = np.transpose(img,(3,0,1,2)).copy()
img = torch.from_numpy(img)

img = img.to(device)


res = predict_by_parts(model,img, crop_size=crop_size)



img = img.detach().cpu().numpy()
res = res.detach().cpu().numpy()



plt.imshow(np.max(res[0,:,:,:], axis=2))
plt.show()
plt.savefig("Predicted.png") 
postprocessing_params = model.postprocessing_params[detection_channel]

detected_points = detect(res[detection_channel,:,:],postprocessing_params['T'],postprocessing_params['h'],postprocessing_params['d'])
detected_points = np.array(detected_points)
print(len(detected_points))
image = mat2gray(np.transpose(np.max(img, axis=3), [1, 2, 0]))

# Pad to RGB
if image.shape[2] == 2:
    image = np.concatenate([image, np.zeros((*image.shape[:2], 1))], axis=2)
print(image.shape)
image[:, :, 0] = 0

plt.imshow(image)
plt.show()
plt.savefig("image.png")

plt.imshow(np.max(res[0,:,:,:],axis=2))
plt.plot(detected_points[:,1],detected_points[:,0],'r.')
plt.show()
plt.savefig("image_results.png")




