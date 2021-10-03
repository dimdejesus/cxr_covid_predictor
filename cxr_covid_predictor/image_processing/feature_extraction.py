import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage import img_as_ubyte

from skimage.measure import regionprops
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
from scipy.stats import skew

def feature_extract_chunks(img, mask):
    # 4 x 4 Grid
    chunks_per_axis = 4
    # 4 x 4 = 16
    num_chunks = chunks_per_axis**2

    # features per chunk
    features = {}
    for i in range(num_chunks):
        #features[f'area{i}'] = []
        #features[f'perimeter{i}'] = []
        #features[f'eccentricity{i}'] = []
        #features[f'major axis{i}'] = []
        #features[f'minor axis{i}'] = []
        features[f'mean{i}'] = 0
        features[f'variance{i}'] = 0
        features[f'skewness{i}'] = 0
        features[f'uniformity{i}'] = 0
        features[f'snr{i}'] = 0

    # Converting img to gray
    img = rgb2gray(img)

    # split into chunks
    for i in range(chunks_per_axis):
        for j in range(chunks_per_axis):
            chunk_idx = (chunks_per_axis*i) + j
            chunk_width = chunk_height = img.shape[0]//4

            img_chunk = img[ chunk_width * i:chunk_width * (i+1), chunk_height * j:chunk_height * (j+1) ]
            #mask_chunk = mask[ chunk_width * i:chunk_width * (i+1), chunk_height * j:chunk_height * (j+1) ]

            # SHAPE BASED FEATURES
            #prop = regionprops(mask)[0]

            #features[f'area{chunk_idx}'].append( prop['area'] )
            #features[f'perimeter{chunk_idx}'].append( prop['perimeter'] )
            #features[f'eccentricity{chunk_idx}'].append( prop['eccentricity'] )
            #features[f'major axis{chunk_idx}'].append( prop['major_axis_length'] )
            #features[f'minor axis{chunk_idx}'].append( prop['minor_axis_length'] )

            # UNFIROMITY, SKEWNESS, TOTAL MEAN, VARIANCE, SNR
            arr = img_chunk[img_chunk != 0] #remove zeroes (masked out pixels) from image
            
            if len(arr) > 0:
                intensity_lvls = np.unique(arr) # Get all intensity levels
                numel = intensity_lvls.size # Number of intensity levels
                features[f'uniformity{chunk_idx}']= numel 

                mean = np.mean(arr)
                sd = np.std(arr)
                
                features[f'mean{chunk_idx}'] = mean 
                features[f'variance{chunk_idx}'] = np.var(arr) 
                features[f'skewness{chunk_idx}'] = skew(arr) 
                features[f'snr{chunk_idx}'] = np.float64(np.where(sd == 0, 0, mean/sd))
            #else:
            #    features[f'uniformity{chunk_idx}'] = 0 
            #    features[f'mean{chunk_idx}'] = 0 
            #    features[f'variance{chunk_idx}'] = 0 
            #    features[f'skewness{chunk_idx}'] = 0 
            #    features[f'snr{chunk_idx}'] = 0 
            
    return features