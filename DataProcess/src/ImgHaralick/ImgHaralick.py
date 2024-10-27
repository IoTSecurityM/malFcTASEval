# -*- coding: utf-8 -*-
import numpy as np
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
import math

class ImgHaralick(object):
    def __init__(self):        
        self.distances = [1]
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
    def get_features(self, binary_path):
        
        with open(binary_path, 'rb') as mal_file:
            mal_biData = np.array(list(mal_file.read()), dtype=np.uint8)
        img_edge = math.ceil(np.sqrt(len(mal_biData)))
        final_img = np.zeros((img_edge * img_edge), dtype=np.uint8)    
        final_img[:len(mal_biData)] = mal_biData       
        img = final_img.reshape((img_edge, img_edge))

        img = (img * 255).astype(np.uint8)
        
        glcm = graycomatrix(img, distances=self.distances, angles=self.angles, symmetric=True, normed=True)
        
        features = []
        properties = ['ASM', 'contrast', 'homogeneity', 'correlation']
        for prop in properties:
            feat = graycoprops(glcm, prop)
            features.extend(feat.flatten())
        
        entroy = -np.sum(glcm * np.log2(glcm + 1e-10), axis=(0, 1))
        features = features + entroy.tolist()[0]
    
        return features
