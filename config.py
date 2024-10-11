import os
import cv2 as cv
import numpy as np
from  matplotlib import pyplot as plt
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

ppm = 29.270

path_to_images = 'images'
