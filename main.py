import numpy as np
from  functions import *
from config import *
from analyze_clusters import Analayze

images = os.listdir(path_to_images)

for i in images:

    img_name = os.path.join(path_to_images,i)
    img = cv.imread(img_name)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #get all dark objects because mostly back ground and eritrocytes is lightt color
    img_gray[img_gray >  img_gray.mean()] = 0
    img_gray[img_gray != 0] = 255

    #find max area object wich mostly is leyko cell and deliver its from original image
    img_max_area = find_max_area(img_gray)
    object_img = get_cell_img(img_max_area, img)


    cluster_img, color_matrix = claster_cell_img(object_img, 3, returnAsGray=True)


    analize_img = Analayze(color_matrix, cluster_img)
    cores = analize_img.get_cores()
    cytoplasm = analize_img.get_cyto()

    img_dizi = [img, object_img,  cores, cytoplasm]


    titles = ['Original image', 'Cell image', 'Core', 'Cytoplasm']

    plot_images(img_dizi, titles)


