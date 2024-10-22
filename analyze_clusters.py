from config import *

import  functions

class Analayze:
    def __init__(self, color_matrix, cluster_img):

        self.color_matrix = color_matrix
        self.cluser_img = cluster_img
        self.core_color = min(self.color_matrix)
        self.cyto_color = max(self.color_matrix)
        self.h, self.w = self.cluser_img.shape
        self.kernel = np.ones((5, 5), np.uint8)


    # method extract cores of cell
    def get_cores(self):
        self.cores = np.zeros([self.h,self.w], dtype=np.uint8)
        for i in range(0,self.cluser_img.shape[0]):
            for j in range(0, self.cluser_img.shape[1]):
                if self.cluser_img[i,j] == self.core_color:
                    self.cores[i,j] = 0
                else:
                    self.cores[i, j] = 1

        self.cores = cv.dilate(self.cores, self.kernel, iterations=3)
        area, count_conturs = functions.find_countors_area(self.cores)
        print(area,count_conturs)

        return  self.cores

    def get_cyto(self):

        self.cyto = np.zeros([self.h,self.w], dtype=np.uint8)
        for i in range(0,self.cluser_img.shape[0]):
            for j in range(0, self.cluser_img.shape[1]):
                if self.cluser_img[i,j] == self.cyto_color:
                    self.cyto[i,j] = 0
                else:
                    self.cyto[i, j] = 1
        self.cyto = cv.erode(self.cyto, self.kernel, iterations=6)
        self.cyto = cv.dilate(self.cyto, self.kernel, iterations=4)

        self.cyto =functions.find_max_area(self.cyto)
        return self.cyto

    def show(self):
        plt.imshow(self.cyto)
        plt.show()
        cv.waitKey(0)

