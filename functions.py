from config import *

def plot_images(img, titles):

    fig, ax = plt.subplots(1, len(img), figsize=(12,6))

    for i in range(len(img)):

        ax[i].set_title(titles[i])

        ax[i].imshow(img[i])

    plt.show()

    cv.waitKey(0)

def find_countors_area(img):

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    area = 0
    count_countors = 0
    for i in contours:
        area += cv.contourArea(i)
        count_countors+=1
    return area, count_countors

def find_max_area(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv.contourArea)

    img = np.zeros(img.shape,np.uint8)
    img = cv.drawContours(img,[c], 0, (255), -1)

    return img


def get_cell_img(img, img_res):

    index_x_not_zero = []
    index_y_not_zero = []

    for i in range(0,img.shape[0]):
        if sum(img[i,:])>0:
            index_y_not_zero.append(i)

    for j in range(0,img.shape[1]):
        if sum(img[:,j])>0:
            index_x_not_zero.append(j)
    x1 = min(index_x_not_zero)
    x2 = max(index_x_not_zero)
    y1 = min(index_y_not_zero)
    y2 = max(index_y_not_zero)
    img_res = img_res[y1:y2, x1:x2]
    return  img_res


def claster_cell_img(img, clustersNumber, returnAsGray: bool):
    K = clustersNumber

    (h, w) = img.shape[:2]
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    clt = KMeans(n_clusters=K, n_init=K)

    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant_K5 = quant.reshape((h, w, 3))

    img = img.reshape((h, w, 3))

    if returnAsGray:
        quant_K5 = cv.cvtColor(quant_K5, cv.COLOR_BGR2GRAY)
    image_K5 = Image.fromarray(quant_K5)

    color_matrix = image_K5.getcolors()

    color_matrix = [cm[1] for cm in color_matrix]
    print('Color matrix: ', color_matrix)
    color_matrix = np.sort(color_matrix)[::-1]

    img = np.zeros((image_K5.size[0], image_K5.size[1], 3))
    img = np.array(image_K5).copy()

    return img, color_matrix


def wather_flow(img):
    object_img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(object_img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
