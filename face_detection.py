import cv2
import numpy as np

def detect(im_path, casc_paths):
    """
    Detect faces using a cascade xml file in an image
    it returns the biggest detected face.

    :param im_path: path to the image
    :param casc_paths: path to the cascade file
    :return: rectangle of the detected face and image
    """
    img = cv2.imread(im_path)
    cascade1 = cv2.CascadeClassifier(casc_paths[0])
    cascade2 = cv2.CascadeClassifier(casc_paths[1])
    cascade3 = cv2.CascadeClassifier(casc_paths[2])
    cascade4 = cv2.CascadeClassifier(casc_paths[3])
    cascade5 = cv2.CascadeClassifier(casc_paths[4])

    rects = cascade1.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20))
    np.append(rects, cascade2.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20)))
    np.append(rects, cascade3.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20)))
    np.append(rects, cascade4.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20)))
    np.append(rects, cascade5.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20)))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]

    # filt_rects = []
    max_sz = 0
    face = []
    for r in range(len(rects)):
        if rects[r][0] <= img.shape[0]/2.0 and rects[r][1] <= 3*img.shape[1]/4.0:
            if abs(rects[r][0] - rects[r][2]) > max_sz:
                max_sz = abs(rects[r][0] - rects[r][2])
                face = [rects[r]]
            # filt_rects.append(rects[r])

    # filt_rects = [[450, 450, 500, 500]]

    return face, img


def box(rects, img, img_name):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    cv2.imwrite(img_name, img)

cascpaths = ['data/cascade.xml',
             'data/haarcascade_frontal_default.xml',
             'data/haarcascade_frontal_alt.xml',
             'data/haarcascade_frontal_alt2.xml',
             'data/haarcascade_profileface.xml']
images_path = '../../Databases/Aging DB/AGE HuPBA/HuPBA_AGE_data_extended.csv'

with open(images_path) as f:
    content = f.readlines()

    for i in range(len(content)):
        aux = content[i].split(',')
        impath = aux[2]

        rects, img = detect('../../Databases/Aging DB/AGE HuPBA/extended/' + impath, cascpaths)
        box(rects, img, 'images/face_detect/face_%i.png' % i)