import sys
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class Main(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.count_img = 0
        self.count_pos = 0
        self.img_path = 'images/face_detect/'
        self.img_name = 'face_0.png'

        self.initUI()

    def nex_img(self):
        img_name = self.img_name
        aux = img_name.split('.')
        aux2 = aux[0].split('_')

        img_num = str(int(aux2[1]) + 1)
        aux2[1] = img_num

        aux[0] = '_'.join(aux2)
        self.img_name = '.'.join(aux)

        self.setWindowBackgroud(self.img_path + self.img_name)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        if e.text() == 'y':
            self.count_img += 1
            self.count_pos += 1
            print self.count_pos / float(self.count_img)

            self.nex_img()

        if e.text() == 'n':
            self.count_img += 1
            print self.count_pos / float(self.count_img)

            self.nex_img()

    def setWindowBackgroud(self, path):
        p = QPalette()
        pixmap = QPixmap(path)
        pixmap = pixmap.scaledToWidth(self.width())
        pixmap = pixmap.scaledToHeight(self.height())
        brush = QBrush(Qt.white, pixmap)

        p.setBrush(QPalette.Active, QPalette.Window, brush)
        p.setBrush(QPalette.Inactive, QPalette.Window, brush)
        p.setBrush(QPalette.Disabled, QPalette.Window, brush)
        self.setPalette(p)
        self.show()

    def initUI(self):

        # x and y coordinates on the screen, width, height
        self.setGeometry(100, 100, 500, 500)

        self.setWindowTitle("Detected Faces")
        self.setWindowBackgroud(self.img_path + self.img_name)
        self.show()


def main():

    app = QApplication(sys.argv)

    main = Main()
    main.show()

    sys.exit(app.exec_())


def detect(im_path, casc_paths, max_rect):
    """
    Detect faces using a cascade xml file in an image
    it returns the biggest detected face.

    :type im_path: str
    :param im_path: path to the image

    :type casc_paths: list of str
    :param casc_paths: path to the cascade file

    :type max_rect: int
    :param max_rect: maximum number of rectangles the function
                     is going to check for faces
    :return: rectangle of the detected face and image
    """
    # Read Image
    img = cv2.imread(im_path)

    # Create cascade detectors
    cascades = []
    for c in casc_paths:
        cascades.append(cv2.CascadeClassifier(c))

    # Stack all the faces founded by the detectors
    for c in cascades[:-1]:
        if 'faces' not in locals() or len(faces) == 0:
            faces = c.detectMultiScale(img, 1.4, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20))
        else:
            aux = c.detectMultiScale(img, 1.4, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20, 20))
            if len(aux) > 0:
                faces = np.vstack((faces, aux))

    # If no faces return
    if len(faces) == 0:
        return [], img


    faces[:, 2:] += faces[:, :2]

    # Find the 'max_rect' biggest faces detected
    max_sz = [-float('inf')]
    max_faces = []
    for r in range(len(faces)):
        # if faces[r][0] <= img.shape[0]/2.0 and faces[r][1] <= 3*img.shape[1]/4.0:
        if len(max_faces) < max_rect:
            max_sz.append(abs(faces[r][0] - faces[r][2]))
            max_faces.append(faces[r])
        elif abs(faces[r][0] - faces[r][2]) > max_sz[-1]:
            max_sz[-1] = abs(faces[r][0] - faces[r][2])
            max_faces[-1] = faces[r]
        max_faces.sort(key=lambda x: abs(x[0] - x[2]), reverse=True)
        max_sz.sort(reverse=True)

    # Detect eyes in those faces and reject the ones that do not contain eyes
    max_sz = 0
    for r in max_faces:
        crop_img = img[r[1]:r[3], r[0]:r[2]]
        eye = cascades[-1].detectMultiScale(crop_img, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (30, 30))

        if len(eye) == 0:
            continue
        elif abs(r[0] - r[2]) > max_sz:
            max_sz = abs(r[0] - r[2])
            face = r

    if 'face' not in locals():
        face = max_faces[0]

    return [face], img


def box(rects, img, img_name):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    cv2.imwrite(img_name, img)


if __name__ == '__main__':
    if sys.argv[1] == 'accuracy':
        # Calculate accuracy of the face detector
        main()  # Accuracy 81.05%

    if sys.argv[1] == 'detect':
        # Run Face detector
        cascpaths = ['data/cascade.xml',
                     'data/haarcascade_frontal_default.xml',
                     'data/haarcascade_frontal_alt.xml',
                     'data/haarcascade_frontal_alt2.xml',
                     'data/haarcascade_profileface.xml',
                     'data/haarcascade_eye.xml']
        images_path = '../../Databases/Aging DB/AGE HuPBA/HuPBA_AGE_data_extended.csv'

        with open(images_path) as f:
            content = f.readlines()

            for i in range(len(content)):
                aux = content[i].split(',')
                impath = aux[2]
                print impath

                rects, img = detect('../../Databases/Aging DB/AGE HuPBA/extended/' + impath, cascpaths, 10)
                box(rects, img, 'images/face_detect/face_%i.png' % i)
