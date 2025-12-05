import cv2
import numpy as np

def rgb_seg(image):
    # SEGMENTATION RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    cond1 = (R > 95) & (G > 40) & (B > 20)
    cond2 = (np.max(img_rgb, axis=2) - np.min(img_rgb, axis=2)) > 15
    cond3 = (np.abs(R - G) > 15) & (R > G) & (R > B)

    return (cond1 & cond2 & cond3).astype(np.uint8) * 255


def ycrcb_seg(image):

    # segmentation de contenu en fonction de la couleur de peau en utilisant HSV #
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

    return cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)


def hsv_seg(image):
    # segmentation de contenu en fonction de la couleur de peau en utilisant HSV #
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
