import cv2
import numpy as np

def fenge(img):
    h,w,_ = img.shape
    w05 = int(w/2)
    h05 = int(h/2)
    beihang_1 = img[:h05,:w05,:]
    beihang_2 = img[:h05,w05:,:]
    beihang_3 = img[h05:,:w05,:]
    beihang_4 = img[h05:,w05:,:]
    cv2.imwrite("./beihang1.jpg",beihang_1)
    cv2.imwrite("./beihang2.jpg",beihang_2)
    cv2.imwrite("./beihang3.jpg",beihang_3)
    cv2.imwrite("./beihang4.jpg",beihang_4)




if __name__ == "__main__":
    img = cv2.imread("./beihang.jpg")
    fenge(img)
    
