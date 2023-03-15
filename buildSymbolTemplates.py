import glob
import numpy as np

import cv2
from buildStateTemplates import states

PATH = "templates/symbol_templates-1/"

class Image():
    def __init__(self, image):
        self.image = image
        self.location = 0

def extractDesignFeatures(img):
    '''
    Extract the features of a plate to get a standard
    template for a state's license plate design
    '''

    # extract design features from license plate
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 300)
    kernel = np.ones((1,3), np.uint8)  #3X3 is correct dont change it
    dilateEdges = cv2.dilate(canny, kernel, iterations=1)
    canny = cv2.Canny(dilateEdges, 100, 300)
    dilateEdges = cv2.dilate(canny, kernel, iterations=2) #2 7
    canny = cv2.Canny(dilateEdges, 100, 300)
    dilateEdges = cv2.dilate(canny, kernel, iterations=1) #1
    canny = cv2.Canny(dilateEdges, 100, 300)
    dilateEdges = cv2.dilate(canny, kernel, iterations=3) #3 7
    #dilateEdges = cv2.dilate(canny, kernel, iterations=1) ### works for 1
    #canny = cv2.Canny(dilateEdges, 100, 300) ### works for 1
    invertedImg = cv2.bitwise_not(dilateEdges)
    kp, des = sift.detectAndCompute(invertedImg, None)
    #cv2.imshow("img", invertedImg)
    #cv2.waitKey(0)

    # return state templates
    return (kp, des)


def createAllSymbolTemplates():
    """
    Create all the state design templates
    """
    symbol_templates = {}
    for state in states:
        search = PATH + state.lower().replace(' ','') + '.*jpg'
        symbol_list = glob.glob(PATH + state.lower().replace(' ', '') + '*.jpg')
        if (len(symbol_list)):
            symbol_templates[state] = {}
            for symbol in symbol_list:
                symbol_image = cv2.imread(symbol)
                symbol_template = extractDesignFeatures(symbol_image)
                symbol_templates[state][symbol] = symbol_template

    # return all state design tempates
    return symbol_templates
