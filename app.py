import os

import cv2
import licensePlateRecognition as pr
import extractDriverInformation as di
import imageToData as td
import glob

class App:

    def __init__(self):
        self.state_temp, self.symbol_temp = pr.buildTemplates()
        self.state = "NotSet"
        self.symbols = "NotSet"
        self.driverName = "NotSet"
        self.correctClassification = 0
        self.incorrectClassification = 0
        self.totalClassification = 0

    def getPlateDetails(self, imagePath, printDetails=None, showImage=None):
        img = cv2.imread(imagePath)
        self.state = pr.findState(img, self.state_temp)
        croppedLetters = td.individualSymbols(img)
        self.symbols = pr.findSymbols(self.state, croppedLetters, self.symbol_temp)
        try:
            self.driverName = di.getDriverName(self.state, self.symbols)
            self.correctClassification += 1
        except:
            print("Driver information not available for plate: {0}-{1}".format(self.state,self.symbols))
            self.driverName = "NotFound"
            self.incorrectClassification += 1

        if printDetails:
            self.printPlateDetails()

        if showImage:
            cv2.imshow("Plate Image",img)
            cv2.waitKey(0)

    def getSegmentedImages(self, imagePath, saveCroppedImages=False):
        img = cv2.imread(imagePath)
        if saveCroppedImages:
            td.individualSymbols(img, imagePath)
        else:
            td.individualSymbols(img)

    def printPlateDetails(self):
        # print license plate information
        print('---------------------------------------')
        print('State:', self.state)
        print('License Plate:', self.symbols)
        print("Driver's Name:", self.driverName)
        print('---------------------------------------')

    def printMetrics(self):
        print('---------------------------------------')
        print('Total classifications:', self.totalClassification)
        print('Correct classifications:', self.correctClassification)
        print("Incorrect classifications:", self.incorrectClassification)
        print('---------------------------------------')

if __name__ == "__main__":
    app = App()
    path = 'test'
    # file_list = glob.glob(path + '/*jpg')
    #"test/Alaska_1.jpg", "test/Alaska_3.jpg", "test/Alaska_5.jpg",  "test/Arizona_2.jpg","test/Arkansas_1.jpg", "test/Arkansas_3.jpg", "test/Arkansas_5.jpg", "test/Alaska_2.jpg",
    #"test/Alaska_4.jpg", "test/Alaska_6.jpg", "test/Arizona_3.jpg",  "test/Arkansas_2.jpg",,
    file_list = ["test/Arkansas_4.jpg", "test/Arkansas_6.jpg"]
    for file in file_list:
        app.getPlateDetails(file, True, True)
        # app.getSegmentedImages(file)




