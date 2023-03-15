import os
import random
import string

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # your path may be different
import cv2

class Renamer():
    '''
    Object to rename all images as per their character recognition
    '''

    def __init__(self, filesDirectory):
        self.fileDir = filesDirectory
        self.filelist = os.listdir(filesDirectory)
        self.image_dict = {}

    def __pouplateDictionary__(self):
        for i in range(ord('0'),ord('Z')+1):
            if(chr(i) not in [':',';','<','=','>','?','@','O']):
                self.image_dict[chr(i)] = []

    def printCharacterFileSegregation(self):
        print(self.image_dict)

    def recognizedCharacter(self,img):
        custom_oem_psm_config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789" +\
                  " --psm 8"
        character = pytesseract.image_to_string(img, config=custom_oem_psm_config)
       # print("------------------")
       # print("Character: " + character)
        return character.replace("\n","")


    def identifyFileNames(self):
        for file in self.filelist:
            img = cv2.imread(self.fileDir+file)
            resized_img = cv2.resize(img,(65,65))
            character = self.recognizedCharacter(resized_img)
            if(character in self.image_dict):
                self.image_dict[character].append(file)
            else:
                self.image_dict[character] = [file]

    def __getStateName__(self,fileName):
        return fileName.split("_")[0]

    def __randomString__(self,length=8):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for _ in range(length))

    def renameFileNames(self):
        if(len(self.image_dict.keys()) == 0):
            print("filenames not populated")

        for character in self.image_dict.keys():
            for itr, file in enumerate(self.image_dict[character]):
                state_name = self.__getStateName__(file)
                rand_string = self.__randomString__()
                os.rename(self.fileDir+file,self.fileDir+state_name+"_"+character+"_"+rand_string+".jpg")

if __name__ == "__main__":
    filepath = "./cropped_images_templates/"
    renamer = Renamer(filepath)
    renamer.identifyFileNames()
    renamer.printCharacterFileSegregation()
    renamer.renameFileNames()