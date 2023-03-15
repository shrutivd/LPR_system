import cv2
import buildLetterTemplates as lt
import LPR_letter_recognition as lr
from PIL import Image
import os

#num_tests = [3, 6, 3, 6, 23, 4, 29, 14, 5, 4,
#    3, 13, 3, 2, 1, 1, 4, 3, 5, 3, 14,
#    4, 12, 1, 4, 4, 2, 20, 13, 5, 3,
#    4, 4, 3, 1, 1, 4, 9, 4, 2,
#    1, 9, 2, 2, 7, 2, 4, 23, 3, 3]

file_Read = os.listdir("reserve")

if __name__ == "__main__":
    character_temp = lr.buildTemplates()
    #for state in st.states[:2]:
    #print("--------------------", state)
        #for i in range(num_tests[st.states.index(state)]):
    for file in file_Read:
        print(file)

        img = cv2.imread('reserve/'+ file)
        width_scale_measure = 0.95
        height_scale_measure = 0.53
        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
        width_scaled, height_scaled = img.shape[1] * width_scale_measure, img.shape[0] * height_scale_measure
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
        #img_cropped[img_cropped > 120] = 255
        #img_cropped[img_cropped <= 120] = 0
        gray_image = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        gray_image[gray_image > 100], gray_image[gray_image <= 100] = 255, 0

        gray_image = cv2.medianBlur(gray_image, 5)

        binaryImage = gray_image.clip(0, 255)
        #binaryImage = binaryImage - 1
        #print(binaryImage)

        temp = cv2.imshow("test",img_cropped)
        cv2.waitKey(0)

        print(lr.findCharacter(binaryImage, character_temp))
