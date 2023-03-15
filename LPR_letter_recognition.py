import cv2
import numpy as np
import buildLetterTemplates as lt


def buildTemplates():
    character_templates = lt.createAllCharactersTemplates()

    return character_templates

def getCharacterLocation(keypoints,matches):
    array = []
    for match in matches:
        array.append([keypoints[match].pt[0],keypoints[match].pt[1]])
    return np.mean(array,axis=0)[0]

def findCharacter(img, character_templates):
    img_features = lt.extractDesignFeatures(img)

    bf = cv2.BFMatcher()

    match_size = 0
    character = []
    char_location = []
    for curr_character in lt.characters:
        curr_template = character_templates[curr_character]
        matches = bf.knnMatch(curr_template[1], img_features[1],k=2)
        verified_matches = []
        for match1, match2 in matches:
            if match1.distance < 0.7 * match2.distance:
                verified_matches.append(match1.trainIdx)

        match_size = len(verified_matches)
        #temp = cv2.drawMatchesKnn(character_templates['img' + curr_character],curr_template[0],img,img_features[0],verified_matches,None)
        if match_size > 20:
            char_location.append((getCharacterLocation(img_features[0],verified_matches), curr_character))

    return sorted(char_location)