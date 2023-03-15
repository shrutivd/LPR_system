import cv2
import numpy as np
import buildStateTemplates as st
import buildSymbolTemplates as syt

DEBUG = True

def buildTemplates():
    # create state templates
    state_templates = st.createAllStateTemplates()
    # create symbol templates
    symbol_templates = syt.createAllSymbolTemplates()

    return state_templates, symbol_templates


def findMatches(img, state_templates):
    img_features = st.extractDesignFeatures(img)

    bf = cv2.BFMatcher()

    match_size = 0
    state = ''
    matches_list = []
    loop_broken = 0
    for curr_state in st.states:
        curr_template = state_templates[curr_state]
        matches = bf.knnMatch(curr_template[1], img_features[1],k=2)
        verified_matches = []
        for match1, match2 in matches:
            if match1.distance < 0.7 * match2.distance:
                verified_matches.append([match1])
        match_size = len(verified_matches)
        state = curr_state
        matches_list.append(match_size)
        if match_size > 60:
            loop_broken = 1
            break

    if loop_broken == 0:
        match_size = max(matches_list)
        state = st.states[matches_list.index(match_size)]

    return state


def findState(img, state_templates):
    img_features = st.extractDesignFeatures(img)
    match_size = 0
    state = ''
    matches_list = []
    loop_broken = 0
    
    for i in range(len(state_templates)):
        keypoint_matches = {}
        curr_template = state_templates[st.states[i]]
        for j in range(len(curr_template[1])):
            dists = []
            dist_dict = {}
            matches = []
            for k in range(len(img_features[1])):
                dist = np.linalg.norm(curr_template[1][j] - img_features[1][k])
                dists.append(dist)
                dist_dict[dist] = [img_features[0][k].pt[0], img_features[0][k].pt[1], dist]

            # add two shortest distances to matches
            matches.append(dist_dict[min(dists)])
            dists.remove(min(dists))
            matches.append(dist_dict[min(dists)])
            keypoint_matches[curr_template[0][j]] = matches

        # ratio test
        list_pairs_matched_keypoints = []
        for point in curr_template[0]:
            p2 = keypoint_matches[point][0]
            p3 = keypoint_matches[point][1]
            if p2[2] < p3[2] * 0.7:
                list_pairs_matched_keypoints.append(
                    [[point.pt[0], point.pt[1]], [p2[0], p2[1]]])


        match_size = len(list_pairs_matched_keypoints)
        state = st.states[i]
        matches_list.append(match_size)
        if match_size > 60:
            loop_broken = 1
            break

    if loop_broken == 0:
        match_size = max(matches_list)
        state = st.states[matches_list.index(match_size)]

    return state



def get2FeatureMatches(des1,des2):
    matches = []
    for i, v1 in enumerate(des1):
        dmin = 10000
        d2min = 10000
        kp1_idx = None
        kp2_idx = None
        for j, v2 in enumerate(des2):
            d = np.linalg.norm(v1-v2)
            if d < dmin:
                d2min = dmin
                dmin = d
                kp1_idx = i
                kp2_idx = j
            elif dmin < d and d2min > d:
                d2min = d

        m = {"dmin": dmin, "d2min": d2min, "kp1_idx": kp1_idx, "kp2_idx": kp2_idx}
        matches.append(m)

    return matches

def findSymbols(state, images, symbol_templates):
    if state not in symbol_templates.keys():
        print("No symbols found for state: " + state)
        return '#' * len(images)
    plateNumber = ''
    charLocation = []
    charTemplates = symbol_templates[state].keys()
    for image in images:
        img = image.image
        most_feature_matched = []
        img_1_features = syt.extractDesignFeatures(img)
        for template in charTemplates:
            img_2_features = symbol_templates[state][template]
            matches = get2FeatureMatches(img_1_features[1], img_2_features[1])

            list_pairs_matched_keypoints = []
            for m in matches:
                if (m['dmin'] / m['d2min']) < 0.7:
                    list_pairs_matched_keypoints.append([img_1_features[0][m['kp1_idx']].pt,
                                                         img_2_features[0][m['kp2_idx']].pt])

            most_feature_matched.append((len(list_pairs_matched_keypoints),template.split("_")[2]))

        if DEBUG:
            print(sorted(most_feature_matched, reverse=True))
        # cv2.imshow("character",image.image)
        # cv2.waitKey(0)
        noMatches, character = sorted(most_feature_matched, reverse=True)[0]
        if noMatches>5:
            charLocation.append((image.location, character))
    charLocation = sorted(charLocation)
    for _,character in charLocation:
        plateNumber += character
    return plateNumber