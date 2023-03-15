import cv2

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
    'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
    'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
    'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 
    'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
    'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
    'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']


def extractDesignFeatures(img):
    '''
    Extract the features of a plate to get a standard
    template for a state's license plate design
    '''
    # extract design features from license plate
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    # return state templates
    return (kp, des)


def createAllStateTemplates():
    """
    Create all the state design templates
    """
    state_templates = {}
    for state in states:
        formatted_state = state.lower().replace(' ', '')
        # print(formatted_state)
        state_image = []
        state_image = cv2.imread('templates/state_templates/' + formatted_state + '-license.jpg')
        state_template = extractDesignFeatures(state_image)
        state_templates[state] = state_template

    # return all state design tempates
    return state_templates
