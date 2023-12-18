
import numpy as np
import sklearn
from pickle import load 

model = load(open('D:\CODE-Codespace\VSCode\MACHINE LEARNING\Projects\Crop_recommendation\model.pkl', "rb"))

crop_dict = {
    1:'rice',
    2:'maize',
    3:'chickpea',
    4:'kidneybeans',
    5:'pigeonpeas',
    6:'mothbeans',
    7:'mungbean',
    8:'blackgram',
    9:'lentil',
    10:'pomegranate',
    11:'banana',
    12:'mango',
    13:'grapes',
    14:'watermelon',
    15:'muskmelon',
    16:'apple',
    17:'orange',
    18:'papaya',
    19:'coconut',
    20:'cotton',
    21:'jute',
    22:'coffee'
}

def crop_pred(N, P, K, T, H, ph, R):
    feature_list = [N, P, K, T, H, ph, R]
    feature = np.array([feature_list])
    crop = model.predict(feature)
    return crop_dict[crop[0]]
