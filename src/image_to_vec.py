import glob
import os
from random import shuffle, seed

import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image


Image.MAX_IMAGE_PIXELS = 1000000000
seed(0)
NUMBER_OF_ARTISTS = 10
NUMBER_OF_SAMPLE_IMAGES = 100


def image_to_vec(image, hx=32, hy=32):
    img = load_img(image, target_size=(hx, hy))
    return img_to_array(img) / 255
 

if __name__ == "__main__":
    X_list, y_list = [], []

    artists = os.listdir(path='../artists')
    shuffle(artists)

    for i, artist in enumerate(artists[:NUMBER_OF_ARTISTS]):
        files = glob.glob("../artists/{}/*.jpg".format(artist))
        shuffle(files)

        for image_file in tqdm(files[:NUMBER_OF_SAMPLE_IMAGES]):
            vec = image_to_vec(image_file)
            X_list.append(vec)
            y_list.append(i)

    X, y = np.array(X_list), np.array(y_list)

    if not os.path.exists('../train'):
        os.mkdir('../train')

    with open('../train/id2artists.txt', mode='w') as f:
        for artist in artists[:NUMBER_OF_ARTISTS]:
            f.write('%s\n' % artist)

    np.save('../train/X', X)
    np.save('../train/y', y)
