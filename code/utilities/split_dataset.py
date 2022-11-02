import glob
import os
from shutil import copy


DATA_DIR = 'D:\\UNIVERSITA\\Magistrale\\SecondoAnno\\Tesi\\Datasets\\MapillaryTrafficSignDetection\\DoI_arrows_annotated\\'
TRAIN_TXT_DST = 'train\\labels\\'
TRAIN_JPG_DST = 'train\\images\\'
TEST_TXT_DST = 'test\\labels\\'
TEST_JPG_DST = 'test\\images\\'


# Percentage of images to be used for the test set
percentage_test = 20

counter = 1
index_test = round(100 / percentage_test)

for pathAndFilename in glob.iglob(os.path.join(DATA_DIR, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    # print(f"\ntitle: {title} - extension: {ext}")

    if counter == index_test: # copy in test
        counter = 1
        copy(DATA_DIR+title+ext, TEST_JPG_DST+title+ext)
        copy(DATA_DIR+title+'.txt', TEST_TXT_DST+title+'.txt')
    else: # copy in train
        counter = counter + 1
        copy(DATA_DIR+title+ext, TRAIN_JPG_DST+title+ext)
        copy(DATA_DIR+title+'.txt', TRAIN_TXT_DST+title+'.txt')
