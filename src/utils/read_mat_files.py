import os
import h5py
import numpy as np
import cv2

def main(original_dir, save_dir, sub_dirs):
    arrays = {}
    for f in os.listdir(original_dir):
        name = f.split('.')[0]
        data = h5py.File(os.path.join(original_dir, f))
        image = np.array(data['cjdata']['image']) 
        label = int(np.array(data['cjdata']['label'])[0][0]) - 1
        sub_dirs[label]
        cv2.imwrite(os.path.join(save_dir, sub_dirs[label],name + '.jpg'), image)


if __name__ == "__main__":
    base_dir = 'C:/Users/ferna/Documents/UFMG/8 Semestre/Causalidade/Dataset/1512427/data'
    base_save_dir = 'C:/Users/ferna/Documents/UFMG/8 Semestre/Causalidade/Dataset/dataset'
    sub_dirs = [ 'benign/meningioma', 'yes', 'benign/pituary']

    main(base_dir, base_save_dir, sub_dirs)