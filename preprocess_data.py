import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from glob import glob
import os
from tqdm.auto import tqdm

# CelebA train datasets 
# raw_data_dir = r'datasets/CelebA/split/train/'
# processed_data_dir = r'datasets/CelebA/processed/train/'

# CelebA validation datasets
# raw_data_dir = r'datasets/CelebA/split/val/'
# processed_data_dir = r'datasets/CelebA/processed/val/'

# CelebA test datasets
raw_data_dir = r'datasets/CelebA/split/test/'
processed_data_dir = r'datasets/CelebA/processed/test'

identity_file = r'datasets/CelebA/identity_CelebA.txt'

identity_map = {}
with open(identity_file, 'r') as f:
  for line in f.readlines():
    parts = line.strip().split()
    img_name = parts[0]
    identity = parts[1]
    identity_map[img_name] = identity

# get visual statement
list_imgs = glob(os.path.join(raw_data_dir,"*.jpg"))
mtcnn = MTCNN(margin=10, select_largest=True, post_process=False, device='cuda:0')

for img_path in tqdm(list_imgs):
    img_name = os.path.basename(img_path) # get image name
    identity = identity_map.get(img_name) # get identity

    if identity is not None:
        img = plt.imread(img_path)
        face = mtcnn(img)

        if face is not None:
            # Creating folders and keeping track of them all
            save_dir = os.path.join(processed_data_dir, identity)  # Create folder by ID
            os.makedirs(save_dir, exist_ok=True)

            # Save rendered face
            face = face.permute(1, 2, 0).int().numpy()
            save_path = os.path.join(save_dir, img_name)
            plt.imsave(save_path, face.astype(np.uint8))
      

      