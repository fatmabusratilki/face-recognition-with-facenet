import os 
import pandas as pd
import shutil

dataset_path = "datasets/CelebA/img_align_celeba/img_align_celeba"
csv_path = "datasets/CelebA/list_eval_partition.csv"
output_dir = "datasets/CelebA/split"

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    img_name = row["image_id"]
    partition = row["partition"]

    src_path = os.path.join(dataset_path,img_name)

    if partition == 0:
        dst_path = os.path.join(train_dir, img_name)
    elif partition == 1:
        dst_path = os.path.join(val_dir, img_name)
    elif partition == 2:
        dst_path = os.path.join(test_dir, img_name)
    
    #  Dosyayı yeni klasöre taşı
    shutil.move(src_path, dst_path)

print("Görseller başarıyla ayrıldı!")