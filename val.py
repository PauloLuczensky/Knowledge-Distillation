import random
from pathlib import Path
import shutil

# Classes
classes = ['healthy', 'leaf blight', 'leaf curl', 'septoria leaf spot', 'verticillium wilt']

# Pastas
images_dir = Path('tomato_yolo/images')
labels_dir = Path('tomato_yolo/labels')
val_split = 0.15  # 15% das imagens para val

# Criar pastas de labels
for split in ['train', 'val', 'test']:
    for cls in classes:
        (labels_dir / split / cls).mkdir(parents=True, exist_ok=True)
        if split == 'val':
            (images_dir / 'val' / cls).mkdir(parents=True, exist_ok=True)

# Criar val e labels
for cls_id, cls in enumerate(classes):
    train_imgs = list((images_dir / 'train' / cls).glob('*.JPG'))
    random.shuffle(train_imgs)
    n_val = int(len(train_imgs) * val_split)
    val_imgs = train_imgs[:n_val]
    train_imgs = train_imgs[n_val:]

    # Copiar imagens para val e criar labels
    for img_path in val_imgs:
        dest_img = images_dir / 'val' / cls / img_path.name
        shutil.copy(img_path, dest_img)
        txt_path = labels_dir / 'val' / cls / f'{img_path.stem}.txt'
        with open(txt_path, 'w') as f:
            f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

    # Criar labels para train
    for img_path in train_imgs:
        txt_path = labels_dir / 'train' / cls / f'{img_path.stem}.txt'
        with open(txt_path, 'w') as f:
            f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

# Criar labels para test
for cls_id, cls in enumerate(classes):
    test_imgs = list((images_dir / 'test' / cls).glob('*.JPG'))
    for img_path in test_imgs:
        txt_path = labels_dir / 'test' / cls / f'{img_path.stem}.txt'
        with open(txt_path, 'w') as f:
            f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

print("Val e labels criados com sucesso!")
