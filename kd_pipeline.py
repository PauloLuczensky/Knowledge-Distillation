import os
from pathlib import Path
from ultralytics import YOLO
import time
import bz2

# =========================
# 1️⃣ Configurações do dataset
# =========================

dataset_dir = Path('tomato_yolo')
classes = ['healthy']
nc = len(classes)

# Criar arquivo data.yaml na raiz
data_yaml_path = Path('tomato_yolo_data.yaml')
data_yaml = f"""
train: {dataset_dir}/images/train
val: {dataset_dir}/images/val
test: {dataset_dir}/images/test
nc: {nc}
names: {classes}
"""
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml)
print("Arquivo data.yaml criado com sucesso!")

# =========================
# 2️⃣ Treinar o modelo Teacher (YOLOv5x)
# =========================

print("Treinando o Teacher (YOLOv5x)...")
teacher_model = YOLO('yolov5x.pt')
teacher_model.train(
    data=str(data_yaml_path),
    imgsz=640,
    batch=5,
    epochs=10,
    project='runs/teacher',
    name='yolov5x_tomato',
    exist_ok=True
)

# =========================
# 3️⃣ Gerar pseudo-labels
# =========================

print("Gerando pseudo-labels com o modelo Teacher...")
pseudo_labels_dir = dataset_dir / 'pseudo_labels'
pseudo_labels_dir.mkdir(parents=True, exist_ok=True)

for img_path in (dataset_dir / 'images/train').rglob('*.jpg'):
    results = teacher_model.predict(str(img_path))
    txt_file = pseudo_labels_dir / f'{img_path.stem}.txt'
    # Converter resultados para formato YOLO
    with open(txt_file, 'w') as f:
        for result in results:
            for *xywh, conf, cls in result.boxes.xywh.tolist():
                f.write(f"{int(cls)} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")

# =========================
# 4️⃣ Treinar o modelo Student (YOLOv5n)
# =========================

print("Treinando o Student (YOLOv5n)...")
student_model = YOLO('yolov5n.pt')
student_model.train(
    data=str(data_yaml_path),
    imgsz=640,
    batch=5,
    epochs=10,
    project='runs/student',
    name='yolov5n_distilled',
    exist_ok=True
)

# =========================
# 5️⃣ Avaliar e comparar desempenho
# =========================

print("Avaliação do Teacher...")
teacher_metrics = teacher_model.val(data=str(data_yaml_path))
print(f"Teacher mAP: {teacher_metrics.metrics_map['0.5']:.4f}")

print("Avaliação do Student...")
student_metrics = student_model.val(data=str(data_yaml_path))
print(f"Student mAP: {student_metrics.metrics_map['0.5']:.4f}")

# Tamanho dos modelos
teacher_size = os.path.getsize('runs/teacher/yolov5x_tomato/weights/best.pt') / (1024*1024)
student_size = os.path.getsize('runs/student/yolov5n_distilled/weights/best.pt') / (1024*1024)
print(f"Tamanho do Teacher: {teacher_size:.2f} MB")
print(f"Tamanho do Student: {student_size:.2f} MB")

# FPS em uma imagem de teste
test_img = next(iter((dataset_dir / 'images/test').rglob('*.jpg')), None)
if test_img:
    start = time.time()
    teacher_model.predict(str(test_img))
    teacher_fps = 1 / (time.time() - start)
    print(f"FPS Teacher: {teacher_fps:.2f}")

    start = time.time()
    student_model.predict(str(test_img))
    student_fps = 1 / (time.time() - start)
    print(f"FPS Student: {student_fps:.2f}")
