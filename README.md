## 📘 Knowledge Distillation with YOLO for Tomato Leaf Disease Detection

Este repositório implementa um pipeline de Knowledge Distillation utilizando YOLOv5 para classificação e detecção de doenças em folhas de tomate.

A ideia central é treinar um modelo Teacher, mais pesado e robusto (YOLOv5x), gerar pseudo-labels e transferir esse conhecimento para um modelo Student, mais leve (YOLOv5n), adequado para uso em dispositivos de borda (edge devices).

### 📂 Estrutura do Projeto
```
KNOWLEDGE-DISTILLATION
├── tomato_yolo/
│   ├── images/
│   │   ├── train/               # imagens de treino
│   │   ├── val/                 # imagens de validação
│   │   └── test/                # imagens de teste
│   │       ├── healthy     
│   │       
├── yolov5/                      # implementação do YOLOv5
├── kd_pipeline.py               # script principal do pipeline
├── val.py                       # script auxiliar de validação
├── labels.py                    # script para organização de labels
└── LICENSE
```
-----

### ⚙️ Pipeline Implementado
1️⃣ Preparação do Dataset

- Estrutura organizada em train/, val/ e test/.

- Geração automática do arquivo data.yaml.

2️⃣ Treinamento do Modelo Teacher (YOLOv5x)

- Modelo robusto e pesado.

- Treinado por 10 épocas (ajustável via parâmetro epochs).

3️⃣ Geração de Pseudo-labels

- O Teacher gera labels preditas para imagens de treino.

- Labels armazenadas em:

    - tomato_yolo/pseudo_labels/

4️⃣ Treinamento do Modelo Student (YOLOv5n)

- Modelo leve e eficiente.

- Treinado utilizando as pseudo-labels.

5️⃣ Avaliação e Comparação

- Métricas principais: Precision, Recall, mAP.

- Comparação de tamanho do modelo e velocidade de inferência (FPS).

-----------

### 🚀 Execução
1️⃣ Clonar o Repositório e Instalar Dependências
```
git clone https://github.com/usuario/knowledge-distillation.git
cd knowledge-distillation
pip install -r requirements.txt
```
2️⃣ Rodar o Pipeline Completo
```
python kd_pipeline.py
```
3️⃣ Estrutura de Saída

Após a execução, você terá:

- runs/teacher/ → pesos e logs do YOLOv5x.

- runs/student/ → pesos e logs do YOLOv5n.

Comparação de métricas diretamente no terminal.

-------------

### 📊 Resultados Obtidos (10 Épocas)

| **Métrica**                  | **Teacher (YOLOv5x)** | **Student (YOLOv5n)** |
|-------------------------------|-----------------------|------------------------|
| **Parâmetros**               | 97M                  | 2.5M                  |
| **GFLOPs**                   | 246.9                | 7.2                    |
| **Tamanho do Modelo**        | ~195 MB              | ~5 MB                  |
| **Tempo de Treino (CPU)**    | ~1h47min             | ~4m30s                 |
| **Precision (P)**            | 0.993                | 0.998                  |
| **Recall (R)**               | 0.969                | 1.000                  |
| **mAP@50**                   | 0.994                | 0.995                  |
| **mAP@50-95**                | 0.901                | 0.995                  |
| **Velocidade de Inferência** | ~1392 ms/imagem      | ~50–70 ms/imagem       |


### 🔍 Insights

- O Teacher é útil para guiar o treinamento, mas é muito pesado para deploy.

- O Student atingiu métricas praticamente iguais com 40x menos parâmetros e 20x mais rápido.

- Indicado para aplicações em Edge Computing e ambientes com restrição de hardware.

### 📖 Referências

[1] YOLOv5 - Ultralytics

[2] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network.

[3] Dataset: Tomato Leaf Diseases (adaptado).