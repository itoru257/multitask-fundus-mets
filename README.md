## Multi-Task Deep Learning for Predicting Metabolic Syndrome from Retinal Fundus Images in a Japanese Health Checkup Dataset

- ✏️ Tohru Itoh, Koichi Nishitsuka, Yasufumi Fukuma, Satoshi Wada 
- 🔗 [medRxiv](https://medrxiv.org/cgi/content/short/2025.05.13.25327551v1)

This repository provides five self-contained Jupyter Notebooks for training and ensembling multiple deep learning models, corresponding to our paper:

---

## 📘 Overview

We train three different neural network models and then make predictions using two ensemble methods..  
The notebooks are organized as follows:

- `train/`: Contains 3 notebooks to train different models
- `ensemble/`: Contains 2 notebooks for ensembling:
  - One without Test-Time Augmentation (TTA)
  - One with Test-Time Augmentation (TTA)

Each notebook is self-contained and includes all training and evaluation steps.

---

## 📁 Folder Structure

```
.
├── train/
│   ├── train_convnext_base-288-METS+AC.ipynb      # ConvNeXt-Base-based training
│   ├── train_seresnext50_32x4d-256-METS+AC.ipynb  # SE-ResNeXt-50-based training
│   ├── train_swinv2_base-256-METS+AC.ipynb        # Swin Transformer V2 Base-based training
│
├── ensemble/
│   ├── ensemble_without_TTA.ipynb    # Ensemble method without TTA
│   ├── ensemble_with_TTA.ipynb       # Ensemble method with TTA
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/itoru257/multitask-fundus-mets.git
cd multitask-fundus-mets
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install other dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open the notebooks under `train/` or `ensemble/` and run them interactively.

---

## 🧠 Step 1: Train Each Model

Run each notebook in `train/` to train the models:

- `train/train_convnext_base-288-METS+AC.ipynb`
- `train/train_seresnext50_32x4d-256-METS+AC.ipynb`
- `train/train_swinv2_base-256-METS+AC.ipynb`

Each model is trained using fundus-specific augmentation, with AC incorporated as an auxiliary task in a multi-task learning.

---

## 🔄 Step 2: Perform Ensembling

After training all models, run one of the following notebooks in the `ensemble/` folder:

- `ensemble/ensemble_without_TTA.ipynb`: The ensemble is performed using simple averaging.
- `ensemble/ensemble_with_TTA.ipynb`: The ensemble is performed using Test-Time Augmentation (TTA).

---

## 📦 Dependencies

This project was tested with the following environment:

- Python 3.12.9
- CUDA 12.1
- PyTorch 2.4.1
- OpenCV 4.11.0
- timm 1.0.15
- albumentations 1.4.17

Other dependencies:

```txt
numpy
pandas
matplotlib
scikit-learn
tqdm
```

---

## 📊 Results Summary

| Method                   | Accuracy (%) | Dataset        | Notes             |
|--------------------------|--------------|----------------|-------------------|
| ConvNeXt-Base            | 66.3         | Validation     | Single model      |
| SE-ResNeXt-50            | 65.6         | Validation     | Single model      |
| Swin Transformer V2 Base | 65.4         | Validation     | Single model      |
| Ensemble Without TTA     | 68.8         | Test           | Simple averaging  |
| Ensemble With TTA        | 69.6         | Test           | TTA-enhanced      |

---

## 📄 Citation

If you use this repository, please cite:

```bibtex
@article{multitask-fundus-mets,
  title={Multi-Task Deep Learning for Predicting Metabolic Syndrome from Retinal Fundus Images in a Japanese Health Checkup Dataset},
  author={Tohru Itoh and Koichi Nishitsuka and Yasufumi Fukuma and Satoshi Wada},
  journal={medRxiv},
  year={2025},
  doi={10.1101/2025.05.13.25327551},
  url={https://medrxiv.org/cgi/content/short/2025.05.13.25327551v1}
}
