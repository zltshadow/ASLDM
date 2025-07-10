# ASLDM

Code for **"Anatomy-aware Sketch-guided Latent Diffusion Model for Orbital Tumor Multi-Parametric MRI Missing Modalities Synthesis"**

This repository contains code and example data for training and testing the ASLDM model on the OTTS dataset, enabling synthesis of missing modalities from multi-parametric MRI using anatomical sketch guidance.

---

当然可以，以下是你需要的 Markdown 格式内容，适用于 GitHub 项目的 `README.md` 文件：

---

## 🧠 OTTS Example Data & Pre-trained Weights

Please download the two pre-trained model weight files from the following Google Drive links:

- [ASLDM Autoencoder Weights](https://drive.google.com/file/d/19xlAWUPOSjpvAb7QHTdGLvASc21iNYOG/view?usp=drive_link)  
- [ASLDM UNet Weights](https://drive.google.com/file/d/19U5g82l7PrM_NmDoJtU5h8dLARfWMlqO/view?usp=sharing)

After downloading, place these weight files into the `checkpoints/` directory of the project. For example:

```
ASLDM/
├── checkpoints/
│   ├── autoencoder.pt
│   └── diffusion_unet.pt
```

> ⚠️ **Note:** Ensure the weight file names match those specified in the code to avoid loading errors.

---

Below are examples of each modality used in our model:
<table> <tr> <td align="center"><b>T1WI</b><br> <img src="data/test/t1n/OTTS-0216-LYM-TR-PH-t1n-slice_032.png" width="180"><br> <i>Anatomical structure</i> </td> <td align="center"><b>T2WI</b><br> <img src="data/test/t2w/OTTS-0216-LYM-TR-PH-t2w-slice_032.png" width="180"><br> <i>Edema/fluid sensitivity</i> </td> <td align="center"><b>T1CE</b><br> <img src="data/test/t1c/OTTS-0216-LYM-TR-PH-t1c-slice_032.png" width="180"><br> <i>Contrast-enhanced lesion</i> </td> <td align="center"><b>DWI</b><br> <img src="data/test/dwi/OTTS-0216-LYM-TR-PH-dwi-slice_032.png" width="180"><br> <i>Diffusion signal</i> </td> </tr> <tr> <td align="center"><b>ADC</b><br> <img src="data/test/adc/OTTS-0216-LYM-TR-PH-adc-slice_032.png" width="180"><br> <i>Quantitative diffusion</i> </td> <td align="center"><b>Seg</b><br> <img src="data/test/seg/OTTS-0216-LYM-TR-PH-seg-slice_032.png" width="180"><br> <i>Tumor mask</i> </td> <td align="center"><b>Sketch</b><br> <img src="data/test/sketch/OTTS-0216-LYM-TR-PH-sketch-slice_032.png" width="180"><br> <i>Structural prior</i> </td> <td></td> </tr> </table>

---

## ⚙️ Setup Instructions

### 1. Environment Setup

We recommend using Python 3.9+ and creating a virtual environment:

```bash
conda create -n asldm python=3.12
conda activate asldm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Additional dependencies might include:

* `torch >= 2.0`
* `torchvision`
* `monai`
* `scikit-image`
* `pillow`
* `numpy`
* `tqdm`
* `itertools`
* `tqdm`
* `natsort`

### 3. Dataset

Place the data in the `data/test/` directory. File names should follow:

```
<data_root>/<modality>/<sample_id>-<modality>-slice_<slice_id>.png
```

---

## 🚀 Inference Example

Run inference with a sketch input and available modalities:

```bash
python test.py 
```

---

## 📂 Folder Structure

```
ASLDM/
├── data/
│   └── test/
│       ├── t1n/
│       ├── t2w/
│       ├── t1c/
│       ├── dwi/
│       ├── adc/
│       ├── seg/
│       └── sketch/
├── models/
├── test.py
├── train.py
└── README.md
```
