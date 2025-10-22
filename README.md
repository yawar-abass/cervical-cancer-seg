# 🧬 Cervical Cancer Cell Detection and Segmentation

A deep learning approach for cervical cell classification and segmentation using CNNs, U-Net, and Detectron2 on the SpikMed dataset.

---

## 📖 Abstract

Cervical cancer remains one of the most prevalent cancers among women worldwide. Early and accurate detection of abnormal cervical cells plays a crucial role in effective diagnosis and treatment.

This project presents a deep learning–based pipeline for cervical cell classification, segmentation, and instance detection, integrating multiple architectures — Convolutional Neural Networks (CNNs) for classification, U-Net for semantic segmentation, and Detectron2 for instance segmentation of overlapping cells. All models were trained and evaluated on the SpikMed dataset, which contains high-resolution microscopic cervical cell images.

---

## 🧠 Methodology

The project comprises three core modules designed to work sequentially:

### 1. Cervical Cell Classification (CNN)

A custom CNN was trained to classify cervical cells into normal and abnormal categories. The network utilized convolutional and pooling layers to extract discriminative features from high-resolution images.

**Key Steps:**

- Input preprocessing and normalization
- Training with cross-entropy loss
- Evaluation using accuracy and F1-score metrics

### 2. Semantic Segmentation (U-Net)

To achieve precise cell boundary detection, a U-Net architecture was implemented. The model effectively segmented individual cells from complex background regions, leveraging skip connections to preserve spatial details.

**Highlights:**

- Encoder–decoder structure based on ResNet backbone
- Data augmentation using Albumentations
- Dice coefficient and IoU metrics for evaluation

### 3. Instance Segmentation (Detectron2)

For overlapping cell scenarios, Detectron2’s Mask R-CNN was applied. It enabled detection and separation of multiple cell instances within a single image, improving the accuracy of segmentation under real-world conditions.

**Configuration:**

- Pretrained COCO weights for initialization
- Fine-tuned using custom SpikMed annotations
- Threshold-based filtering for high-confidence masks

---

## 🧪 Experimental Setup

| Component          | Model      | Dataset | Purpose                     |
| ------------------ | ---------- | ------- | --------------------------- |
| Classification     | CNN        | SpikMed | Label cervical cells        |
| Segmentation       | U-Net      | SpikMed | Pixel-level mask generation |
| Instance Detection | Detectron2 | SpikMed | Overlapping cell detection  |

**Training Details:**

- Framework: PyTorch, Detectron2
- Image Size: 256×256
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss Functions: BCEWithLogits (U-Net), CrossEntropy (CNN)
- Epochs: 20–30 depending on convergence

---

## 📊 Results

| Model      | Task                  | Dice Score | IoU  | Accuracy |
| ---------- | --------------------- | ---------- | ---- | -------- |
| CNN        | Cell Classification   | —          | —    | 0.95     |
| U-Net      | Semantic Segmentation | 0.92       | 0.88 | —        |
| Detectron2 | Instance Segmentation | 0.90       | 0.85 | —        |

Visual outputs include segmentation masks, class activation maps, and overlapping instance predictions.

---

## 🧩 Repository Structure

```plaintext
cervical-cancer-segmentation/
│
├── notebooks/
│   ├── 1_spikemed_classification.ipynb      # CNN for cell classification
│   ├── 2_unet_segmentation.ipynb            # U-Net training & segmentation
│   ├── 3_detectron_overlap.ipynb            # Detectron2 instance segmentation
│
├── src/
│   ├── unet_model.py                        # U-Net architecture
│   ├── dataset.py                            # Dataset handling
│   ├── utils.py                              # Helper functions (optional)
│
├── data/                                    # Local dataset (not uploaded)
└── outputs/                                 # Predictions, metrics, visualizations


```

## Installation

```bash
# Clone the repository
git clone https://github.com/yawar-abass/cervical-cancer-seg.git
cd cervical-cancer-segmentation

# Install dependencies
pip install -r requirements.txt

# Open notebooks for training or inference
jupyter notebook notebooks/

```

## Future work

- Integrate Transformer-based segmentation architectures (e.g., SegFormer or SwinUNet)
- Deploy model inference via a web-based dashboard for real-time analysis
- Extend dataset for multi-class cervical cell detection
- Explore self-supervised and few-shot learning for limited data scenarios
- Optimize overlapping instance segmentation with improved post-processing techniques
