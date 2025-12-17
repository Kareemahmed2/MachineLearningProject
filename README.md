# Material Stream Identification System

A machine learning-based waste material classification system using SVM and k-NN classifiers.

## 🎯 Project Overview

This project implements an **Automated Material Stream Identification (MSI) System** for classifying waste materials into seven categories:

| ID | Class | Description |
|----|-------|-------------|
| 0 | Glass | Bottles, jars |
| 1 | Paper | Newspapers, office paper |
| 2 | Cardboard | Boxes, cardboard sheets |
| 3 | Plastic | Water bottles, plastic film |
| 4 | Metal | Aluminum cans, steel scrap |
| 5 | Trash | Non-recyclable waste |
| 6 | Unknown | Out-of-distribution items |

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train SVM classifier
python Train_SVM.py

# Train k-NN classifier
python Train_KNN.py
```

### Real-Time Classification

```bash
# Run with SVM (default)
python realtime_camera.py

# Run with k-NN
python realtime_camera.py --model knn

# Use specific camera
python realtime_camera.py --camera 1
```

**Controls:**
- `q` - Quit
- `m` - Switch model (SVM ↔ k-NN)
- `s` - Save screenshot

## 📁 Project Structure

```
MachineLearningProject/
├── config.py                  # Configuration and constants
├── ImageLoader.py             # Data loading and augmentation
├── FeatureExtractor_SVM.py    # SVM feature extraction
├── FeatureExtractor_KNN.py    # k-NN feature extraction
├── Train_SVM.py               # SVM training script
├── Train_KNN.py               # k-NN training script
├── realtime_camera.py         # Real-time application
├── requirements.txt           # Dependencies
├── Technical_Report.md        # Documentation
├── test_set/                  # Dataset
└── models/                    # Saved models
```

## 🔧 Feature Extraction

### SVM Features (~2300 dimensions)
- HOG (Histogram of Oriented Gradients)
- Color Histogram (LAB)
- Multi-scale LBP
- GLCM Texture Features
- Entropy Statistics
- Edge Features
- Hu Moments
- Color Moments

### k-NN Features (~1090 dimensions)
- Color Histogram (HSV)
- LBP Texture
- Intensity Statistics
- Edge Distribution
- Gabor Texture
- Shape Features
- Dominant Colors

## 📊 Performance

| Metric | SVM | k-NN |
|--------|-----|------|
| Target Accuracy | ≥85% | ≥85% |
| Rejection Mechanism | Probability-based | Distance + Voting |

## 👥 Team

- [Add team member names]

## 📝 License

Cairo University - Faculty of Computing and Artificial Intelligence