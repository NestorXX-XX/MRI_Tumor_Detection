
# Brain Tumor Detection Using MRI

## Overview
This project implements an automated system to detect brain tumors in MRI images using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras. The project includes a **web application** for image upload and prediction, as well as a **Jupyter Notebook** for training and evaluating the model.

---

## Features
1. **Web Application**:
   - Upload an MRI image and receive predictions (Tumor/Healthy).
   - Interactive result display with model confidence.
   - Visual representation of input images.

2. **Dataset Preparation**:
   - Automated splitting of the dataset into training, validation, and test sets.
   - Supports balanced data splitting and handles corrupt/non-image files.

3. **Model Training**:
   - Custom CNN and MobileNet-based model for binary classification.
   - Integrated callbacks for early stopping and model checkpointing.
   - Training and validation accuracy/loss visualization.

4. **Evaluation**:
   - Overall accuracy and success percentage calculation.
   - Logs misclassified images for further debugging.

5. **Utilities**:
   - Batch image preprocessing.
   - Automated dataset handling (image checks, folder structure setup).
   - Saves training metadata and charts for reproducibility.

---

## Dataset
The dataset used is the [Brain Tumor Detection MRI Dataset](https://www.kaggle.com/abhranta/brain-tumor-detection-mri), containing MRI scans divided into `yes` (tumor) and `no` (healthy) categories.

### Dataset Structure
The dataset is organized into folders:
```
brain-tumor-mri-dataset/
├── Brain_Tumor_Detection/
│   ├── yes/
│   └── no/
```

### Splitting Strategy
The dataset is split into:
| Split       | Percentage | Purpose                              |
|-------------|------------|--------------------------------------|
| **Training**| 70%        | Train the model                     |
| **Validation**| 15%      | Hyperparameter tuning               |
| **Testing** | 15%        | Evaluate the model on unseen data   |

---

## Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Required libraries listed in `requirements.txt`

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows, use myenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root of the project directory with the following content:
   ```env
   DJANGO_SECRET_KEY="any"
   DJANGO_DEBUG=True
   ```

5. Download the dataset and place it in the appropriate directory.

6. Run the Jupyter Notebook for training and evaluating a new model:
   ```bash
   jupyter notebook model.ipynb
   ```

7. Launch the web application with the model provided:
   ```bash
   python manage.py runserver
   ```

### Usage
- **Web App**:
  1. Navigate to `http://127.0.0.1:8000/`.
  2. Upload an MRI image and get predictions.

- **Notebook**:
  - Train, validate, and evaluate the model interactively.
  - Visualize results and log misclassifications.

---

## Model Details

### Architecture
The model leverages MobileNet as the base for feature extraction, followed by:
- Flatten layer
- Dense layers with ReLU and sigmoid activations
- Dropout layers for regularization

### Metrics
- **Accuracy**: Measures correct predictions.
- **Loss**: Binary cross-entropy for classification.

---

## Evaluation Results
### Summary
| Class       | Total Images | Correct Predictions | Success Percentage |
|-------------|--------------|---------------------|--------------------|
| **Healthy** | 916          | 905                 | 98.80%             |
| **Tumor**   | 326          | 315                 | 96.63%             |

### Overall Accuracy
The model achieves an **average success percentage** of **95.36%** across all categories.

---

## Folder Structure
```
brain-tumor-detection/
├── Tumor_Detection_APP/        # Web application code
├── model.ipynb # Jupyter Notebook for training
├── requirements.txt            # Dependencies
├── models/                     # Saved models and metadata
├── split_dataset/              # Training, validation, test splits
│   ├── train/
│   ├── val/
│   └── test/
└── README.md                   # Project documentation
```

---

## Future Work
- Expand the dataset for improved generalization.
- Explore other deep learning architectures like EfficientNet.
- Add multi-class classification for tumor types.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Dataset: [Brain Tumor Detection MRI](https://www.kaggle.com/abhranta/brain-tumor-detection-mri)
- Frameworks: TensorFlow, Django
