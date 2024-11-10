---

# Ch.-2-Machine-Learning-and-Deep-Learning

This repository includes a set of machine learning and deep learning assignments aimed at building practical skills in classification, clustering, regression, and fraud detection using different algorithms and frameworks.

---

## Notebooks Overview

### 1. Case 1: Customer Exit Prediction (`02_Kelompok_B_1.ipynb`)

This task focuses on building a classification model to predict whether a bank customer will exit, based on features in a banking dataset.

- **Dataset**: `SC_HW1_bank_data.csv`
  
- **Libraries Used**: `Pandas`, `NumPy`, `Scikit-learn`

- **Steps**:
  1. **Data Preprocessing**: Removal of irrelevant columns, one-hot encoding, and normalization.
  2. **Modeling**: Three models—Random Forest, SVC, and Gradient Boosting—are trained and evaluated.
  3. **Hyperparameter Tuning**: Grid search for the optimal model parameters.
  4. **Evaluation**: Accuracy, classification report, and confusion matrix metrics to assess model performance.

- **Outcome**: Gradient Boosting Classifier provided the highest accuracy with efficient processing time compared to other models.

---

### 2. Case 2: Data Segmentation with KMeans Clustering (`02_Kelompok_B_2.ipynb`)

This task involves clustering customer data using unsupervised learning, specifically the KMeans algorithm.

- **Dataset**: `cluster_s1.csv`
  
- **Libraries Used**: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`

- **Steps**:
  1. **Data Preparation**: Removal of irrelevant columns and data preprocessing.
  2. **Optimal Cluster Count**: Silhouette Score helps determine the best number of clusters.
  3. **Modeling with KMeans**: Cluster the data and visualize results.

- **Outcome**: The best k-value was selected based on the highest Silhouette Score, and a scatter plot visualizes the data clustering.

---

### 3. Case 3: California House Price Prediction with Neural Networks (`02_Kelompok_B_3.ipynb`)

This assignment applies a neural network model built with TensorFlow-Keras to predict house prices based on California housing data.

- **Dataset**: California House Price dataset from `Scikit-Learn`
  
- **Libraries Used**: `Pandas`, `NumPy`, `TensorFlow`, `Keras`, `Scikit-learn`, `Matplotlib`

- **Steps**:
  1. **Data Preparation**: Data split into training, validation, and test sets, with standardization and normalization.
  2. **Model Building**: A Multilayer Perceptron (MLP) with dual inputs is created.
  3. **Training and Evaluation**: Model training, monitoring loss to avoid overfitting.
  4. **Model Saving**: Save the trained model for future predictions.

- **Outcome**: The neural network effectively predicts house prices, with model metrics and visualizations showcasing its performance.

---

### 4. Case 4: Fraud Detection in Credit Card Transactions (`02_Kelompok_B_4.ipynb`)

This case involves building a classification model using PyTorch to detect fraudulent transactions in a credit card dataset.

- **Dataset**: Credit Card Fraud 2023 dataset
  
- **Libraries Used**: `Pandas`, `cuDF`, `cuML`, `NumPy (cuPy)`, `Scikit-learn`, `PyTorch`

- **Steps**:
  1. **Dataset Import with GPU**: Loading and preprocessing the data on GPU for efficiency.
  2. **Data Conversion**: Convert data into tensors and prepare it for PyTorch DataLoader.
  3. **Model Building**: Design a Multilayer Perceptron with four hidden layers in PyTorch.
  4. **Training and Evaluation**: Train the model and fine-tune for an accuracy target of at least 95%.

- **Outcome**: The model achieves a high accuracy, indicating effective fraud detection capabilities.

---

## Running the Notebooks

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `02_Kelompok_B_1.ipynb` through `02_Kelompok_B_4.ipynb` files to Colab.
3. Execute each cell in order, following the instructions within each notebook for smooth operation and analysis.

---
