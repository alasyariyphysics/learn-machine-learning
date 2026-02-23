# Machine Learning & Deep Learning: A Comprehensive Metadata Guide

This document serves as a high-level technical reference for the various paradigms, algorithms, and use cases within the Machine Learning (ML) and Deep Learning (DL) ecosystem.

## Table of Contents

1. [Supervised Learning](https://www.google.com/search?q=%231-supervised-learning&authuser=1)
   * [Regression](https://www.google.com/search?q=%23a-regression&authuser=1)
   * [Classification](https://www.google.com/search?q=%23b-classification&authuser=1)
2. [Unsupervised Learning](https://www.google.com/search?q=%232-unsupervised-learning&authuser=1)
   * [Clustering](https://www.google.com/search?q=%23a-clustering&authuser=1)
   * [Dimensionality Reduction](https://www.google.com/search?q=%23b-dimensionality-reduction&authuser=1)
   * [Association](https://www.google.com/search?q=%23c-association&authuser=1)
3. [Deep Learning (Neural Networks)](https://www.google.com/search?q=%233-deep-learning-neural-networks&authuser=1)
4. [Ensemble Methods](https://www.google.com/search?q=%234-ensemble-methods&authuser=1)
5. [Reinforcement Learning](https://www.google.com/search?q=%235-reinforcement-learning&authuser=1)
6. [Specialized &amp; Modern Paradigms](https://www.google.com/search?q=%236-specialized--modern-paradigms&authuser=1)

---

## 1. Supervised Learning

The model learns from **labeled** training data (Input **X**→ Output **y**).

### A. Regression (Predicting Continuous Values)

Used when the output variable is a real or continuous value (e.g., salary, weight, temperature).

| Algorithm                                 | Description                                                                          | Example Problem                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| **Linear Regression**               | Fits a straight line (**y**=**m**x**+**c) to model relationships.        | **Physics:**Predicting spring extension based on force (Hooke's Law).         |
| **Polynomial Regression**           | Fits a curve for non-linear relationships.                                           | **Biology:**Modeling bacterial growth rates over time.                        |
| **Support Vector Regression (SVR)** | Finds a function that approximates data within a margin of tolerance (**ϵ**). | **Finance:**Predicting future stock prices based on historical trends.        |
| **Decision Tree Regression**        | Splits data into branches based on a series of decision rules.                       | **Real Estate:**Predicting house prices based on square footage and location. |
| **Random Forest Regression**        | An ensemble of decision trees to improve accuracy and control overfitting.           | **Meteorology:**Predicting rainfall amount (mm) based on humidity/temp.       |

![regression vs classification, AI generated](https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcTphPNn7HfXYaCzOEXr7V9qtOBhxEj475xEsnaxYzELMOZGu3ZUyO6AVMUCn8knjkZ0f4MzKJmwdgN_NnJuOaD6aNkKhvKjD_HAPDbhjnEEz2LQi3E)**Shutterstock**

### B. Classification (Predicting Discrete Categories)

Used when the output variable is a category (e.g., Yes/No, Spam/Not Spam, Digit 0-9).

| Algorithm                               | Description                                                              | Example Problem                                                        |
| --------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| **Logistic Regression**           | Uses a sigmoid function to output a probability (0 to 1).                | **Medicine:**Predicting if a tumor is Malignant (1) or Benign (0).     |
| **K-Nearest Neighbors (K-NN)**    | Classifies a point based on the majority class of its 'k' neighbors.     | **Pattern Recognition:**Recognizing handwritten digits (0-9).          |
| **Support Vector Machines (SVM)** | Finds a hyperplane that maximizes the margin between classes.            | **Neuroscience:**Detecting "Resting" vs. "Active" brain states in EEG. |
| **Naive Bayes**                   | Probabilistic classifier based on Bayes' Theorem (feature independence). | **NLP:**Classifying emails as "Spam" vs. "Inbox."                      |
| **Decision Tree Classification**  | Uses a tree structure to classify data via a sequence of questions.      | **Diagnosis:**Diagnosing a disease based on a symptom checklist.       |

---

## 2. Unsupervised Learning

The model learns from **unlabeled** data, identifying hidden structures or patterns.

### A. Clustering (Grouping Similar Data)

| Algorithm                         | Description                                                          | Example Problem                                                          |
| --------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **K-Means Clustering**      | Partitions data into 'K' clusters based on distance to a centroid.   | **Marketing:**Customer segmentation (Budget vs. Spender).                |
| **Hierarchical Clustering** | Builds a tree of clusters (dendrogram) showing nested relationships. | **Genetics:**Reconstructing evolutionary trees based on DNA.             |
| **DBSCAN**                  | Density-based clustering; finds arbitrary shapes and ignores noise.  | **Astronomy:**Identifying star clusters while ignoring background noise. |

### B. Dimensionality Reduction (Simplifying Data)

Crucial for visualizing high-dimensional data or reducing noise.

| Algorithm | Description | Example Problem |
| :--- | :--- | : :--- |
| **PCA** | Projects data onto orthogonal axes that maximize variance. | **Neuroscience:** Compressing 64 EEG channels into a 2D plot. |
| **t-SNE / UMAP** | Non-linear techniques designed for 2D/3D visualization. | **Genomics:** Visualizing clusters of different cell types in RNA-seq. |
| **ICA** | Separates a multivariate signal into additive subcomponents. | **Signal Processing:** Separating a single voice from background noise. |

### C. Association (Finding Rules)

| Algorithm                 | Description                                                   | Example Problem                                                              |
| ------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Apriori / Eclat** | Identifies frequent itemsets and generates association rules. | **Retail:**Market Basket Analysis (e.g., "People who buy bread buy butter"). |

---

## 3. Deep Learning (Neural Networks)

Models mimicking the human brain using layers of artificial neurons.

| Algorithm                              | Description                                         | Best For                                                      |
| -------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------- |
| **Multi-Layer Perceptron (MLP)** | Simplest DL form with fully connected layers.       | Tabular data classification/regression.                       |
| **CNN**                          | Uses filters to scan data; translation invariant.   | **Computer Vision:**Image classification, YOLO, MRI analysis. |
| **RNN / LSTM / GRU**             | Features "memory" loops to process sequences.       | **Time-Series:**Speech recognition, stock prediction.         |
| **Transformers**                 | Uses "Attention" to process input data in parallel. | **GenAI:**GPT, BERT, Google Translate, AlphaFold.             |
| **Autoencoders**                 | Compresses then reconstructs input data.            | **Anomaly Detection:**Denoising images, fraud detection.      |

![a neural network architecture, AI generated](https://encrypted-tbn1.gstatic.com/licensed-image?q=tbn:ANd9GcSX9oiM6wlGZrbqacJCf0uqfcyZYbffIDF42NuuI0FfLRPZwxS4Mqw7kZHlsnaZs7V3lke450VVDampBIGdZo7KS4y-s3VM-q2wDXMemeEtZA7y6gU)**Getty Images**

---

## 4. Ensemble Methods ("Wisdom of Crowds")

Combining multiple "weak" models to create one strong predictor.

* **Random Forest:** Builds many decision trees in parallel ( **Bagging** ). Robust against overfitting.
* **Gradient Boosting Machines (GBM):** Builds trees sequentially; each tree corrects the previous one's errors.
* **XGBoost / LightGBM / CatBoost:** Optimized, high-speed Gradient Boosting. The current industry standard for tabular data.
* **AdaBoost:** Increases the weight of misclassified points so the next model focuses on "hard" cases.

---

## 5. Reinforcement Learning (RL)

An **Agent** learns by interacting with an **Environment** via **Rewards** and  **Punishments** .

| Algorithm                        | Description                                         | Example Problem                                           |
| -------------------------------- | --------------------------------------------------- | --------------------------------------------------------- |
| **Q-Learning**             | Learns a "Q-table" of best actions for every state. | **Gaming:**Tic-Tac-Toe, simple maze navigation.           |
| **Deep Q-Networks (DQN)**  | Uses a Neural Network to approximate the Q-table.   | **Advanced Gaming:**Atari, Super Mario.                   |
| **Policy Gradients (PPO)** | Optimizes the strategy (policy) directly.           | **Robotics:**Teaching a robot to walk; self-driving cars. |

![reinforcement learning feedback loop, AI generated](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcThnAAr78Ocl6gGEXDM4gXuLRx_3yhwlBMnGHfSiBkJYncn_AYRhSKzM_dtqlVQ7KkeeARkjmM6SP_qj4b5jlI3Lp12qxYP1cJZLtNZ5zBUeERiHKE)**Getty Images**

## 6. Specialized & Modern Paradigms

* **Semi-Supervised Learning:** Combines a small amount of labeled data with a large amount of unlabeled data. Critical for medical imaging.
* **Self-Supervised Learning:** The model generates its own labels (e.g., predicting a hidden word in a sentence). This is the foundation of LLMs.
* **Generative Adversarial Networks (GANs):** A "Generator" creates fake data while a "Discriminator" tries to catch it. Used for DeepFakes and image upscaling.
