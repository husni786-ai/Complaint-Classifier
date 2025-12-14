# 🤖 Customer Complaint Categorizer using Naïve Bayes

This project implements a foundational text classification model designed to automatically categorize customer complaints into specific operational areas. It utilizes the **Term Frequency-Inverse Document Frequency (TF-IDF)** technique for text vectorization and the **Multinomial Naïve Bayes** algorithm for classification.

---

## 🎯 Project Goals

The primary objective of this notebook is to establish an end-to-end Machine Learning pipeline for text classification:

1.  **Data Preparation:** Create and structure a labeled dataset of complaint texts.
2.  **Feature Engineering:** Convert raw text data into numerical features suitable for ML model training using TF-IDF.
3.  **Model Training:** Train a probabilistic classifier (`MultinomialNB`) to learn the association between word features and complaint categories.
4.  **Evaluation:** Assess the model's performance metrics (Accuracy, Precision, Recall, F1-Score).
5.  **Prediction:** Demonstrate the utility of the trained model by classifying new, unseen complaints.

## 🛠️ Technology Stack and Dependencies

The project is built using standard Python libraries for data science and machine learning.

| Library | Role |
| :--- | :--- |
| `pandas` | Data structure and manipulation (creating the DataFrame). |
| `scikit-learn` | Core machine learning components (Model, Vectorizer, Metrics). |
| `numpy` | Efficient numerical operations. |

### Installation

To run this project, ensure you have Python installed, and then install the required dependencies:

```bash
pip install pandas scikit-learn numpy# Complaint-Classifier
Simple complaint-classifier project using naive-bayes
