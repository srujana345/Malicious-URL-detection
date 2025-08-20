# Malicious URL Detection

This project aims to detect malicious URLs using machine learning techniques. URLs are classified into categories such as phishing, malware, defacement, and benign, using a variety of features extracted from the URLs themselves. The core implementation is provided in a Jupyter notebook: [`Malicious_URL_Detection.ipynb`](Malicious_URL_Detection.ipynb).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features Extracted](#features-extracted)
- [Models Used](#models-used)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Overview

Malicious URLs are commonly used for cyber attacks such as phishing, malware distribution, and website defacement. This notebook demonstrates how to extract meaningful features from URLs and train various machine learning models to automatically classify URLs into malicious or benign categories.

## Dataset

The dataset used is [`malicious_phish.csv`](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset), which contains 651,191 URLs labeled as:
- `phishing`
- `malware`
- `defacement`
- `benign`

The notebook downloads the dataset directly from Kaggle using the Kaggle API.

## Features Extracted

Several features are extracted from each URL, including but not limited to:

- Length of URL
- Presence of IP address in URL
- Number of digits
- Use of shortening services (e.g., bit.ly, goo.gl)
- Presence of abnormal patterns or suspicious words (e.g., "login", "account", "paypal")
- Number of directories
- Use of HTTPS
- Port usage
- Subdomain length
- Counts of special characters (e.g., '@', '?', '-', '=', '.', '#', '%', '+', '$', '!', '*', ',')

These features help the models differentiate between benign and malicious URLs.

## Models Used

The following machine learning models are implemented:

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Naive Bayes (GaussianNB)
- XGBoost Classifier

Performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

## Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/srujana345/Malicious-URL-detection.git
   cd Malicious-URL-detection
   ```

2. **Install requirements**

   - Python 3.x is required.
   - Install Jupyter and other dependencies:

     ```bash
     pip install -r requirements.txt
     ```

   - Make sure you have the Kaggle API key (`kaggle.json`) and set it up as described in [Kaggle API documentation](https://github.com/Kaggle/kaggle-api).

3. **Run the Notebook**

   Open the notebook in Jupyter or Google Colab:

   ```bash
   jupyter notebook Malicious_URL_Detection.ipynb
   ```

   Or open it directly in [Google Colab](https://colab.research.google.com/github/srujana345/Malicious-URL-detection/blob/main/Malicious_URL_Detection.ipynb).

4. **Follow the cells in order** to:
   - Download the dataset
   - Preprocess and extract features
   - Train and evaluate models

## Results

The notebook displays the evaluation metrics for all classifiers. Random Forest and XGBoost typically provide the best performance for this type of classification problem.

## References

- [Malicious URLs Dataset on Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## License

This project is licensed under the MIT License.

---

**Author:** [srujana345](https://github.com/srujana345)
