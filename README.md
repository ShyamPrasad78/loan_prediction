# 📊 Loan Default Prediction

This project predicts whether a loan will default using machine learning.

## 🚀 Features
- SMOTENC for class imbalance
- XGBoost, LightGBM, CatBoost, and Logistic Regression
- Stacking and threshold tuning
- Evaluation using ROC-AUC, PR-AUC
- 
## 📦 Dataset

Due to GitHub's 100MB file size limit, the dataset used in this project is **not included** in the repository.

### 🔗 Download Manually

Please download the dataset manually from [Kaggle - Lending Club Loan Data]([https://www.kaggle.com/datasets/wordsforthewise/lending-club])

Once downloaded, place the file:

accepted_2007_to_2018Q4.csv

## 📁 Folder Structure
loan-default-prediction/
|

├── data/
|

├── models/
|

├── notebooks/
|

├── src/
|

├── plots/
|

├── requirements.txt
|

└── README.md

## ✅ Results

| Metric     | Value  |
|------------|--------|
| Accuracy   | 82%    |
| ROC-AUC    | 0.843  |
| PR-AUC     | 0.520  |

## 🔧 How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/main.ipynb

#or

python main.py
```

## 📬 Contact

Author: shyamprasad78

GitHub: https://github.com/shyamprasad78


### 💡 Helpful Tools

- **Preview Markdown**:
  - In **VSCode**: Click "Open Preview" or press `Ctrl + Shift + V`.
  - In **Jupyter Notebook**: Add a markdown cell and paste parts of your README to preview.


