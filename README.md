# 🧬 OncoScan — Breast Cancer Prediction System

A machine learning-powered web application that predicts whether a tumor is **Benign (Non-cancerous)** or **Malignant (Cancerous)** using clinical diagnostic data. Built with **Flask** and trained on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.
> ⚡ Full-stack ML project combining Machine Learning + Web Development for real-time healthcare prediction.

---

## 🚀 Features

- 🧠 **ML-Based Prediction** (SVM, Random Forest, Logistic Regression)
- 📊 **30 Clinical Input Features** (Mean, Standard Error, Worst)
- ⚡ **Real-time Prediction**
- 🧪 **Demo Mode (Sample Data)**
- 📄 Clean and structured UI for medical input
- 🔐 User authentication (Login / Register)
- 📌 Educational clinical disclaimer

---

## 🧠 Model Information

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Samples:** 569
- **Features:** 30
- **Models Used:**
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - Logistic Regression

**Accuracy:**
- Support Vector Machine (SVM): ~97%
- Random Forest: ~98%
- Logistic Regression: ~95%

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS
- **ML Libraries:** Scikit-learn, NumPy, Pandas
- **Database:** SQLite
- **Model Storage:** Pickle (.pkl)

---

## ⚠️ Note on Model Files

Model (`.pkl`) and database (`.db`) files are not included in this repository.

To generate them:
- Run `model_training.ipynb`
- Models and scaler will be created automatically
  

## 📂 Project Structure
```bash
Breast_Cancer_Prediction/
│
├── templates/
│ ├── home.html
│ ├── login.html
│ ├── register.html
│ └── predict.html
│
├── app.py
├── model_training.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```
---

## ▶️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Chetendra-Singh/Breast_Cancer_Prediction.git
cd Breast_Cancer_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```
### 4. Open in Browser
```bash
http://127.0.0.1:5000
```

## 🧪 How It Works
1. User enters medical report values (30 features)
2. Data is scaled using a trained scaler
3. ML model predicts:
   Benign (Non-cancerous) or
   Malignant (Cancerous)
4. Result is displayed instantly

## ⚠️ Disclaimer

This tool is for educational and research purposes only.
It should NOT be used for real medical diagnosis.
Always consult a certified medical professional.

## 🔮 Future Improvements
- 📄 Upload PDF medical reports
- 📷 Image/X-ray based prediction (Deep Learning)
- 📊 Prediction confidence score (%)
- 🌐 Deploy on cloud (Render / Railway)
- 🎨 Enhanced UI/UX

## 👨‍💻 Team

- Chetendra Singh — https://github.com/Chetendra-Singh  
- Charvi Gehija — https://github.com/charvigehija  
- Chesta Bageria — https://github.com/chestabageria  
- Aryan Prajapati — https://github.com/aryxnprajapati  

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!


---


