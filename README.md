# 🌲 Algerian Forest Fires – Data Cleaning, EDA & Machine Learning Prediction

This repository contains the full workflow for analyzing and predicting forest fires in Algeria using the **Algerian Forest Fires Dataset**.
It includes **data preprocessing**, **exploratory data analysis (EDA)**, **model training**, and **saved machine-learning artifacts** for deployment.

---

## 📚 **Project Overview**

Forest fires are a major environmental concern in Northern Algeria, particularly in the Bejaia and Sidi Bel-Abbes regions. This project aims to:

* Understand climate and environmental factors influencing fires
* Build models to predict **fire occurrence** or **FWI (Fire Weather Index)**
* Provide reproducible data cleaning and modeling pipelines
* Save trained models for future integration into apps or APIs

---

## 📁 **Repository Structure**

```
│
├── Algerian_forest_fires_dataset_cleaned_dataset.csv   # Cleaned and structured dataset
├── Algerian_forest_fires_dataset_UPDATE.csv            # Raw, original dataset (unprocessed)
│
├── cleaningdatasetandeda.ipynb                         # Data cleaning + EDA notebook
├── modeltraining.ipynb                                 # Model training & evaluation notebook
│
├── ridge.pkl                                           # Trained Ridge Regression model
├── scaler.pkl                                          # Fitted scaler used during training
│
└── README.md                                            # Project documentation
```

---

## 📊 **Dataset Description**

### **Cleaned Dataset (`Algerian_forest_fires_dataset_cleaned_dataset.csv`)**

* **243 rows × 15 columns**
* Includes meteorological features:

  * `Temperature`, `RH`, `Ws`, `Rain`
* Fire weather indices:

  * `FFMC`, `DMC`, `DC`, `ISI`, `BUI`, `FWI`
* Date features:

  * `day`, `month`, `year`
* Labels:

  * `Classes` (`fire` / `not fire`)
* Region:

  * `0` = Bejaia
  * `1` = Sidi Bel-Abbes

### **Raw Dataset (`Algerian_forest_fires_dataset_UPDATE.csv`)**

* 247 rows
* Original dataset with combined data column (requires cleaning)

---

## 📓 Notebooks

### **🔧 `cleaningdatasetandeda.ipynb`**

Covers the full data cleaning pipeline:

* Fixing formatting issues
* Converting raw text lines into structured columns
* Handling missing values
* Exploratory Data Analysis (EDA)

  * correlation heatmaps
  * feature distributions
  * fire vs. non-fire comparisons

---

### **🤖 `modeltraining.ipynb`**

Includes:

* Data splitting
* Feature scaling
* Training ML models (Ridge Regression, etc.)
* Evaluation metrics
* Saving:

  * `ridge.pkl` (trained model)
  * `scaler.pkl` (training-time scaler)

---

## 🧠 **Machine Learning Models**

The main saved model is:

* **Ridge Regression (`ridge.pkl`)**
  Used to predict **FWI** or fire risk based on meteorological and index variables.

The model requires standardized input, handled by:

* **`scaler.pkl`** – ensures consistent preprocessing on new data.

---

## 🚀 **How to Run Locally**

### **1️⃣ Clone the repo**

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

### **2️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

*(If you need a requirements file, I can generate one.)*

### **3️⃣ Run the notebooks**

Use Jupyter or VSCode:

```bash
jupyter notebook
```

### **4️⃣ Load the trained model (example)**

```python
import pickle
import numpy as np

model = pickle.load(open("ridge.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

sample = np.array([[30, 40, 15, 0, 65, 5, 40, 10, 20, 2, 4, 0.5]])
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print("Predicted FWI:", prediction[0])
```

---

## 📈 **Results & Insights**

* Temperature, wind speed, and fire-weather indices correlate strongly with fire occurrence.
* FWI can be predicted with good accuracy using a Ridge Regression model.
* Data cleaning significantly improved dataset quality and model performance.

---

## 🛠️ **Technologies Used**

* Python
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebook
* Pickle

---

## 🤝 **Contributions**

Contributions, issues, and feature requests are welcome!
Feel free to **fork** this project and submit a PR.

---

## 📜 **License**

This project is open-source under the **MIT License**.

