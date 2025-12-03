# 🌲 Algerian Forest Fires – Data Cleaning, EDA & Machine Learning Prediction

### **📍 Dataset Overview**

The **Algerian Forest Fires Dataset** contains meteorological and fire-weather measurements collected from two regions in Algeria:

* **Bejaia** (Northeast Algeria)
* **Sidi Bel-Abbès** (Northwest Algeria)

Each region contributes **122 instances**, giving a total of **244 samples**.

### **📅 Time Period**

Data was collected **from June 2012 to September 2012**, capturing the peak of the fire season.

---

### **📊 Dataset Composition**

* **Total instances:** 244
* **Regions:**

  * 122 Bejaia
  * 122 Sidi Bel-Abbès
* **Input attributes:** 11
* **Output attribute:** 1 (target class)
* **Class distribution:**

  * **Fire:** 138 instances
  * **Not fire:** 106 instances

---

### **📑 Attributes Included**

The 11 input features consist of:

* Meteorological variables (Temperature, RH, Wind speed, Rain)
* Fire Weather Index (FWI) system components:

  * FFMC, DMC, DC, ISI, BUI, FWI
* Date-related fields (day, month, year)

The **output attribute** is:

* `Classes` → *fire* / *not fire*

---

If you'd like, I can integrate this directly into the previously generated README or format it as a standalone documentation section.


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

