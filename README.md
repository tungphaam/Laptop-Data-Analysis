# 💻 Data Analysis on Laptop Dataset

This project explores and visualizes patterns within a laptop dataset — analyzing price variations based on brand, specifications, screen size, and performance-related features. It demonstrates data cleaning, exploratory data analysis (EDA), and predictive modeling using Python’s data science stack.

---

## 📊 Project Overview

The primary goal of this project is to uncover the factors that most strongly influence laptop prices and to communicate these insights through clear visualizations and models.

**Key objectives:**

* Perform thorough data cleaning and feature preprocessing
* Analyze feature correlations and distributions
* Visualize brand and specification trends affecting price
* Build and evaluate a regression model to predict laptop prices

---

## 🧠 Tools & Libraries

* **Python 3.9+**
* pandas, numpy
* seaborn, matplotlib
* scikit-learn
* warnings (for cleaner output)

---

## 📁 Repository Structure

```
Laptop-Data-Analysis/
│
├── notebooks/
│   └── Data_Analysis_on_Laptop_dataset.ipynb
│
├── data/
│   └── laptop_data.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── visualization.py
│   └── modeling.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🔍 Key Insights (Examples)

* **Brand** and **Processor type** are among the strongest predictors of price.
* **RAM** and **Storage capacity** also show a positive correlation with laptop price.
* **Touchscreen** and **Display resolution** significantly impact high-end laptop pricing.

---

## ⚙️ How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/Laptop-Data-Analysis.git
   cd Laptop-Data-Analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:

   ```bash
   jupyter notebook notebooks/Data_Analysis_on_Laptop_dataset.ipynb
   ```

4. Run all cells to reproduce the analysis and visualizations.

---

## 📈 Example Visualizations

* Distribution of laptop prices by brand
* Correlation heatmap between features
* Regression model predictions vs actual prices

---

## 🧩 Future Improvements

* Add feature importance plots from advanced models
* Extend analysis to gaming vs productivity laptops
* Deploy as an interactive dashboard (Streamlit/Plotly)

---

## 👤 Author

**Tung Pham**
Data Analyst | Python | Visualization | Machine Learning
📫 [Your email or LinkedIn here]

