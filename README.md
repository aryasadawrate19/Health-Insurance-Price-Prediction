# 💸 Medical Insurance Charge Predictor

This Streamlit web app predicts **annual medical insurance premiums** using customer data like age, BMI, smoking status, number of children, and region. It uses an ensemble of machine learning models trained on real-world health insurance data to give fast, personalized predictions — along with insight into what’s driving the cost.

---

## 🧠 About the Model

The prediction model is an **ensemble of three regressors**:
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `XGBRegressor`

These models were trained using a log-transformed target (`log1p(charges)`) to improve prediction performance. The final prediction is the average of all three models’ outputs, transformed back using `expm1`.

### 📊 Model Performance

| Metric                | Value         |
|-----------------------|---------------|
| MAE (Mean Absolute Error) | ₹1,983.70     |
| MSE (Mean Squared Error) | ₹19,060,953.44 |
| R² Score              | 0.88 (88%)    |

> 📌 A high R² value (0.88) indicates that the model explains 88% of the variation in insurance charges.

---

## 📂 Dataset

**Source**: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

**Features Used**:
- `age`: Age of the person
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of dependents
- `smoker`: Smoking status
- `region`: Geographic region (northeast, northwest, southeast, southwest)
- `charges`: Target — the medical insurance cost

### 🧠 Additional Engineered Features:
- `bmi_over_30`: Binary indicator if BMI > 30
- `age_bmi_interaction`: Age × BMI
- `smoker_age_interaction`: Age if smoker, else 0
- `children_over_2`: Binary flag if children > 2

---

## 🎯 Key Features of the Web App

- ✅ **Instant Premium Prediction**
- 🔄 **Bi-directional sliders & inputs** for Age, BMI, and Dependents
- 📏 **Built-in BMI Calculator** (with auto-apply to form)
- 📊 **Risk Assessment** (Low, Medium, High)
- 📉 **Impact Visualization** of factors using bar chart
- 💡 **Personalized Health Tips**
- 🧾 **Transparent Model Metrics & Explanation**
- ❓ **FAQ Section** built in
- 💅 **Fully styled and responsive layout**

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/insurance-premium-predictor.git
cd insurance-premium-predictor
```

## 🚀 Getting Started

### 2. Install Requirements

```bash
pip install -r requirements.txt
```
### 3. Add the trained model
Place your trained `insurance_model.pkl` in the root directory.

### 4. Run the file
bash
```
streamlit run app.py
```

## 🧠 How the Prediction Works

The model uses tree-based regressors trained on log-transformed charges. We engineered several interaction terms and binary indicators to help the model better understand nonlinear relationships. The app then:

- Builds a feature vector from your input

- Applies the same preprocessing as training

- Predicts using the trained ensemble

- Returns the estimated premium and risk label


## FAQ

Q: What’s the most influential factor?
A: Smoking. It can increase your premium by 3–4×.

Q: How accurate is the prediction?
A: The model explains 88% of the variation in real charges, and is usually off by around ₹1,900.

Q: How can I reduce my premium?
A: Stop smoking, maintain a healthy BMI (< 30), and practice preventive health.
