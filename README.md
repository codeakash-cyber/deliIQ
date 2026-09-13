# DeliIQ - Delivery Intelligence.

## Overview
DeliIQ is an interactive machine learning web application built with Streamlit. It predicts food delivery delay risks before orders are placed or dispatched. By using a 30-minute threshold, the system classifies deliveries as either on-time or delayed based on traffic levels, weather conditions, distance, and rider metrics.

---

## Key Features
* **Home Dashboard:** Provides an overview of the project objectives, target users (customers and restaurant managers), data insights, and system metrics.
* **Prediction Simulator:** Allows users to manually configure variables like traffic level, weather conditions, order type, vehicle type, distance, and location coordinates to test delay probabilities in real-time.
* **Model Performance & Analytics:** Compares four distinct machine learning algorithms using accuracy scores, precision, recall, F1 scores, and visual charts, along with a feature importance graph highlighting the root causes of delays.

---

## Machine Learning Models
The application trains and evaluates the following four models:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting** (Top-performing model for feature importance insights)

---

## Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## File Structure
```text
├── Food_Time new.csv       # Dataset containing delivery logs and features
├── deliIQ.py               # Main Streamlit web application script
└── README.md               # Project documentation
