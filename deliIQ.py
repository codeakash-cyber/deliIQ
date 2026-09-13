import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DeliIQ - Delivery Delay Predictor", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Delay", "Model Performance"])

@st.cache_data
def load_data():
    df = pd.read_csv("Food_Time new.csv")
    
    categorical_columns = ["Traffic_Level", "weather_description", "Type_of_order", "Type_of_vehicle"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    numerical_columns = ["Delivery_person_Age", "Delivery_person_Ratings", "temperature", "humidity", "precipitation", "Distance (km)"]
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    
    df["TARGET"] = pd.to_numeric(df["TARGET"], errors="coerce")
    df = df.dropna(subset=["TARGET"])
    
    columns_to_drop = ["ID", "Delivery_person_ID"]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    gps_columns = ["Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"]
    for col in gps_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    
    df["TARGET"] = (df["TARGET"] > 30).astype(int)
    
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    
    return df, encoders

@st.cache_resource
def train_models(x_train, y_train, x_test, y_test):
    models = {}
    
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(x_train, y_train)
    y_pred_lr = model_lr.predict(x_test)
    models['Logistic Regression'] = {
        'model': model_lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'f1': f1_score(y_test, y_pred_lr, zero_division=0)
    }
    
    model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    model_dt.fit(x_train, y_train)
    y_pred_dt = model_dt.predict(x_test)
    models['Decision Tree'] = {
        'model': model_dt,
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'precision': precision_score(y_test, y_pred_dt, zero_division=0),
        'recall': recall_score(y_test, y_pred_dt, zero_division=0),
        'f1': f1_score(y_test, y_pred_dt, zero_division=0)
    }
    
    model_rf = RandomForestClassifier(n_estimators=3, random_state=42)
    model_rf.fit(x_train, y_train)
    y_pred_rf = model_rf.predict(x_test)
    models['Random Forest'] = {
        'model': model_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'f1': f1_score(y_test, y_pred_rf, zero_division=0)
    }
    
    model_gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_gb.fit(x_train, y_train)
    y_pred_gb = model_gb.predict(x_test)
    models['Gradient Boosting'] = {
        'model': model_gb,
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb, zero_division=0),
        'recall': recall_score(y_test, y_pred_gb, zero_division=0),
        'f1': f1_score(y_test, y_pred_gb, zero_division=0)
    }
    
    return models

df, encoders = load_data()
y = df["TARGET"]
x = df.drop("TARGET", axis=1)
feature_columns = x.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

models = train_models(x_train_scaled, y_train, x_test_scaled, y_test)

if page == "Home":
    st.title("DeliIQ: AI-Powered Food Delivery Time & Delay Risk Analysis")
    st.markdown("### *Predicting delivery delays before they happen to save time and stress.*")
    st.write("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Deliveries Analyzed", len(df))
    with col2:
        st.metric("Delayed Orders Rate", f"{(df['TARGET'].sum() / len(df) * 100):.1f}%")
    with col3:
        st.metric("Top Performing Model", "Gradient Boosting")
    
    st.write("---")
    
    st.subheader("What is this project about?")
    st.write("""
    Normally, food delivery apps only tell you where your food is *after* you have already placed the order. 
    **DeliIQ** is different. It acts as a **smart planning tool** that lets you check the risk of a delay **before** you place an order or send out a delivery rider. 
    
    We set a simple rule: Any delivery taking **more than 30 minutes** is marked as delayed. Our app uses machine learning to look at weather, traffic, and distance to predict if your order will arrive on time.
    """)
    
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Who can use this?")
        st.write("""
        * **For Customers:** You can test different scenarios. For example: *"If I order right now during heavy rain, what are the chances my food gets delayed compared to waiting an hour?"*
        * **For Restaurant Managers:** Managers can use it as a "what-if" tool to see how heavy traffic or long distances will impact delivery times before assigning a delivery partner.
        """)
    
    with c2:
        st.subheader("Key Highlights")
        st.write("""
        * **Interactive Simulator:** Change traffic, weather, and distance settings manually to see instant risk results.
        * **Model Comparison:** We test and compare **4 different AI models** (like Random Forest and Gradient Boosting) to find the most accurate one.
        * **Explainable AI:** Uses feature importance charts to show which exact factors cause late orders.
        """)
        
    st.write("---")
    
    st.subheader("Why DeliIQ is Useful")
    st.write("""
    Unlike basic prediction scripts, DeliIQ is built as a complete interactive tool with practical advantages:
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        * **Proactive Planning:** Instead of tracking orders after they are late, it lets you check risk factors *before* sending out a delivery.
        * **Clear 30-Minute SLA:** Uses a clean binary threshold (On-Time vs. Delayed) instead of messy time estimations.
        """)
    with col_b:
        st.markdown("""
        * **Data Insights:** By integrating Feature Importance charts (via Gradient Boosting), your project mathematically proves the root causes of delays (such as traffic weight versus weather impact), offering complete transparency into why the model makes its choices.
        * **Interactive Simulator:** Fully functional web and terminal apps that let users test different scenarios in real-time.
        """)
        
    st.write("---")
    st.info("**Tech Stack:** Built using Python, Streamlit, Scikit-Learn, Pandas, and Matplotlib.")
    
    st.write("---")

elif page == "Predict Delay":
    st.title("Make a Prediction & Simulate Risks")
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delivery Details")
        traffic = st.selectbox("Traffic Level", list(encoders["Traffic_Level"].classes_))
        traffic_encoded = encoders["Traffic_Level"].transform([traffic])[0]
        
        weather = st.selectbox("Weather Condition", list(encoders["weather_description"].classes_))
        weather_encoded = encoders["weather_description"].transform([weather])[0]
        
        order_type = st.selectbox("Order Type", list(encoders["Type_of_order"].classes_))
        order_encoded = encoders["Type_of_order"].transform([order_type])[0]
        
        vehicle = st.selectbox("Vehicle Type", list(encoders["Type_of_vehicle"].classes_))
        vehicle_encoded = encoders["Type_of_vehicle"].transform([vehicle])[0]
    
    with col2:
        st.subheader("Other Information")
        age = st.slider("Delivery Person Age", 15, 50, 30)
        rating = st.slider("Delivery Person Rating", 1.0, 5.0, 4.5)
        distance = st.slider("Distance (km)", 1.0, 60.0, 10.0)
        temperature = st.slider("Temperature (C)", 5.0, 30.0, 20.0)
        humidity = st.slider("Humidity (%)", 25.0, 100.0, 60.0)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("Restaurant Location")
        rest_lat = st.number_input("Restaurant Latitude", value=12.9)
        rest_long = st.number_input("Restaurant Longitude", value=77.6)
    
    with col4:
        st.write("Delivery Location")
        del_lat = st.number_input("Delivery Latitude", value=13.0)
        del_long = st.number_input("Delivery Longitude", value=77.7)
    
    precipitation = st.slider("Precipitation (mm)", 0.0, 1.5, 0.0)
    
    st.write("---")
    model_choice = st.selectbox("Select Machine Learning Model", list(models.keys()))
    
    if st.button("Predict", type="primary"):
        user_input = pd.DataFrame([[
            traffic_encoded, weather_encoded, order_encoded, vehicle_encoded,
            age, rating, rest_lat, rest_long, del_lat, del_long,
            temperature, humidity, precipitation, distance
        ]], columns=feature_columns)
        
        user_input_scaled = scaler.transform(user_input)
        
        selected_engine = models[model_choice]['model']
        predicted_class = selected_engine.predict(user_input_scaled)[0]
        probability = selected_engine.predict_proba(user_input_scaled)
        
        confidence = round(probability[0][predicted_class] * 100, 2)
        
        st.write("---")
        st.subheader("Prediction Result")
        st.write(f"**Model Used:** {model_choice}")
        
        if predicted_class == 1:
            st.error("Prediction: DELAY (Exceeds 30-minute threshold)")
        else:
            st.success("Prediction: NOT DELAY (Within 30-minute threshold)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Probability / Confidence", f"{confidence}%")
        with col2:
            st.metric("Model Status", "Active")
        with col3:
            st.metric("Model Accuracy", f"{models[model_choice]['accuracy']:.1%}")

elif page == "Model Performance":
    st.title("Model Performance Comparison & Feature Insights")
    st.write("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Accuracy Scores")
        for model_name, metrics in models.items():
            st.write(f"**{model_name}**: {metrics['accuracy']:.2%}")
    
    with col2:
        st.subheader("Accuracy Comparison Chart")
        model_names = list(models.keys())
        accuracies = [models[m]['accuracy'] for m in model_names]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red'])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, 
                    f"{yval:.2%}", ha="center", va="bottom")
        
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    st.write("---")
    st.subheader("Detailed Metrics Table")
    
    metrics_data = {
        'Model': list(models.keys()),
        'Accuracy': [f"{models[m]['accuracy']:.4f}" for m in models.keys()],
        'Precision': [f"{models[m]['precision']:.4f}" for m in models.keys()],
        'Recall': [f"{models[m]['recall']:.4f}" for m in models.keys()],
        'F1 Score': [f"{models[m]['f1']:.4f}" for m in models.keys()]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)
    
    st.write("---")
    st.subheader("Feature Importance Chart (Gradient Boosting)")
    st.write("This vertical bar graph shows which factors influence delivery delays the most according to our top-performing model.")
    
    gb_model = models['Gradient Boosting']['model']
    importances = gb_model.feature_importances_
    
    feat_imp_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='teal')
    ax2.set_xlabel("Relative Importance Score")
    ax2.set_title("Feature Importance for Delivery Delays")
    
    st.pyplot(fig2)
