import streamlit as st
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="DeliIQ - Delivery Delay Predictor", layout="wide")

# Set visual style for charts
sns.set_theme(style="whitegrid")

@st.cache_data
def load_and_process_data():
    df = pd.read_csv("Food_Time new.csv")
    
    # Categorical columns
    categorical_columns = ["Traffic_Level","weather_description","Type_of_order", "Type_of_vehicle"]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Numerical columns
    numerical_columns = ["Delivery_person_Age","Delivery_person_Ratings","temperature","humidity", "precipitation","Distance (km)"]

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

    gps_columns = ["Restaurant_latitude","Restaurant_longitude","Delivery_location_latitude","Delivery_location_longitude"]
    
    for col in gps_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    DELAY_THRESHOLD = 30
    df["TARGET"] = (df["TARGET"] > DELAY_THRESHOLD).astype(int)

    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders

df, encoders = load_and_process_data()

y = df["TARGET"]
x = df.drop("TARGET", axis=1)
feature_columns = x.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Models
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred_lr = model.predict(x_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

model1 = DecisionTreeClassifier(max_depth=5, random_state=42)
model1.fit(x_train, y_train)
y_pred_dt = model1.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

model2 = RandomForestClassifier(n_estimators=3, random_state=42)
model2.fit(x_train, y_train)
y_pred_rf = model2.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

model3 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
model3.fit(x_train, y_train)
y_pred_gb = model3.predict(x_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

# Streamlit App Layout
st.title("DeliIQ: AI-Powered Delivery Delay Predictor")
st.write("---")

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose Section", ["Simulator & Prediction", "Model Performance & Analytics"])

if app_mode == "Simulator & Prediction":
    st.subheader("Select Machine Learning Model")
    selected_model_name = st.selectbox(
        "Choose Model", 
        ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    )
    
    if selected_model_name == "Logistic Regression":
        selected_model = model
    elif selected_model_name == "Decision Tree":
        selected_model = model1
    elif selected_model_name == "Random Forest":
        selected_model = model2
    else:
        selected_model = model3

    st.write("---")
    st.subheader("Enter Delivery Information")

    # Reload original raw categories for dropdown display labels
    raw_df = pd.read_csv("Food_Time new.csv")
    
    col1, col2 = st.columns(2)
    with col1:
        traffic = st.selectbox("Traffic Level", raw_df["Traffic_Level"].dropna().unique())
        traffic_encoded = encoders["Traffic_Level"].transform([str(traffic)])[0]
        
        weather = st.selectbox("Weather Condition", raw_df["weather_description"].dropna().unique())
        weather_encoded = encoders["weather_description"].transform([str(weather)])[0]
        
        order_type = st.selectbox("Type of Order", raw_df["Type_of_order"].dropna().unique())
        order_encoded = encoders["Type_of_order"].transform([str(order_type)])[0]
        
        vehicle = st.selectbox("Type of Vehicle", raw_df["Type_of_vehicle"].dropna().unique())
        vehicle_encoded = encoders["Type_of_vehicle"].transform([str(vehicle)])[0]

    with col2:
        age = st.number_input("Delivery Person Age", value=30.0)
        rating = st.number_input("Delivery Person Rating", value=4.5)
        distance = st.number_input("Distance (km)", value=10.0)
        temperature = st.number_input("Temperature", value=25.0)
        humidity = st.number_input("Humidity", value=60.0)

    col3, col4 = st.columns(2)
    with col3:
        restaurant_latitude = st.number_input("Restaurant Latitude", value=12.9)
        restaurant_longitude = st.number_input("Restaurant Longitude", value=77.6)
    with col4:
        delivery_latitude = st.number_input("Delivery Location Latitude", value=13.0)
        delivery_longitude = st.number_input("Delivery Location Longitude", value=77.7)

    precipitation = st.number_input("Precipitation", value=0.0)

    if st.button("Run Prediction", type="primary"):
        user_input = pd.DataFrame([[
            traffic_encoded,
            weather_encoded,
            order_encoded,
            vehicle_encoded,
            age,
            rating,
            restaurant_latitude,
            restaurant_longitude,
            delivery_latitude,
            delivery_longitude,
            temperature,
            humidity,
            precipitation,
            distance
        ]], columns=feature_columns)

        user_input_scaled = scaler.transform(user_input)
        final_result = selected_model.predict(user_input_scaled)
        probability = selected_model.predict_proba(user_input_scaled)

        st.write("---")
        st.subheader("Final Prediction Result")
        st.write(f"**Model Used:** {selected_model_name}")

        predicted_class = final_result[0]
        if predicted_class == 1:
            st.error("Prediction: DELAY")
        else:
            st.success("Prediction: NOT DELAY")

        st.metric("Probability / Confidence", f"{round(probability[0][predicted_class] * 100, 2)}%")

elif app_mode == "Model Performance & Analytics":
    st.subheader("Model Comparison Summary")
    
    model_accuracies = {
        "Logistic Regression": acc_lr,
        "Decision Tree": acc_dt,
        "Random Forest": acc_rf,
        "Gradient Boosting": acc_gb
    }
    
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Logistic Regression", f"{acc_lr:.2f}")
    col_b.metric("Decision Tree", f"{acc_dt:.2f}")
    col_c.metric("Random Forest", f"{acc_rf:.2f}")
    col_d.metric("Gradient Boosting", f"{acc_gb:.2f}")
    
    st.write(f"### Best Model: **{best_model_name}**")

    # Bar chart for accuracies
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'green', 'orange', 'red'])
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Model Accuracies")
    ax.set_ylim(0, 1.1)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, str(round(yval, 2)), ha="center", va="bottom")

    st.pyplot(fig)

    st.write("---")
    st.subheader("Top Reasons for Delay (Feature Importance)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    pd.Series(model2.feature_importances_, index=feature_columns).nlargest(5).plot(kind="barh", ax=ax2, color='teal')
    ax2.set_title("Top Reasons for Delay")
    ax2.set_xlabel("Importance")
    st.pyplot(fig2)
