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

# Set visual style for charts
sns.set_theme(style="whitegrid")

df = pd.read_csv("Food_Time new.csv")
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Categorical columns
categorical_columns = [
    "Traffic_Level",
    "weather_description",
    "Type_of_order",
    "Type_of_vehicle"
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Numerical columns
numerical_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "temperature",
    "humidity",
    "precipitation",
    "Distance (km)"
]

for col in numerical_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

df["TARGET"] = pd.to_numeric(df["TARGET"], errors="coerce")

df = df.dropna(subset=["TARGET"])

print("\nNumber of Unique TARGET Values:")
print(df["TARGET"].nunique())

columns_to_drop = ["ID", "Delivery_person_ID"]

for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)

gps_columns = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude"
]

for col in gps_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

DELAY_THRESHOLD = 30

df["TARGET"] = (df["TARGET"] > DELAY_THRESHOLD).astype(int)

print("\nTarget Distribution:")
print(df["TARGET"].value_counts())

print("\nTarget Meaning:")
print("0 = Not Delay")
print("1 = Delay")

encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    encoders[col] = encoder

print("\nMissing Values:")
print(df.isnull().sum())

y = df["TARGET"]
x = df.drop("TARGET", axis=1)

feature_columns = x.columns.tolist()

print("\nFeature Columns:")
print(feature_columns)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# MODEL 1: LOGISTIC REGRESSION
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

acc_lr = accuracy_score(y_test, y_pred)
rec_score = recall_score(y_test, y_pred, zero_division=0)
pre_score = precision_score(y_test, y_pred, zero_division=0)
fscore = f1_score(y_test, y_pred, zero_division=0)

print("-- Logistic Regression model --")
print("Accuracy_score :", acc_lr)
print("Recall_score   :", rec_score)
print("Precision_score:", pre_score)
print("F1_score       :", fscore)


# MODEL 2: DECISION TREE
model1 = DecisionTreeClassifier(max_depth=5, random_state=42)
model1.fit(x_train, y_train)
y_pred = model1.predict(x_test)

acc_dt = accuracy_score(y_test, y_pred)
rec_score = recall_score(y_test, y_pred, zero_division=0)
pre_score = precision_score(y_test, y_pred, zero_division=0)
fscore = f1_score(y_test, y_pred, zero_division=0)

print("-- Decision Tree model --")
print("Accuracy_score :", acc_dt)
print("Recall_score   :", rec_score)
print("Precision_score:", pre_score)
print("F1_score       :", fscore)


# MODEL 3: RANDOM FOREST
model2 = RandomForestClassifier(n_estimators=3, random_state=42)
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)

acc_rf = accuracy_score(y_test, y_pred)
rec_score = recall_score(y_test, y_pred, zero_division=0)
pre_score = precision_score(y_test, y_pred, zero_division=0)
fscore = f1_score(y_test, y_pred, zero_division=0)

print("-- Random Forest model --")
print("Accuracy_score :", acc_rf)
print("Recall_score   :", rec_score)
print("Precision_score:", pre_score)
print("F1_score       :", fscore)


# MODEL 4: GRADIENT BOOSTING
model3 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)

acc_gb = accuracy_score(y_test, y_pred)
rec_score = recall_score(y_test, y_pred, zero_division=0)
pre_score = precision_score(y_test, y_pred, zero_division=0)
fscore = f1_score(y_test, y_pred, zero_division=0)

print("-- Gradient Boosting model --")
print("Accuracy_score :", acc_gb)
print("Recall_score   :", rec_score)
print("Precision_score:", pre_score)
print("F1_score       :", fscore)

# ============================================
# MODEL COMPARISON 
# ============================================

model_accuracies = {
    "Logistic Regression": acc_lr,
    "Decision Tree": acc_dt,
    "Random Forest": acc_rf,
    "Gradient Boosting": acc_gb
}

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model_name]

print("\n--------------------------------------------")
print(" MODEL COMPARISON SUMMARY")
print("LR accuracy:", acc_lr)
print("DT accuracy:", acc_dt)
print("RF accuracy:", acc_rf)
print("GB accuracy:", acc_gb)
print("-----------------------------------------------")
print("best model is:", best_model_name)

plt.figure(figsize=(8, 5))
bars = plt.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'green', 'orange', 'red'])
plt.ylabel("Accuracy Score")
plt.title("Model Accuracies")
plt.ylim(0, 1.1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, str(round(yval, 2)), ha="center", va="bottom")

plt.show()

# 2. Simple Feature Importance Chart (Vertical Bar Graph)
plt.figure()
pd.Series(model2.feature_importances_, index=feature_columns).nlargest(5).plot(kind="barh")
plt.title("Top Reasons for Delay")
plt.xlabel("Importance")
plt.show()
# SELECT MODEL FOR PREDICTION
# ============================================

print("\n--------------------------------------------")
print(" SELECT MODEL FOR PREDICTION")
print("--------------------------------------------")

while True:
    print("\nAvailable Models:")
    print("1 = Logistic Regression")
    print("2 = Decision Tree")
    print("3 = Random Forest")
    print("4 = Gradient Boosting")

    choice = input("\nEnter model number (1-4): ")

    if choice == "1":
        selected_model_name="Logistic Regression"
        selected_model = model
        break
    elif choice == "2":
        selected_model_name="Decision Tree"
        selected_model = model1
        break
    elif choice == "3":
        selected_model_name= "Random Forest"
        selected_model= model2
        break
    elif choice == "4":
        selected_model_name= "Gradient Boosting"
        selected_model = model3
        break
    else:
        print("Invalid choice! Please pick 1, 2, 3, or 4.")

print("\nSelected Model:", selected_model_name)

# ENTER DELIVERY INFORMATION
# ============================================
print("\n--------------------------------------------")
print(" ENTER DELIVERY INFORMATION")
print("--------------------------------------------")

# 1. Traffic Level
while True:
    print("\nAvailable Traffic Levels:")
    print(list(encoders["Traffic_Level"].classes_))
    traffic = input("Enter Traffic Level: ")
    
    if traffic in encoders["Traffic_Level"].classes_:
        traffic_encoded = encoders["Traffic_Level"].transform([traffic])[0]
        break
    print("Invalid Traffic Level. Please try again.")

# 2. Weather Condition
while True:
    print("\nAvailable Weather Conditions:")
    print(list(encoders["weather_description"].classes_))
    weather = input("Enter Weather Condition: ")
    
    if weather in encoders["weather_description"].classes_:
        weather_encoded = encoders["weather_description"].transform([weather])[0]
        break
    print("Invalid Weather Condition. Please try again.")

# 3. Order Type
while True:
    print("\nAvailable Order Types:")
    print(list(encoders["Type_of_order"].classes_))
    order_type = input("Enter Type of Order: ")
    
    if order_type in encoders["Type_of_order"].classes_:
        order_encoded = encoders["Type_of_order"].transform([order_type])[0]
        break
    print("Invalid Order Type. Please try again.")

# 4. Vehicle Type
while True:
    print("\nAvailable Vehicle Types:")
    print(list(encoders["Type_of_vehicle"].classes_))
    vehicle = input("Enter Type of Vehicle: ")
    
    if vehicle in encoders["Type_of_vehicle"].classes_:
        vehicle_encoded = encoders["Type_of_vehicle"].transform([vehicle])[0]
        break
    print("Invalid Vehicle Type. Please try again.")

# Numerical inputs
age = float(input("Enter Delivery Person Age: "))
rating = float(input("Enter Delivery Person Rating: "))
restaurant_latitude = float(input("Enter Restaurant Latitude: "))
restaurant_longitude = float(input("Enter Restaurant Longitude: "))
delivery_latitude = float(input("Enter Delivery Location Latitude: "))
delivery_longitude = float(input("Enter Delivery Location Longitude: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
precipitation = float(input("Enter Precipitation: "))
distance = float(input("Enter Distance (km): "))

# CREATE USER INPUT DATAFRAME
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

# SCALE USER INPUT
user_input_scaled = scaler.transform(user_input)

# FINAL PREDICTION
final_result = selected_model.predict(user_input_scaled)

probability = selected_model.predict_proba(user_input_scaled)

# DISPLAY RESULT
print("\n==========================================")
print(" FINAL PREDICTION")
print("==========================================")

print("Model Used: " + selected_model_name)

predicted_class = final_result[0]

if predicted_class == 1:
    print("Prediction: DELAY")
else:
    print("Prediction: NOT DELAY")

print("Probability: " + str(round(probability[0][predicted_class] * 100, 2)) + "%")