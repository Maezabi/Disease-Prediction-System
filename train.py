import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Disease Prediction System - Training ===")

# 1. Data Loading
df = pd.read_csv('dataset.csv')
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Data Cleaning
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# Clean column names (remove extra spaces)
df.columns = [col.strip() for col in df.columns]

# 3.Feature Processing (Define Target and Features)
target = 'Heart Disease'   # Change to another disease if you want

# Drop ALL other disease columns from features
disease_columns = ['Heart Disease', 'Diabetes', 'Stroke', 'Kidney Disease',
                   'Cancer', "Alzheimer's Disease", 'COPD', 'Liver Disease',
                   "Parkinson's Disease", 'Tuberculosis']  # Add more if needed

# Features = everything except disease targets
X = df.drop(columns=disease_columns)
y = df[target]

print(f"Features used: {X.columns.tolist()}")

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'label_encoder.pkl')

# Encode categorical features
feature_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le
joblib.dump(feature_encoders, 'feature_encoders.pkl')

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 5. Data Balancing (SMOTE)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# 7. Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train_bal)
joblib.dump(model, 'model.pkl')

# 8. Evaluation
y_pred = model.predict(X_test_scaled)
print("\n✅ Accuracy:", round(model.score(X_test_scaled, y_test), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Cross Validation
print("\n=== Cross Validation ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train_bal, cv=skf, scoring='f1_macro')
print("Stratified K-Fold Mean F1-score:", round(cv_scores.mean(), 4))

# 10. Model Saving
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print("\n✅ Training Completed! All files saved.")
