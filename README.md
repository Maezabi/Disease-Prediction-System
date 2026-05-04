# Disease Prediction System (ML + GUI

## Problem Statement
Build a complete machine learning system that predicts Heart Diseases based on patient symptoms and deploy it as a Streamlit web application.

## Dataset Description
The Healthcare Disease Prediction Dataset provides 1,000 structured patient records with a combination of personal, lifestyle, and medical risk factors. It is built to support disease prediction and health analytics by including features like age, gender, blood pressure, cholesterol, glucose, BMI, smoking, alcohol consumption, exercise, and family history, along with multiple disease indicators. This dataset is ideal for building predictive models, comparing classification algorithms, and exploring healthcare related machine learning projects.
 
Contains 1000+ patient records with risk factors (symptoms) and disease indicators (Heart Disease, Diabetes, Stroke).  
We predict **Heart Disease** using all risk factors.

## How to Run Project
1. `pip install -r requirements.txt`
2. `python train.py` (generates models)
3. `streamlit run app.py`

## Model Used
- **RandomForestClassifier** (required)
- SMOTE for class balancing
- StandardScaler
- Stratified train-test split & Stratified K-Fold CV

## Results Summary
- Accuracy: 0.615
- Mean F1-score (CV): -

## Screenshots of GUI

<img width="1474" height="687" alt="Screenshot_4-5-2026_124511_localhost" src="https://github.com/user-attachments/assets/25bd621a-bd0b-4daf-87b0-945c48310ac4" />

<img width="1474" height="687" alt="Screenshot_4-5-2026_123719_localhost" src="https://github.com/user-attachments/assets/e4786270-cc9f-4023-98c6-b209ce9d6fd9" />

<img width="1474" height="687" alt="Screenshot_4-5-2026_124251_localhost" src="https://github.com/user-attachments/assets/02f40fed-4461-4c00-98ff-c48cbfb94f0f" />

<img width="1474" height="687" alt="Screenshot_4-5-2026_12446_localhost" src="https://github.com/user-attachments/assets/80b37281-0100-4e2e-904f-b699a78cbb6e" />

<img width="1474" height="687" alt="Screenshot_4-5-2026_124134_localhost" src="https://github.com/user-attachments/assets/d6bd0dac-9277-4816-9ca3-6183693cd851" />




## Bonus Implemented
- Confusion matrix heatmap
- Feature importance visualization
