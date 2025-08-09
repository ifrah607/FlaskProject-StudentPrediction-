from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
# read binary
model = pickle.load(open("model/rf_model.pkl", "rb"))

app = Flask(__name__)

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting form inputs from list
        features = [
            request.form.get("Hours_Studied"),
            request.form.get("Attendance"),
            request.form.get("Parental_Involvement"),
            request.form.get("Access_to_Resources"),
            request.form.get("Extracurricular_Activities"),
            request.form.get("Sleep_Hours"),
            request.form.get("Previous_Scores"),
            request.form.get("Motivation_Level"),
            request.form.get("Internet_Access"),
            request.form.get("Tutoring_Sessions"),
            request.form.get("Family_Income"),
            request.form.get("Teacher_Quality"),
            request.form.get("School_Type"),
            request.form.get("Peer_Influence"),
            request.form.get("Physical_Activity"),
            request.form.get("Learning_Disabilities"),
            request.form.get("Parental_Education_Level"),
            request.form.get("Distance_from_Home"),
            request.form.get("Gender")
        ]

        # Handle missing values
        if None in features or "" in features:
            return render_template('index.html', prediction_text='Error: Missing or empty input fields')

        # Convert features to numeric values where applicable #label encoding
        processed_features = preprocess_input(features) #function

        # Make prediction
        prediction = model.predict([processed_features])
        return render_template('index.html', prediction_text=f'Predicted Exam Score: {prediction[0]:.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in prediction: {str(e)}')

# Preprocessing function
# This example uses simple mapping for categorical data
# You should customize it according to your dataset specifics
def preprocess_input(features):
    # Example mappings for categorical features
    categorical_mapping = {
        "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
        "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
        "Extracurricular_Activities": {"No": 0, "Yes": 1},
        "Internet_Access": {"No": 0, "Yes": 1},
        "Learning_Disabilities": {"No": 0, "Yes": 1},
        "School_Type": {"Public": 0, "Private": 1},
        "Peer_Influence": {"Negative": 0, "Neutral": 1, "Positive": 2},
        "Gender": {"Male": 0, "Female": 1},
        "Distance_from_Home": {"Near": 0, "Moderate": 1, "Far": 2},
        "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
        "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
        "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
    }

    processed_features = []
    for i, value in enumerate(features):
        if value.replace('.', '', 1).isdigit():  # Handle numeric inputs "2"
            processed_value = float(value)
            processed_features.append(processed_value)
        else:
            # Use mapping if feature is categorical, else assign default value
            mapped_value = None
            for category, mapping in categorical_mapping.items():
                if value in mapping:
                    mapped_value = mapping[value]
                    break
            processed_value = mapped_value if mapped_value is not None else 0
            processed_features.append(processed_value)
        
        # Print original and processed values for debugging
        print(f"Feature {i}: Original: {value}, Processed: {processed_value}")

    return processed_features

if __name__ == "__main__":
    app.run(debug=True)