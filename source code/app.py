import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from dotenv import load_dotenv
from googletrans import Translator

load_dotenv()
app = Flask(__name__)

# Load the trained models and scaler
model_path = "svm_models.pkl"
scaler_path = "scaler.pkl"

with open(model_path, 'rb') as f:
    models = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the GenAI API key
api_key = os.getenv('MY_GENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set MY_GENAI_API_KEY in .env.")

# Configure GenAI API
genai.configure(api_key=api_key)

# Initialize the translator
translator = Translator()

# Define the predictors (features)
predictors = ['hb_gdL', 'crp_mgL', 'ferritin_ugL', 'stfr_mgL', 'bodyiron',
              'rbp_umolL', 'zn_umol', 'se_umol', 'b12_pmolL', 'folate_nmolL',
              'totalvitd_nmolL', 'zn_corrected_timeandmeal', 'zn_adj_aftercorrection', 'se_adj']

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html exists with the correct form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the JSON request
        input_data = request.get_json()
        
        # Extract feature values from the input JSON
        input_values = np.array([float(input_data[feature]) for feature in predictors]).reshape(1, -1)
        
        # Scale the input
        scaled_input = scaler.transform(input_values)
        
        # Predict deficiencies
        deficiencies = []
        for vitamin, model in models.items():
            pred = model.predict(scaled_input)
            if pred[0] == 1:
                deficiencies.append(vitamin)

        # Extract food preferences and selected language
        food_preferences = input_data.get('food_preferences', 'balanced diet')
        selected_language = input_data.get('language', 'en')

        # Translate food preferences to English (if not already in English)
        if selected_language != 'en':
            translated_preferences = translator.translate(food_preferences, src=selected_language, dest='en').text
        else:
            translated_preferences = food_preferences

        # Generate personalized suggestions using GenAI
        if deficiencies:
            prompt = f"""
                        For each of the following vitamin deficiencies: {', '.join(deficiencies)}, generate at least 5 food items names only that can help address the deficiency. 
                        For each food item, provide a short explanation of how it helps with the deficiency. 
                        Please format your response as follows:

                        For {deficiencies[0]}:
                        - Food Item 1
                        - Food Item 2
                        - Food Item 3
                        - Food Item 4 
                        - Food Item 5 

                        For {deficiencies[1]}:
                        - Food Item 1
                        - Food Item 2
                        - Food Item 3
                        - Food Item 4
                        - Food Item 5

                        (Repeat for all deficiencies in the list)"""\
                     f"Personalized food recommendations should consider the following preference: {translated_preferences}."
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            ai_generated_text = response.text

            # Translate the GenAI response back to the selected language
            if selected_language != 'en':
                ai_generated_text = translator.translate(ai_generated_text, src='en', dest=selected_language).text
        else:
            ai_generated_text = "No deficiencies detected. Maintain a balanced diet."

        # Return results as JSON
        return jsonify({
            'status': 'success',
            'ai_generated_text': ai_generated_text,
            'deficiencies': deficiencies if deficiencies else 'None'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
