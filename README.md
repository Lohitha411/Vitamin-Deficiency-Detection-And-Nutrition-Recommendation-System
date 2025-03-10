# Vitamin-Deficiency-Detection-And-Nutrition-Recommendation-System
**Project Overview:**
This project is a **Vitamin Deficiency Prediction and Food Recommendation System** that helps users identify potential vitamin deficiencies based on their biomarker values and provides personalized food recommendations to address those deficiencies. The system supports **AI integration** for enhanced prediction accuracy and **multi-lingual input and output**, allowing users to interact in their preferred language (e.g., Telugu, Hindi, Tamil, English).

**Features**
Deficiency Prediction: Predicts vitamin deficiencies (e.g., Vitamin A, Vitamin D) based on user-provided biomarker values.
Food Recommendations: Provides a list of food items that can help address the identified deficiencies.
Multi-Lingual Support: Supports multiple languages (Telugu, Hindi, Tamil, English) for input and output.
User-Friendly Interface: Dynamic virtual keyboard for typing in selected languages.
Generative AI Integration: Uses Google's Generative AI API to generate personalized food recommendations.

**Technologies Used**
Frontend: HTML, CSS, JavaScript, SimpleKeyboard library for virtual keyboard
Backend: Flask framework, Scikit-learn for machine learning models, Google Generative AI API for food recommendations
Translation: googletrans library for multi-lingual support
Other Tools: Google Cloud Console for API key management, Pickle for model serialization

**How It Works**
User Input: Users enter their biomarker values (e.g., hemoglobin, ferritin) and food preferences and their preferred language (e.g., Telugu, Hindi, Tamil, English).
Deficiency Prediction: The backend uses pre-trained machine learning models to predict vitamin deficiencies based on the input biomarker values.
Translation: The user’s food preferences are translated from the selected language to English.
Generative AI: The translated input is sent to the Generative AI API, which generates food recommendations.
Translation Back: The AI’s response is translated back to the user’s selected language.
Output: The final recommendations are displayed to the user in their preferred language.
