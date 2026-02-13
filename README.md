# 🥗 Smart Meal Planner

## Project Description and Purpose

The **Smart Meal Planner** is an AI-powered nutrition assistant designed to help users achieve their dietary goals—whether it's weight loss, muscle gain, or maintenance. 

Built entirely in Python, this application takes your personal metrics (age, gender, height, weight, activity level) and dietary preferences (vegetarian, vegan, allergies) to calculate your precise daily caloric and macronutrient needs (TDEE). It then uses a **K-Nearest Neighbors (KNN)** machine learning algorithm to recommend meals from a diverse dataset that best match your nutritional targets.

A unique feature of this project is the integration of **SHAP (SHapley Additive exPlanations)**, providing transparent, AI-driven explanations for *why* specific meals were recommended based on their nutritional profile.

## Key Features
- **Personalized Nutrition Calculation**: Calculates BMR and TDEE based on individual user data.
- **AI-Driven Recommendations**: Uses KNN to find meals closest to your optimal macro-nutrient distribution.
- **Dietary Customization**: Supports Vegan, Vegetarian, and Non-Veg diets with allergy filtering.
- **Explainable AI**: Visualizes feature importance using SHAP values to explain recommendations.
- **Interactive UI**: Clean, responsive interface built with Streamlit.

## Tech Stack

This project is built using a robust Python-based data science stack:

- **Language**: Python 3.10+
- **Frontend Framework**: [Streamlit](https://streamlit.io/) (for rapid, interactive web apps)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (for KNN algorithm and preprocessing)
- **Model Explainability**: [SHAP](https://shap.readthedocs.io/) (for interpretability)
- **Visualization**: [Matplotlib](https://matplotlib.org/) using Streamlit's pyplot integration.

## Setup and Installation Instructions

Follow these steps to get the application running on your local machine.

### Prerequisites
- Python 3.8 or higher installed.

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Smart Meal Planer"
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
You can run the application using the provided batch script or directly via the command line.

**Option A: Using the Batch Script (Windows)**
Double-click `run_app.bat` or run it from the terminal:
```powershell
.\run_app.bat
```

**Option B: Using Command Line**
```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## Screenshots

> *Add a screenshot of your running application here to mock the UI.*

![Smart Meal Planner UI](mock_screenshot.png)

## How It Works
1. **Enter Details**: Input your age, weight, height, and activity level in the sidebar.
2. **Set Goals**: Choose your goal (e.g., "Weight Loss") and dietary preferences.
3. **Generate Plan**: Click "Generate Meal Plan".
4. **View Results**: The app displays your daily targets and suggested meal options for breakfast, lunch, dinner, etc.
5. **Analyze**: Scroll down to see the SHAP plot explaining which nutritional factors influenced the top recommendation.

## License
This project is open-source and available under the MIT License.
