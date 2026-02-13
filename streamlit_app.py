import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import shap
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Smart Meal Planner", page_icon="🥗", layout="wide")

# Styling
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🥗 Smart Meal Planner")
st.markdown("### AI-Powered Nutrition & SHAP Explanations (100% Python)")

# --- Sidebar Inputs ---
st.sidebar.header("Enter Your Details")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", ["male", "female"])
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=180)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=75)
activity_level = st.sidebar.selectbox("Activity Level", ["sedentary", "light", "moderate", "active"])
goal = st.sidebar.selectbox("Goal", ["maintain", "loss", "hard_loss", "gain", "hard_gain"])
meals_per_day = st.sidebar.selectbox("Meals Per Day", [3, 2, 4])
diet_pref = st.sidebar.selectbox("Diet Preference", ["non_veg", "veg", "vegan"])
allergies_input = st.sidebar.text_input("Allergies (comma separated)", "")

# --- Logic (Copied from app.py) ---
def calculate_bmr(weight, height, age, gender):
    if gender == 'male':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

def calculate_tdee(bmr, activity_level):
    multipliers = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'active': 1.725}
    return bmr * multipliers.get(activity_level, 1.2)

def calculate_target_calories(tdee, goal):
    goal_multipliers = {'hard_gain': 1.3, 'gain': 1.2, 'maintain': 1.0, 'loss': 0.8, 'hard_loss': 0.7}
    return tdee * goal_multipliers.get(goal, 1.0)

# --- Load Data with Caching ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("All_Diets.csv")
        df.columns = df.columns.str.strip().str.lower()
        for col in ['calories', 'protein', 'carbohydrate', 'total_fat']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# --- Main App Logic ---
if st.button("Generate Meal Plan", type="primary"):
    if df.empty:
        st.error("Data not loaded.")
        st.stop()

    with st.spinner("Analyzing nutritional data..."):
        # 1. Calc Targets
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity_level)
        daily_cals = calculate_target_calories(tdee, goal)
        
        daily_protein = (daily_cals * 0.25) / 4
        daily_carbs = (daily_cals * 0.50) / 4
        daily_fat = (daily_cals * 0.25) / 9
        
        target_calories = daily_cals / meals_per_day
        target_protein = daily_protein / meals_per_day
        target_carbs = daily_carbs / meals_per_day
        target_fat = daily_fat / meals_per_day

        target_vector = np.array([target_calories, target_protein, target_carbs, target_fat])

        # Display Targets
        st.subheader("Your Meal Targets")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calories", f"{target_calories:.0f} kcal")
        col2.metric("Protein", f"{target_protein:.1f} g")
        col3.metric("Carbs", f"{target_carbs:.1f} g")
        col4.metric("Fat", f"{target_fat:.1f} g")

        # 2. Filter Data
        filtered_df = df.copy()
        
        if diet_pref == 'vegan':
            filtered_df = filtered_df[filtered_df['name'].str.contains('vegan', case=False, na=False)]
        elif diet_pref == 'veg':
            meat_keywords = ['chicken', 'beef', 'pork', 'turkey', 'fish', 'salmon', 'shrimp', 'crab', 'ham', 'bacon', 'steak', 'sausage', 'meatball', 'tuna', 'cod', 'lamb']
            filtered_df = filtered_df[~filtered_df['name'].str.contains('|'.join(meat_keywords), case=False, na=False)]
        
        if allergies_input:
            allergies_list = [a.strip() for a in allergies_input.split(',')]
            for allergy in allergies_list:
                if allergy:
                    filtered_df = filtered_df[~filtered_df['name'].str.contains(allergy, case=False, na=False)]
        
        if filtered_df.empty:
            st.warning("No specific meals found for these filters. Showing best matches from full database.")
            filtered_df = df.copy()

        # 3. KNN Search
        feature_cols = ['calories', 'protein', 'carbohydrate', 'total_fat']
        X = filtered_df[feature_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        target_vector_reshaped = target_vector.reshape(1, -1)
        target_scaled = scaler.transform(target_vector_reshaped)
        
        n_neighbors = min(meals_per_day * 5, len(filtered_df))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X_scaled)
        
        distances, indices = knn.kneighbors(target_scaled)
        # flatten indices array
        all_indices = indices[0]
        
        # Display Meals Breakdown
        # User requested no specific names like Breakfast/Lunch/Dinner
        current_labels = [f"Meal {i+1}" for i in range(meals_per_day)]

        st.markdown("---")
        for meal_idx in range(meals_per_day):
            st.subheader(f"🥣 {current_labels[meal_idx]}")
            
            # Slice 5 distinct meals for this slot
            start = meal_idx * 5
            end = start + 5
            slot_indices = all_indices[start:end]
            slot_meals = filtered_df.iloc[slot_indices]
            
            for _, row in slot_meals.iterrows():
                with st.expander(f"**{row['name']}** ({row['calories']} kcal)", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.write(f"**P:** {row['protein']}g")
                    c2.write(f"**C:** {row['carbohydrate']}g")
                    c3.write(f"**F:** {row['total_fat']}g")
                    c4.write(f"*{row.get('cuisine_type', 'General')}*")

        # 4. SHAP Explanation (Global for the "Best Match")
        st.markdown("---")
        st.subheader("💡 AI Explanation (Based on Best Overall Match)")
        st.write("How nutritional features influenced the top recommendation's score:")
        
        best_match_features_scaled = X_scaled[all_indices[0]].reshape(1, -1)
        best_match_features_df = pd.DataFrame(best_match_features_scaled, columns=feature_cols)
        X_train_df = pd.DataFrame(X_scaled, columns=feature_cols) 
        
        # Target variable y = negative absolute error from target
        y = -np.abs(X_scaled - target_scaled).mean(axis=1)
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        explainer = shap.LinearExplainer(model, X_scaled)
        shap_values = explainer.shap_values(best_match_features_df)
        
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, best_match_features_df, plot_type="bar", show=False)
        st.pyplot(fig)
