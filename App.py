import streamlit as st
import pandas as pd
import pickle  # or joblib if you saved the model that way


# Load trained modelgit 
with open(r"model\beer_servings_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🍺 Total Alcohol Servings Prediction App")
st.image(
    "https://i.pinimg.com/1200x/f9/39/24/f93924205fa02e3af33607d2057e3417.jpg",
    caption="Welcome to the Beer Servings App",
    use_container_width=True
)

st.write("### Enter details to predict the total alcohol servings (Beer + Wine + Spirit)")

# --- User Inputs ---
wine_servings = st.number_input("Wine Servings(litres)", min_value=0, max_value=500, value=50)
spirit_servings = st.number_input("Spirit Servings(litres)", min_value=0, max_value=500, value=100)
beer_servings = st.number_input("Beer Servings(litres)", min_value=0, max_value=500, value=200)

# --- Prepare input for prediction ---
# Add placeholders for columns required by the model
input_data = pd.DataFrame([{
    "Unnamed: 0": 0,  # placeholder
    "country": "Unknown",  # placeholder
    "total_litres_of_pure_alcohol": 0,  # placeholder
    "continent": "Unknown",  # placeholder
    "beer_servings": beer_servings,
    "wine_servings": wine_servings,
    "spirit_servings": spirit_servings
}])

# --- Prediction ---
if st.button("Predict Total Alcohol Consumption"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Estimated Total Litres of Pure Alcohol: {prediction[0]:.2f}")
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
