import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import streamlit as st
import os

# ========== TRAIN THE MODEL IF NOT ALREADY SAVED ==========
MODEL_FILE = "beer_servings_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.write("🔄 Training model for the first time...")
    # Load your data
    df = pd.read_csv("beer-servings.csv")

    # Features & target
    X = df[["beer_servings", "wine_servings", "spirit_servings"]]
    y = df["total_litres_of_pure_alcohol"]

    # Drop rows with missing y
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Pipeline: impute missing X + model
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", LinearRegression())
    ])

    pipeline.fit(X, y)

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipeline, f)

    st.write("✅ Model trained and saved to beer_servings_model.pkl")

# ========== LOAD TRAINED MODEL ==========
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# ========== STREAMLIT UI ==========
st.title("🍺 Total Alcohol Servings Prediction App")
st.image(
    "https://i.pinimg.com/1200x/f9/39/24/f93924205fa02e3af33607d2057e3417.jpg",
    caption="Welcome to the Beer Servings App",
    use_container_width=True
)

st.write("### Enter details to predict the total alcohol servings (Beer + Wine + Spirit)")

# User inputs
wine_servings = st.number_input("Wine Servings (litres)", min_value=0, max_value=500, value=50)
spirit_servings = st.number_input("Spirit Servings (litres)", min_value=0, max_value=500, value=100)
beer_servings = st.number_input("Beer Servings (litres)", min_value=0, max_value=500, value=200)

# Prepare input for prediction — only the columns the model was trained on
input_data = pd.DataFrame([{
    "beer_servings": beer_servings,
    "wine_servings": wine_servings,
    "spirit_servings": spirit_servings
}])

# Prediction button
if st.button("Predict Total Alcohol Consumption"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Total Litres of Pure Alcohol: {prediction[0]:.2f}")