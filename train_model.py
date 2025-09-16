import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import pickle

# --- Load data ---
df = pd.read_csv("beer-servings.csv")

# --- Features and target ---
X = df[["beer_servings", "wine_servings", "spirit_servings"]]
y = df["total_litres_of_pure_alcohol"]

# --- Drop rows where y is NaN ---
mask = y.notna()
X = X[mask]
y = y[mask]

# --- Build pipeline: impute missing values + model ---
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # fills NaNs in X with column mean
    ("model", LinearRegression())
])

# --- Train model ---
pipeline.fit(X, y)

# --- Save the trained pipeline ---
with open("beer_servings_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved to beer_servings_model.pkl")
