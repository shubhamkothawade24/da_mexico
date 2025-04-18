import pandas as pd
from glob import glob
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# -------------------------------
# 1. Data Wrangling Function
# -------------------------------
def wrangle(filepath):
    df = pd.read_csv(filepath)

    # Filter for relevant data
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Remove outliers in surface area
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    df = df[df["surface_covered_in_m2"].between(low, high)]

    # Split lat-lon
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]

    # Drop irrelevant/multicollinear columns
    drop_cols = [
        "lat-lon", "place_with_parent_names", "expenses", "floor",
        "operation", "property_type", "currency", "properati_url",
        "price", "price_aprox_local_currency", "price_per_m2", "price_usd_per_m2",
        "surface_total_in_m2", "rooms"
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df

# -------------------------------
# 2. Load and Combine Data
# -------------------------------
def load_data():
    files = glob("mexico-city-real-estate-*.csv")
    frames = [wrangle(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    return df

# -------------------------------
# 3. Train Model
# -------------------------------
def train_model(df):
    target = "price_aprox_usd"
    features = ["surface_covered_in_m2", "lat", "lon", "neighborhood"]
    X = df[features]
    y = df[target]

    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        SimpleImputer(),
        LinearRegression()
    )
    model.fit(X, y)
    
    joblib.dump(model, "model.pkl")
    return model

# -------------------------------
# 4. Predict Function
# -------------------------------
def make_prediction(area, lat, lon, neighborhood):
    model = joblib.load("model.pkl")
    input_df = pd.DataFrame([{
        "surface_covered_in_m2": area,
        "lat": lat,
        "lon": lon,
        "neighborhood": neighborhood
    }])
    pred = model.predict(input_df)[0]
    return round(pred, 2)

# -------------------------------
# 5. Evaluate Model
# -------------------------------
def evaluate_model(model, df):
    X = df[["surface_covered_in_m2", "lat", "lon", "neighborhood"]]
    y = df["price_aprox_usd"]
    y_pred = model.predict(X)
    return mean_absolute_error(y, y_pred)

# -------------------------------
# 6. Main Flow
# -------------------------------
if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    mae = evaluate_model(model, df)
    print(f"Model trained successfully! MAE: ${mae:,.2f}")
    
    # Test prediction
    example = make_prediction(80, 19.42, -99.14, "Palermo")
    print(f"Predicted Price for sample: ${example:,.2f}")
