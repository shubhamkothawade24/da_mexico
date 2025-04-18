import pandas as pd
from glob import glob
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder

def wrangle(filepath):
    df = pd.read_csv(filepath)
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    df = df[df["surface_covered_in_m2"].between(low, high)]
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    drop_cols = [
        "lat-lon", "place_with_parent_names", "expenses", "floor",
        "operation", "property_type", "currency", "properati_url",
        "price", "price_aprox_local_currency", "price_per_m2", "price_usd_per_m2",
        "surface_total_in_m2", "rooms"
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df

def load_data():
    files = glob("mexico-city-real-estate-*.csv")
    frames = [wrangle(file) for file in files]
    return pd.concat(frames, ignore_index=True)

def train_model(df):
    X = df[["surface_covered_in_m2", "lat", "lon", "neighborhood"]]
    y = df["price_aprox_usd"]
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        SimpleImputer(),
        LinearRegression()
    )
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    return model

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
