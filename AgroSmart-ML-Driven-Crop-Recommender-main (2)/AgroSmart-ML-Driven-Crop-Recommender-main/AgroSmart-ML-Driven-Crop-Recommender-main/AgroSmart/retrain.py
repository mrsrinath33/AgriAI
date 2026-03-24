import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
import joblib
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models dir if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

print("=== Training Crop Recommender ===")
df_crop = pd.read_csv(os.path.join(DATA_DIR, 'Crop_recommendation.csv'))
x_crop = df_crop.drop('label', axis=1)
y_crop = df_crop['label']
x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(x_crop, y_crop, random_state=1, test_size=0.2)
model_crop = RandomForestClassifier()
model_crop.fit(x_train_c, y_train_c)
joblib.dump(model_crop, os.path.join(MODELS_DIR, 'crop_app.pkl'))
print("✓ Crop Recommender trained and saved.")

print("\n=== Training Fertilizer Recommender ===")
df_fert = pd.read_csv(os.path.join(DATA_DIR, 'fertilizer_recommendation.csv'))

# Label encoding
soil_type_label_encoder = LabelEncoder()
df_fert["Soil Type"] = soil_type_label_encoder.fit_transform(df_fert["Soil Type"])
crop_type_label_encoder = LabelEncoder()
df_fert["Crop Type"] = crop_type_label_encoder.fit_transform(df_fert["Crop Type"])
fertname_label_encoder = LabelEncoder()
df_fert["Fertilizer Name"] = fertname_label_encoder.fit_transform(df_fert["Fertilizer Name"])

X_fert = df_fert[df_fert.columns[:-1]]
y_fert = df_fert[df_fert.columns[-1]]

upsample = SMOTE()
X_fert_up, y_fert_up = upsample.fit_resample(X_fert, y_fert)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fert_up.values, y_fert_up, test_size=0.2, random_state=3)

model_fert = RandomForestClassifier()
model_fert.fit(X_train_f, y_train_f)
joblib.dump(model_fert, os.path.join(MODELS_DIR, 'fertilizer_app.pkl'))
print("✓ Fertilizer Recommender trained and saved.")

print("\n=== Training Yield Prediction ===")
df_yield = pd.read_csv(os.path.join(DATA_DIR, 'yield_df.csv'))
if 'Unnamed: 0' in df_yield.columns:
    df_yield.drop('Unnamed: 0', axis=1, inplace=True)
df_yield.drop_duplicates(inplace=True)

col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df_yield = df_yield[col]

X_y = df_yield.drop('hg/ha_yield', axis=1)
y_y = df_yield['hg/ha_yield']

X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_y, y_y, test_size=0.2, random_state=0, shuffle=True)

ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
scale = StandardScaler()
preprocesser_y = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),
        ('OneHotEncode', ohe, [4, 5])
    ],
    remainder='passthrough'
)

# Fit and transform
X_train_dummy = preprocesser_y.fit_transform(X_train_y)

dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train_y)

pickle.dump(dtr, open(os.path.join(MODELS_DIR, 'dtr.pkl'), "wb"))
pickle.dump(preprocesser_y, open(os.path.join(MODELS_DIR, 'preprocesser.pkl'), "wb"))
print("✓ Yield Prediction trained and saved.")

print("\nDone! All models have been successfully retrained.")
