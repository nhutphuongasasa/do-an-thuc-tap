import joblib
import os
import numpy as np

model_dir = "/home/phuong/Documents/do an thuc tpa tot nghiep/TrafficGuard/models"
model_path = os.path.join(model_dir, "rf_final_model.pkl")
scaler_path = os.path.join(model_dir, "scaler_final.pkl")
encoder_path = os.path.join(model_dir, "label_encoder.pkl")
feature_list_path = os.path.join(model_dir, "feature_list.pkl")

print("Checking files...")
for name, path in [("Model", model_path), ("Scaler", scaler_path), ("Encoder", encoder_path), ("Feature List", feature_list_path)]:
    print(f"{name} exists: {os.path.exists(path)}")

if os.path.exists(model_path):
    print("Loading model...")
    model = joblib.load(model_path)
    print("Model type:", type(model))
    if hasattr(model, "classes_"):
        print("Model classes:", model.classes_)

if os.path.exists(scaler_path):
    print("Loading scaler...")
    scaler = joblib.load(scaler_path)
    print("Scaler type:", type(scaler))

if os.path.exists(encoder_path):
    print("Loading encoder...")
    encoder = joblib.load(encoder_path)
    print("Encoder type:", type(encoder))
    if hasattr(encoder, "classes_"):
        print("Encoder classes:", encoder.classes_)

if os.path.exists(feature_list_path):
    print("Loading feature list...")
    feature_list = joblib.load(feature_list_path)
    print("Feature list:", feature_list)
