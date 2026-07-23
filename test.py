import joblib

model_text = joblib.load("model/text_best_model.pkl")

print(model_text.n_features_in_)
print(type(model_text))

obj_text = joblib.load("model/text_best_extractor.pkl")
print(type(obj_text))
print(type(obj_text).__name__)
print(obj_text)


model = joblib.load("model/best_model.pkl")

print(model.n_features_in_)
print(type(model))

obj = joblib.load("model/feature_info.pkl")
print(type(obj))
print(obj)

extractor = joblib.load("model/best_extractor_vectorizer.pkl")

print(type(extractor))
print(type(extractor).__name__)
print(extractor)