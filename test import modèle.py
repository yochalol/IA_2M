import pickle

with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))