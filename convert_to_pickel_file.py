import pickle

# Serialize the entire model object to a .pkl file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


# Deserialize the model object from the .pkl file
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
