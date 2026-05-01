import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from preprocessing import load_data, preprocess_data

# Load data
df = load_data("../data/student_data.csv")

# Preprocess
X, y = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("../models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
