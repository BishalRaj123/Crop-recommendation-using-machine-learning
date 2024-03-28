import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("data/Crop_recommendation.csv")

print(df.head())
# Select independent and dependent variable
df = df.to_numpy()
X = df[1:, 3:-1]
y = df[1:, -1]


# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the model
classifier = RandomForestClassifier(random_state=42)

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))

