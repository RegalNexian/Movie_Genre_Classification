import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
dataset_folder = os.path.join('D:\\Projects\\Movie_Genre_Classification\\Dataset\\Genre Classification Dataset')

# Create a try-except block to handle file not found errors
try:

# Load the dataset
# Assuming there's a CSV file with movie data in the dataset folder
# Adjust the filename as needed based on your actual dataset
data_file = os.path.join(dataset_folder, 'train_data.txt')
df = pd.read_csv(data_file, sep=':::', engine='python', header=None, names=['id', 'title', 'genre', 'plot'])

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())
print("\nSample data:")
print(df.head())

# Preprocess the data
# Assuming the dataset has columns for movie descriptions/plots and genres
# Adjust column names based on your actual dataset
X = df['plot'].fillna('')  # Replace with your text column name
y = df['genre'].fillna('')  # Replace with your genre column name

# Check if genres are multi-label (comma-separated)
if y.str.contains(',').any():
    print("Detected multi-label genres. Processing with MultiLabelBinarizer...")
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    # Split genres by comma and strip whitespace
    y_multilabel = y.str.split(',').apply(lambda x: [genre.strip() for genre in x])
    y = mlb.fit_transform(y_multilabel)
    # Use OneVsRestClassifier for multi-label classification
    rf_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    lr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    print(f"Processed {len(mlb.classes_)} unique genre labels")
else:
    print("Using single-label genre classification")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction - convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Random Forest model
print("\nTraining Random Forest model...")
from tqdm import tqdm
try:
    # Use tqdm for better progress tracking if available
    with tqdm(total=100, desc="RF Progress") as pbar:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)
        rf_model.fit(X_train_tfidf, y_train)
        pbar.update(100)
except ImportError:
    # Fallback if tqdm is not installed
    print("Training... ", end="")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
    rf_model.fit(X_train_tfidf, y_train)
    print("Done!")

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")
try:
    # Use tqdm for better progress tracking if available
    with tqdm(total=100, desc="LR Progress") as pbar:
        lr_model = LogisticRegression(max_iter=1000, random_state=42, verbose=0)
        lr_model.fit(X_train_tfidf, y_train)
        pbar.update(100)
except ImportError:
    # Fallback if tqdm is not installed
    print("Training... ", end="")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
    lr_model.fit(X_train_tfidf, y_train)
    print("Done!")

# Evaluate models
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return accuracy, y_pred

# Evaluate Random Forest
rf_accuracy, rf_preds = evaluate_model(rf_model, X_test_tfidf, y_test, "Random Forest")

# Evaluate Logistic Regression
lr_accuracy, lr_preds = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")

# Compare models
print("\nModel Comparison:")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Visualize results
plt.figure(figsize=(10, 6))
models = ['Random Forest', 'Logistic Regression']
accuracies = [rf_accuracy, lr_accuracy]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylim(0, 1.0)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.savefig(os.path.join('d:', 'Projects', 'Movie_Genre_Classification', 'model_comparison.png'))
plt.close()

# Save models
import pickle

# Define a consistent output directory
output_dir = os.path.join('d:', 'Projects', 'Movie_Genre_Classification')

# Use consistent path joining for all model files
with open(os.path.join(output_dir, 'random_forest_model.pkl'), 'wb') as f:
    pickle.dump(rf_model, f)
    
with open(os.path.join(output_dir, 'logistic_regression_model.pkl'), 'wb') as f:
    pickle.dump(lr_model, f)

with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)

print("\nModels saved successfully!")

# Function to predict genre for new movies
def predict_genre(plot, model, vectorizer):
    plot_tfidf = vectorizer.transform([plot])
    prediction = model.predict(plot_tfidf)
    return prediction[0]

# Example usage
print("\nExample prediction:")
sample_plot = "A group of astronauts travel through a wormhole in search of a new home for humanity."
print(f"Sample plot: {sample_plot}")
print(f"Predicted genre (Random Forest): {predict_genre(sample_plot, rf_model, tfidf)}")
print(f"Predicted genre (Logistic Regression): {predict_genre(sample_plot, lr_model, tfidf)}")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    print("Please ensure the dataset files exist in the specified location.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()