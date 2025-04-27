import pickle
import os
import sys

# Load the models and vectorizer
model_dir = os.path.join('d:\\Projects\\Movie_Genre_Classification')

with open(os.path.join(model_dir, 'random_forest_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)
    
with open(os.path.join(model_dir, 'logistic_regression_model.pkl'), 'rb') as f:
    lr_model = pickle.load(f)
    
with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

def predict_genre(plot, model_name='rf'):
    """
    Predict the genre of a movie based on its plot.
    
    Args:
        plot (str): The plot/description of the movie
        model_name (str): 'rf' for Random Forest, 'lr' for Logistic Regression
    
    Returns:
        str: Predicted genre
    """
    plot_tfidf = tfidf.transform([plot])
    
    if model_name.lower() == 'rf':
        prediction = rf_model.predict(plot_tfidf)
    elif model_name.lower() == 'lr':
        prediction = lr_model.predict(plot_tfidf)
    else:
        raise ValueError("Model name must be 'rf' or 'lr'")
    
    return prediction[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_genre.py \"Movie plot description\" [model_name]")
        sys.exit(1)
    
    plot = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'rf'
    
    print(f"Predicted genre: {predict_genre(plot, model_name)}")