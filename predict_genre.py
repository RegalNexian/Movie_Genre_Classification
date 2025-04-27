import pickle
import os
import sys

# Load the models and vectorizer
model_dir = os.path.join('d:\\Projects\\Movie_Genre_Classification')

try:
    with open(os.path.join(model_dir, 'random_forest_model.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
        
    with open(os.path.join(model_dir, 'logistic_regression_model.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
        
    with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
        
except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}")
    print("Please ensure you have trained the models first by running movie_genre_classifier.py")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")
    sys.exit(1)

def predict_genre(plot, model_name='rf'):
    """
    Predict the genre of a movie based on its plot.
    
    Args:
        plot (str): The plot/description of the movie
        model_name (str): 'rf' for Random Forest, 'lr' for Logistic Regression
    
    Returns:
        str or list: Predicted genre(s)
    """
    try:
        plot_tfidf = tfidf.transform([plot])
        
        if model_name.lower() == 'rf':
            prediction = rf_model.predict(plot_tfidf)
        elif model_name.lower() == 'lr':
            prediction = lr_model.predict(plot_tfidf)
        else:
            raise ValueError("Model name must be 'rf' or 'lr'")
        
        # Check if this is a multi-label prediction (array of 0s and 1s)
        if isinstance(prediction[0], (list, np.ndarray)) and len(prediction[0]) > 1:
            # For multi-label, we need to convert back to genre names
            # This assumes we have access to the MultiLabelBinarizer's classes_
            # If not, we'll just return the binary array
            try:
                # If mlb is available in the model file
                if hasattr(rf_model, 'classes_'):
                    # For single-label multi-class
                    return prediction[0]
                elif hasattr(rf_model, 'multilabel_'):
                    # For multi-label classification with OneVsRestClassifier
                    return prediction[0]
                else:
                    return prediction[0]
            except Exception:
                return prediction[0]
        else:
            # For single-label prediction
            return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error: Could not predict genre"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_genre.py \"Movie plot description\" [model_name]")
        sys.exit(1)
    
    plot = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'rf'
    
    print(f"Predicted genre: {predict_genre(plot, model_name)}")