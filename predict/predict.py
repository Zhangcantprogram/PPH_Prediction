import pandas as pd
import joblib
from pathlib import Path


def predict_and_save():
    """Load trained XGBoost model and add predictions to new dataset"""

    # =====================================
    # Configuration (Modify these paths)
    # =====================================
    MODEL_PATH = Path("../client/xgboost_best_fold9_auc0.8671.pkl")  # Trained model path
    INPUT_DATA_PATH = Path("../dataset/dataset.xlsx")  # Input data for prediction
    OUTPUT_DATA_PATH = Path("dataset_with_predictions.xlsx")  # Output file path
    THRESHOLD = 0.5  # Classification threshold

    # =====================================
    # Model and Data Loading
    # =====================================
    try:
        # Load trained model
        model = joblib.load(MODEL_PATH)
        print(f"✅ Successfully loaded model from {MODEL_PATH}")

        # Load data for prediction
        df = pd.read_excel(INPUT_DATA_PATH)
        print(f"📊 Loaded data with shape: {df.shape}")

    except Exception as e:
        print(f"❌ Error loading files: {str(e)}")
        return

    # =====================================
    # Data Preparation
    # =====================================
    # Ensure feature columns match training data
    # Remove label column if present
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]

    # =====================================
    # Prediction
    # =====================================
    try:
        # Get prediction probabilities
        proba = model.predict_proba(X)[:, 1]

        # Convert probabilities to class predictions
        predictions = (proba >= THRESHOLD).astype(int)
        print("🎯 Successfully generated predictions")

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return

    # =====================================
    # Create New Dataset with Predictions
    # =====================================
    # Create modified dataframe copy
    df_with_pred = df.copy()

    # Add predictions as new columns
    df_with_pred['prediction'] = predictions
    df_with_pred['prediction_probability'] = proba  # Optional probability column

    # =====================================
    # Save Results
    # =====================================
    try:
        df_with_pred.to_excel(OUTPUT_DATA_PATH, index=False)
        print(f"💾 Saved predictions to {OUTPUT_DATA_PATH}")
        print("✅ Operation completed successfully!")

    except Exception as e:
        print(f"❌ Failed to save results: {str(e)}")


if __name__ == "__main__":
    predict_and_save()