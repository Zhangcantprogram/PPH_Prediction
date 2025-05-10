import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, recall_score, f1_score, \
    precision_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import joblib
import argparse

# Global configurations
RANDOM_STATE = 55
BORDER = "=" * 70


# --------------------------------------
# Data preparation
# --------------------------------------
def load_data():
    """Load and preprocess dataset"""
    print(BORDER)
    # Load dataset - modify path as needed
    df = pd.read_excel('../dataset/dataset.xlsx')

    # Extract features and target
    y = df['label']
    X = df.drop(columns=['label'])

    print(f"Dataset loaded, shape: {df.shape}")
    print(f"Number of features: {X.shape[1]}, Class balance: {sum(y) / len(y):.2%}")
    return X, y


# --------------------------------------
# Model configuration with command-line arguments
# --------------------------------------
def get_model_params(args):
    """Get model parameters with command-line arguments"""
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'colsample_bytree': args.colsample_bytree,
        'subsample': args.subsample,
        'reg_lambda': args.reg_lambda,
        'reg_alpha': args.reg_alpha,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'objective': 'binary:logistic',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }
    return params


# --------------------------------------
# Metric calculations
# --------------------------------------
def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate and return evaluation metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'recall': recall_score(y_true, y_pred, average="macro"),
        'precision': precision_score(y_true, y_pred, average="macro"),
        'f1': f1_score(y_true, y_pred, average="macro"),
    }


# --------------------------------------
# Main execution
# --------------------------------------
def main(args):
    X, y = load_data()

    # Initialize configurations
    model_params = get_model_params(args)
    smote = SMOTE(sampling_strategy=0.5073671668721272, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # Track best model
    best_metrics = {
        'fold': 0,
        'auc': 0.0,
        'model': None,
        'val_report': ''
    }

    # Store metrics
    train_metrics, val_metrics = [], []
    all_y_true, all_y_pred = [], []

    print(BORDER)
    print("Starting 10-fold cross-validation training...")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        # Data splitting
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE oversampling
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # Model training
        model = XGBClassifier(**model_params)
        model.fit(X_res, y_res)

        # Training set evaluation
        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)[:, 1]
        train_metrics.append(calculate_metrics(y_train, train_pred, train_proba))

        # Validation set evaluation
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        current_metrics = calculate_metrics(y_val, val_pred, val_proba)
        val_metrics.append(current_metrics)

        # Update best model
        if current_metrics['roc_auc'] > best_metrics['auc']:
            best_metrics.update({
                'fold': fold,
                'auc': current_metrics['roc_auc'],
                'model': model,
                'val_report': classification_report(y_val, val_pred, digits=4)
            })
            print(f"★ Fold {fold} new best | AUC: {current_metrics['roc_auc']:.4f}")

        # Collect predictions for global report
        all_y_true.extend(y_val)
        all_y_pred.extend(val_pred)

        # Print fold results
        print(f"\n▶ Fold {fold} validation results")
        print(f"  Training: Acc {train_metrics[-1]['accuracy']:.4f} | "
              f"AUC {train_metrics[-1]['roc_auc']:.4f} | F1 {train_metrics[-1]['f1']:.4f}")
        print(f"  Validation: Acc {current_metrics['accuracy']:.4f} | "
              f"AUC {current_metrics['roc_auc']:.4f} | F1 {current_metrics['f1']:.4f}")
        print(f"  Classification Report:\n{classification_report(y_val, val_pred, digits=4)}")
        print(BORDER)

    # --------------------------------------
    # Results summary
    # --------------------------------------
    def summarize_metrics(metrics, label):
        """Helper function to summarize metrics"""
        df = pd.DataFrame(metrics)
        print(f"\n{label} metrics (mean ± std)")
        print(f"  Accuracy : {df.accuracy.mean():.4f} (±{df.accuracy.std():.4f})")
        print(f"  ROC AUC  : {df.roc_auc.mean():.4f} (±{df.roc_auc.std():.4f})")
        print(f"  Recall   : {df.recall.mean():.4f} (±{df.recall.std():.4f})")
        print(f"  Precision: {df.precision.mean():.4f} (±{df.precision.std():.4f})")
        print(f"  F1 Score : {df.f1.mean():.4f} (±{df.f1.std():.4f})")

    print(BORDER)
    summarize_metrics(train_metrics, "Training")
    print(BORDER)
    summarize_metrics(val_metrics, "Validation")

    # Global classification report
    print(BORDER)
    print("\nGlobal Classification Report:")
    print(classification_report(all_y_true, all_y_pred, digits=4))

    # Save best model
    if best_metrics['model'] is not None:
        filename = f'xgboost_best_fold{best_metrics["fold"]}_auc{best_metrics["auc"]:.4f}.pkl'
        joblib.dump(best_metrics['model'], filename)
        print(BORDER)
        print("Best Model Information:")
        print(f"  ▪ From fold {best_metrics['fold']}")
        print(f"  ▪ Validation AUC: {best_metrics['auc']:.4f}")
        print(f"  ▪ Saved as: {filename}")
        print("\nClassification Report:")
        print(best_metrics['val_report'])
    else:
        print("Warning: No valid model found")


if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='XGBoost Classifier Training')

    # Add model parameters with default values
    parser.add_argument('--n_estimators', type=int, default=736,
                        help='Number of boosting rounds')
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Maximum tree depth')
    parser.add_argument('--learning_rate', type=float, default=0.014266634402839335,
                        help='Boosting learning rate')
    parser.add_argument('--colsample_bytree', type=float, default=0.7326456299835774,
                        help='Subsample ratio of columns when constructing each tree')
    parser.add_argument('--subsample', type=float, default=0.6114773832368773,
                        help='Subsample ratio of the training instances')
    parser.add_argument('--reg_lambda', type=float, default=1.0168305927212278,
                        help='L2 regularization term on weights')
    parser.add_argument('--reg_alpha', type=float, default=1.0101358532128872,
                        help='L1 regularization term on weights')
    parser.add_argument('--gamma', type=float, default=3.3725020365673264,
                        help='Minimum loss reduction required to make a further partition')
    parser.add_argument('--min_child_weight', type=int, default=2,
                        help='Minimum sum of instance weight needed in a child')

    args = parser.parse_args()

    main(args)