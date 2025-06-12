# xgboost_model/model_training_evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder # Sadece tip ipucu için gerekli olabilir
import logging

class ModelTrainer:
    """
    Handles training, evaluation, and analysis of the machine learning model.
    """
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def train_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series, features_to_keep: list) -> Pipeline:
        # ... (Mevcut train_xgboost_model metodunun içeriği) ...
        logging.info("\n--- XGBoost Model Training ---")

        numeric_features = [f for f in features_to_keep if X_train[f].dtype in ['int64', 'float64']]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ],
            remainder='passthrough'
        )

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                objective=self.config["XGB_OBJECTIVE"],
                eval_metric=self.config["XGB_EVAL_METRIC"],
                n_estimators=self.config["XGB_N_ESTIMATORS"],
                learning_rate=self.config["XGB_LEARNING_RATE"],
                max_depth=self.config["XGB_MAX_DEPTH"],
                min_child_weight=self.config["XGB_MIN_CHILD_WEIGHT"],
                subsample=self.config["XGB_SUBSAMPLE"],
                colsample_bytree=self.config["XGB_COLSAMPLE_BYTREE"],
                reg_alpha=self.config["XGB_REG_ALPHA"],
                reg_lambda=self.config["XGB_REG_LAMBDA"],
                random_state=self.config["XGB_RANDOM_STATE"],
                use_label_encoder=False
            ))
        ])

        logging.info("Training XGBoost Model...")
        try:
            self.model.fit(X_train, y_train)
            logging.info("XGBoost Model training completed.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

        return self.model


    def evaluate_model_metrics(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, class_names: list) -> tuple:
        # ... (Mevcut evaluate_model_metrics metodunun içeriği) ...
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_xgboost_model first.")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        train_score = self.model.score(X_train, y_train)
        test_score = accuracy_score(y_test, y_pred)

        logging.info(f"Train Accuracy: {train_score:.4f}")
        logging.info(f"Test Accuracy: {test_score:.4f}")
        logging.info(f"Overfitting Gap (Train - Test): {train_score - test_score:.4f}")

        logging.info("\n" + "="*70)
        logging.info("                 ✨ Model Performance Evaluation on Test Set ✨")
        logging.info("="*70)
        logging.info(f"\n🚀 Overall Test Accuracy: {test_score:.4f}")
        logging.info(f"Dataset Size: {len(y_test)} samples\n")

        logging.info("--- Confusion Matrix ---")
        conf_mat = confusion_matrix(y_test, y_pred)
        conf_mat_df = pd.DataFrame(conf_mat, index=[f'Actual {c}' for c in class_names],
                                   columns=[f'Predicted {c}' for c in class_names])
        logging.info(f"\n{conf_mat_df.to_string()}")

        logging.info("\n--- Detailed Classification Report ---")
        logging.info(f"\n{classification_report(y_test, y_pred, target_names=class_names, zero_division=0)}")

        logging.info("\n--- Error Rates Per Class ---")
        errors = y_test != y_pred
        for i, class_name in enumerate(class_names):
            class_mask = y_test == i
            if class_mask.sum() > 0:
                class_error_rate = errors[class_mask].sum() / class_mask.sum()
                logging.info(f"- {class_name}: {class_error_rate:.4f}")
            else:
                logging.info(f"- {class_name}: Not present in test set. Error rate N/A.")

        return y_pred, y_pred_proba

    def analyze_model_diagnosis(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray,
                                class_names: list, features_to_keep: list, test_df_original: pd.DataFrame):
        # ... (Mevcut analyze_model_diagnosis metodunun içeriği) ...
        if accuracy_score(y_test, y_pred) > self.config["HIGH_ACCURACY_THRESHOLD"]:
            logging.warning("\n🚨 WARNING: Unusually High Accuracy Detected!")
            logging.warning("This could be due to:")
            logging.warning("  1. Exceptionally high natural separability in the data.")
            logging.warning("  2. Potential hidden data leakage sources.")
            logging.warning("  3. Data being synthetic or highly simulated.")

            logging.info("\n--- Analysis: Discriminative Power of Sensor Data ---")
            sensor_features_for_analysis = self.config["SENSOR_COLUMNS_FOR_OUTLIER_REMOVAL"]
            for feature in sensor_features_for_analysis:
                if feature in test_df_original.columns:
                    feature_by_state = test_df_original.groupby('state')[feature].agg(['mean', 'std'])
                    logging.info(f"\n📊 {feature} Statistics by State:\n{feature_by_state.to_string()}")

                    if len(feature_by_state) > 1:
                        means = feature_by_state['mean'].values
                        class_separation = np.max(means) - np.min(means)
                        avg_std = feature_by_state['std'].mean()
                        separation_ratio = class_separation / avg_std if avg_std > 0 else float('inf')
                        logging.info(f"  Class Separability Ratio (Max Mean Diff / Avg Std): {separation_ratio:.2f}")

                        if separation_ratio > self.config["HIGH_DISCRIMINATIVE_POWER_RATIO"]:
                            logging.warning(f"  ⚠️  High Discriminative Power for '{feature}' detected! This could indicate a strong signal, potentially from data leakage.")
                else:
                    logging.warning(f"\nWarning: Feature '{feature}' not found in the test_df_original for discriminative power analysis.")

        logging.info("\n--- Feature Importances ---")
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.named_steps['classifier'].feature_importances_

            feature_importance_df = pd.DataFrame({
                'feature': features_to_keep,
                'importance': importances
            }).sort_values('importance', ascending=False)

            logging.info(f"\n{feature_importance_df.to_string(index=False)}")

            max_importance = importances.max()
            if max_importance > self.config["DOMINANT_FEATURE_IMPORTANCE_THRESHOLD"] and len(features_to_keep) > 1:
                dominant_feature = feature_importance_df.iloc[0]['feature']
                logging.warning(f"\n⚠️  WARNING: '{dominant_feature}' is extremely dominant ({max_importance:.3f})!")
                logging.warning("This strongly suggests potential **overfitting** or a severe **data leakage** issue.")
        else:
            logging.info("Feature importances are not available for the selected model type or after preprocessing.")


    def compare_with_baselines(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray) -> float:
        # ... (Mevcut compare_with_baselines metodunun içeriği) ...
        logging.info("\n" + "="*50)
        logging.info("      📊 Comparison with Baseline Models")
        logging.info("="*50)

        dummy_model_most_frequent = DummyClassifier(strategy='most_frequent')
        dummy_model_most_frequent.fit(X_train, y_train)
        dummy_score_most_frequent = dummy_model_most_frequent.score(X_test, y_test)

        logging.info(f"**Baseline (Most Frequent) Accuracy:** {dummy_score_most_frequent:.4f}")
        current_model_score = accuracy_score(y_test, y_pred)
        logging.info(f"**XGBoost Model Accuracy:** {current_model_score:.4f}")
        improvement = current_model_score - dummy_score_most_frequent
        logging.info(f"**Improvement over Most Frequent Baseline:** {improvement:.4f}\n")

        dummy_random_stratified = DummyClassifier(strategy='stratified', random_state=self.config["XGB_RANDOM_STATE"])
        dummy_random_stratified.fit(X_train, y_train)
        dummy_random_score_stratified = dummy_random_stratified.score(X_test, y_test)
        logging.info(f"**Baseline (Random Stratified) Accuracy:** {dummy_random_score_stratified:.4f}")
        logging.info("A good model should significantly outperform these baselines.\n")
        logging.info("="*50)
        return improvement


    def cross_validation_check(self, X_train: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, y_pred: np.ndarray) -> float:
        # ... (Mevcut cross_validation_check metodunun içeriği) ...
        logging.info("\n" + "="*60)
        logging.info("         🔄 Time-Series Cross-Validation Check")
        logging.info("="*60)

        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_xgboost_model first.")

        tscv = TimeSeriesSplit(n_splits=3)

        try:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='accuracy', error_score='raise')
        except Exception as e:
            logging.error(f"Error during cross-validation: {e}")
            logging.warning("Cross-validation skipped due to error.")
            return np.nan

        logging.info(f"**Cross-Validation Scores (Accuracy) across folds:** {np.array2string(cv_scores, precision=4, separator=', ')}")
        logging.info(f"**Cross-Validation Mean Accuracy:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        standalone_test_accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"**Standalone Test Set Accuracy:** {standalone_test_accuracy:.4f}")

        cv_gap = cv_scores.mean() - standalone_test_accuracy
        logging.info(f"**Difference (CV Mean - Test Score):** {cv_gap:.4f}")

        if abs(cv_gap) > self.config["CV_TEST_PERFORMANCE_GAP_THRESHOLD"]:
            logging.warning("\n⚠️  WARNING: Significant difference between Cross-Validation and Test performance!")
            logging.warning("This may indicate model instability, a non-representative test set,")
            logging.warning("or potential issues with data distribution over time.")
        else:
            logging.info("\n✅ CV and Test performance are reasonably consistent, suggesting good model stability.")
        logging.info("="*60)
        return cv_gap


    def improve_error_class_performance(self, y_test: pd.Series, y_pred_proba: np.ndarray,
                                        label_encoder: LabelEncoder, class_names: list):
        # ... (Mevcut improve_error_class_performance metodunun içeriği) ...
        logging.info("\n" + "="*50)
        logging.info("  🚨 ERROR CLASS PERFORMANCE ANALYSIS")
        logging.info("="*50)

        if 'error' not in label_encoder.classes_:
            logging.warning("Warning: 'error' class not found in the dataset's unique states. Skipping error class analysis.")
            return

        try:
            error_class_idx = list(label_encoder.classes_).index('error')
        except ValueError:
            logging.warning("Error: 'error' class label not found in the label encoder's classes. Skipping analysis.")
            return

        # 1. CLASS IMBALANCE ANALYSIS
        logging.info("\n--- 1. Class Imbalance Analysis ---")
        class_counts = pd.Series(y_test).map(lambda x: class_names[x]).value_counts()
        logging.info("Class distribution in the test set:")
        logging.info(f"\n{class_counts.to_string()}")

        if 'error' in class_counts:
            error_ratio = class_counts['error'] / class_counts.sum()
            logging.info(f"**Error class ratio:** {error_ratio * 100:.2f}%")
            if error_ratio < self.config["ERROR_CLASS_IMBALANCE_THRESHOLD"]:
                logging.warning("  ⚠️  Note: 'error' class is significantly imbalanced in the test set.")
            else:
                logging.info("  'error' class balance seems reasonable in the test set.")
        else:
            logging.info("  'error' class has 0 instances in the test set. No imbalance analysis possible.")
            return

        # 2. THRESHOLD OPTIMIZATION FOR THE "ERROR" CLASS
        logging.info("\n--- 2. Threshold Optimization for 'Error' Class ---")
        error_probabilities = y_pred_proba[:, error_class_idx]

        true_error_mask = (y_test == error_class_idx)
        error_probs_true = error_probabilities[true_error_mask]
        error_probs_false = error_probabilities[~true_error_mask]

        if len(error_probs_true) > 0:
            logging.info(f"Probability statistics for the 'error' class:")
            logging.info(f"  **True error instances (n={len(error_probs_true)})**: Mean probability = {error_probs_true.mean():.4f}, Std Dev = {error_probs_true.std():.4f}")
        else:
            logging.info("No true 'error' instances in the test set. Cannot analyze their probabilities.")
            return

        if len(error_probs_false) > 0:
            logging.info(f"  **Other classes (n={len(error_probs_false)})**: Mean probability = {error_probs_false.mean():.4f}, Std Dev = {error_probs_false.std():.4f}")
        else:
            logging.info("No other class instances in the test set for probability comparison.")

        if true_error_mask.sum() > 0 and (~true_error_mask).sum() > 0:
            precision, recall, thresholds = precision_recall_curve(
                (y_test == error_class_idx).astype(int),
                error_probabilities
            )
            f1_scores = np.zeros_like(precision)
            valid_indices = (precision + recall) != 0
            f1_scores[valid_indices] = 2 * (precision[valid_indices] * recall[valid_indices]) / (precision[valid_indices] + recall[valid_indices])

            best_threshold_idx = np.argmax(f1_scores)

            if len(thresholds) > 0:
                best_threshold = thresholds[best_threshold_idx]
                best_f1 = f1_scores[best_threshold_idx]
                best_precision = precision[best_threshold_idx]
                best_recall = recall[best_threshold_idx]

                logging.info(f"\n**Optimal threshold for 'error' class prediction (based on F1-score)**: {best_threshold:.4f}")
                logging.info(f"  - **F1-score at this threshold**: {best_f1:.4f}")
                logging.info(f"  - **Precision at this threshold**: {best_precision:.4f}")
                logging.info(f"  - **Recall at this threshold**: {best_recall:.4f}")

                logging.info("\n💡 Actionable Insight: This optimal threshold can be implemented in a real-time system.")
                logging.info("If the model's predicted probability for 'error' exceeds this threshold,")
                logging.info("the system can classify the instance as 'error' to improve detection of critical states.")
            else:
                logging.warning("Could not determine optimal threshold due to insufficient data points for precision-recall curve.")
        else:
            logging.info("Not enough positive and negative samples for 'error' class to perform threshold optimization.")
        logging.info("="*50)