import numpy as np
import pandas as pd
import json
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class GestureModelTrainer:
    """
    Trainer for custom gesture recognition models
    """
    
    def __init__(self):
        self.data_dir = "data/raw"
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model types
        self.model_types = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42),
            'neural_network': None  # Will be created dynamically
        }
        
        # Training results
        self.training_results = {}
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess gesture data
        
        Returns:
            Tuple of (features, labels)
        """
        print("Loading gesture data...")
        
        all_data = []
        gesture_labels = []
        
        # Find all JSON files in data directory
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        if not json_files:
            raise ValueError("No data files found. Please collect some gesture data first.")
        
        for file_path in json_files:
            gesture_name = os.path.basename(file_path).split('_')[0]
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for sample in data:
                features = np.array(sample['features'])
                all_data.append(features)
                gesture_labels.append(gesture_name)
        
        features = np.array(all_data)
        labels = np.array(gesture_labels)
        
        print(f"Loaded {len(features)} samples for {len(set(labels))} gestures")
        print(f"Gesture classes: {list(set(labels))}")
        
        return features, labels
    
    def preprocess_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
        """
        Preprocess data for training
        
        Args:
            features: Feature array
            labels: Label array
            
        Returns:
            Tuple of (X_train, y_train, scaler, label_encoder)
        """
        print("Preprocessing data...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, 
                scaler, label_encoder)
    
    def create_neural_network(self, input_dim: int, num_classes: int) -> keras.Model:
        """
        Create a neural network model
        
        Args:
            input_dim: Number of input features
            num_classes: Number of gesture classes
            
        Returns:
            Compiled neural network model
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, save_models: bool = True) -> Dict:
        """
        Train all model types
        
        Args:
            save_models: Whether to save trained models
            
        Returns:
            Dictionary with training results
        """
        print("Starting model training...")
        
        # Load and preprocess data
        features, labels = self.load_data()
        X_train, X_test, y_train, y_test, scaler, label_encoder = self.preprocess_data(features, labels)
        
        results = {}
        
        # Train each model type
        for model_name, model in self.model_types.items():
            print(f"\nTraining {model_name}...")
            
            if model_name == 'neural_network':
                # Create neural network
                model = self.create_neural_network(X_train.shape[1], len(label_encoder.classes_))
                
                # Train neural network
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1
                )
                
                # Evaluate
                y_pred = np.argmax(model.predict(X_test), axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'history': history,
                    'scaler': scaler,
                    'label_encoder': label_encoder
                }
                
            else:
                # Train traditional ML model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'scaler': scaler,
                    'label_encoder': label_encoder
                }
            
            print(f"{model_name} accuracy: {accuracy:.4f}")
        
        # Save models if requested
        if save_models:
            self._save_models(results)
        
        self.training_results = results
        return results
    
    def _save_models(self, results: Dict):
        """Save trained models"""
        print("Saving models...")
        
        for model_name, result in results.items():
            if model_name == 'neural_network':
                # Save neural network
                model_path = os.path.join(self.models_dir, f"{model_name}.h5")
                result['model'].save(model_path)
                
                # Save preprocessing components
                scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
                label_encoder_path = os.path.join(self.models_dir, f"{model_name}_label_encoder.pkl")
                
                import pickle
                with open(scaler_path, 'wb') as f:
                    pickle.dump(result['scaler'], f)
                with open(label_encoder_path, 'wb') as f:
                    pickle.dump(result['label_encoder'], f)
                
            else:
                # Save traditional ML model
                model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
                scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
                label_encoder_path = os.path.join(self.models_dir, f"{model_name}_label_encoder.pkl")
                
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(result['scaler'], f)
                with open(label_encoder_path, 'wb') as f:
                    pickle.dump(result['label_encoder'], f)
        
        print("Models saved successfully!")
    
    def evaluate_models(self) -> Dict:
        """
        Evaluate trained models
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.training_results:
            print("No trained models found. Please train models first.")
            return {}
        
        print("Evaluating models...")
        
        evaluation_results = {}
        
        for model_name, result in self.training_results.items():
            print(f"\nEvaluating {model_name}...")
            
            # Load test data
            features, labels = self.load_data()
            X_train, X_test, y_train, y_test, _, _ = self.preprocess_data(features, labels)
            
            # Make predictions
            if model_name == 'neural_network':
                y_pred = np.argmax(result['model'].predict(X_test), axis=1)
            else:
                y_pred = result['model'].predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
        
        return evaluation_results
    
    def plot_results(self, evaluation_results: Dict):
        """Plot training and evaluation results"""
        if not evaluation_results:
            print("No evaluation results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gesture Recognition Model Evaluation', fontsize=16)
        
        # Plot 1: Accuracy comparison
        model_names = list(evaluation_results.keys())
        accuracies = [evaluation_results[name]['accuracy'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracies, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
        
        # Plot 2: Confusion matrices
        for i, model_name in enumerate(model_names[:2]):
            row = i // 2
            col = i % 2
            
            conf_matrix = evaluation_results[model_name]['confusion_matrix']
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[row, col])
            axes[row, col].set_title(f'{model_name} Confusion Matrix')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Plot 3: Training history (for neural network)
        if 'neural_network' in self.training_results:
            history = self.training_results['neural_network']['history']
            axes[1, 0].plot(history.history['accuracy'], label='Training Accuracy')
            axes[1, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[1, 0].set_title('Neural Network Training History')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
        
        # Plot 4: Feature importance (for Random Forest)
        if 'random_forest' in self.training_results:
            rf_model = self.training_results['random_forest']['model']
            feature_importance = rf_model.feature_importances_
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
            
            # Sort by importance
            sorted_idx = np.argsort(feature_importance)[::-1]
            axes[1, 1].bar(range(len(feature_importance)), 
                          feature_importance[sorted_idx])
            axes[1, 1].set_title('Random Forest Feature Importance')
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Importance')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_gesture(self, features: np.ndarray, model_name: str = 'neural_network') -> str:
        """
        Predict gesture using trained model
        
        Args:
            features: Input features
            model_name: Name of the model to use
            
        Returns:
            Predicted gesture label
        """
        if model_name not in self.training_results:
            raise ValueError(f"Model '{model_name}' not found in training results")
        
        result = self.training_results[model_name]
        model = result['model']
        scaler = result['scaler']
        label_encoder = result['label_encoder']
        
        # Preprocess features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        if model_name == 'neural_network':
            prediction = np.argmax(model.predict(features_scaled), axis=1)
        else:
            prediction = model.predict(features_scaled)
        
        # Decode label
        gesture_label = label_encoder.inverse_transform(prediction)[0]
        
        return gesture_label

def main():
    """Main function for model training"""
    trainer = GestureModelTrainer()
    
    print("Gesture Model Trainer")
    print("=" * 30)
    
    try:
        # Train models
        print("Training models...")
        results = trainer.train_models(save_models=True)
        
        # Evaluate models
        print("\nEvaluating models...")
        evaluation_results = trainer.evaluate_models()
        
        # Plot results
        print("\nPlotting results...")
        trainer.plot_results(evaluation_results)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main() 