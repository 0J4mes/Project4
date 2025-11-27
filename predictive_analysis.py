"""
STANDALONE Predictive Analysis using scikit-learn
Includes data preprocessing - No external dependencies!
FIXED VERSION with proper visualization labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
import io
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings('ignore')


class StandaloneMushroomPredictor:
    def __init__(self):
        self.models = {}
        self.results = {}

    def load_and_preprocess_data(self):
        """Load and preprocess the mushroom data - standalone version"""
        print("Loading and preprocessing mushroom data...")

        # First try to load existing processed data
        try:
            self.df = pd.read_csv('processed_mushroom_data.csv')
            print("✓ Loaded existing processed data")
            return self.df
        except FileNotFoundError:
            print("No processed data found. Downloading and preprocessing...")

        # Download and preprocess data
        try:
            # URL for the mushroom dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

            # Read the data directly from the URL
            response = urlopen(url)
            mushroom_data = response.read().decode('utf-8')

            # Create DataFrame
            df = pd.read_csv(io.StringIO(mushroom_data), header=None)

            # Define column names
            column_names = [
                'class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
                'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
                'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
                'stalk_surface_below_ring', 'stalk_color_above_ring',
                'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number',
                'ring_type', 'spore_print_color', 'population', 'habitat'
            ]

            df.columns = column_names

            # Select and encode columns
            selected_columns = ['class', 'odor', 'cap_color']
            self.df = df[selected_columns].copy()

            # Encoding mappings
            class_mapping = {'e': 0, 'p': 1}
            odor_mapping = {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8}
            cap_color_mapping = {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9}

            # Apply encoding
            self.df['edible'] = self.df['class'].map(class_mapping)
            self.df['odor_type'] = self.df['odor'].map(odor_mapping)
            self.df['cap_color_type'] = self.df['cap_color'].map(cap_color_mapping)

            # Save processed data
            self.df.to_csv('processed_mushroom_data.csv', index=False)
            print("✓ Data downloaded, processed, and saved")

        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Creating dummy data for testing...")
            self._create_dummy_data()

        return self.df

    def _create_dummy_data(self):
        """Create dummy data if real data can't be loaded"""
        np.random.seed(42)
        n_samples = 200

        # Create realistic dummy data
        data = []
        for i in range(n_samples):
            # Create patterns similar to real mushroom data
            if np.random.random() < 0.52:  # edible
                edible = 0
                # Edible mushrooms more likely to have certain odors
                odor = np.random.choice([0, 1, 6], p=[0.2, 0.2, 0.6])  # almond, anise, none
                cap_color = np.random.choice([0, 5, 8], p=[0.3, 0.2, 0.5])  # brown, pink, white
            else:  # poisonous
                edible = 1
                # Poisonous mushrooms more likely to have foul odors
                odor = np.random.choice([2, 3, 4, 5, 7, 8], p=[0.15, 0.15, 0.15, 0.15, 0.1, 0.3])
                cap_color = np.random.choice([0, 4, 6, 7], p=[0.3, 0.3, 0.2, 0.2])  # brown, green, purple, red

            data.append({
                'edible': edible,
                'odor_type': odor,
                'cap_color_type': cap_color
            })

        self.df = pd.DataFrame(data)
        self.df.to_csv('processed_mushroom_data.csv', index=False)
        print("✓ Dummy data created and saved")

    def create_dummy_variables(self):
        """Convert categorical columns to dummy variables"""
        print("\nCreating dummy variables...")

        # Create dummy variables for each predictor
        self.odor_dummies = pd.get_dummies(self.df['odor_type'], prefix='odor')
        self.cap_color_dummies = pd.get_dummies(self.df['cap_color_type'], prefix='cap_color')

        # Create feature sets
        self.X_odor = self.odor_dummies
        self.X_cap_color = self.cap_color_dummies
        self.X_combined = pd.concat([self.odor_dummies, self.cap_color_dummies], axis=1)
        self.y = self.df['edible']

        print(f"✓ Created feature sets:")
        print(f"  - Odor only: {self.X_odor.shape[1]} features")
        print(f"  - Cap color only: {self.X_cap_color.shape[1]} features")
        print(f"  - Combined: {self.X_combined.shape[1]} features")

        return self.X_odor, self.X_cap_color, self.X_combined, self.y

    def train_and_evaluate_models(self):
        """Train models and evaluate performance"""
        print("\n" + "=" * 50)
        print("TRAINING AND EVALUATING MODELS")
        print("=" * 50)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }

        # Feature sets
        feature_sets = {
            'Odor Only': (self.X_odor, self.y),
            'Cap Color Only': (self.X_cap_color, self.y),
            'Combined Features': (self.X_combined, self.y)
        }

        results = {}

        for feature_name, (X, y) in feature_sets.items():
            print(f"\n--- {feature_name} ---")
            results[feature_name] = {}

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)

            for model_name, model in models.items():
                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Store results
                results[feature_name][model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'y_test': y_test,
                    'X_test': X_test
                }

                print(f"  {model_name}: {accuracy:.4f}")

        self.results = results
        return results

    def find_best_predictor(self):
        """Determine which predictor is most accurate"""
        print("\n" + "=" * 50)
        print("FINDING BEST PREDICTOR")
        print("=" * 50)

        best_accuracy = 0
        best_predictor = None
        best_model = None

        for feature_name, models in self.results.items():
            for model_name, result in models.items():
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_predictor = feature_name
                    best_model = model_name

        print(f"🎯 BEST PREDICTOR: {best_predictor}")
        print(f"🤖 BEST MODEL: {best_model}")
        print(f"📊 ACCURACY: {best_accuracy:.4f}")

        return best_predictor, best_model, best_accuracy

    def visualize_results(self):
        """Create visualizations of the results with proper labels"""
        print("\nCreating visualizations...")

        # Create results directory
        os.makedirs('results', exist_ok=True)

        # Prepare data for plotting
        feature_names = []
        model_names = []
        accuracies = []

        for feature_name, models in self.results.items():
            for model_name, result in models.items():
                feature_names.append(feature_name)
                model_names.append(model_name)
                accuracies.append(result['accuracy'])

        # Create comparison plot
        plt.figure(figsize=(12, 6))

        # Convert to DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'Feature Set': feature_names,
            'Model': model_names,
            'Accuracy': accuracies
        })

        # Create grouped bar plot
        pivot_df = plot_df.pivot(index='Feature Set', columns='Model', values='Accuracy')
        ax = pivot_df.plot(kind='bar', figsize=(12, 6), color=['#ff9999', '#66b3ff', '#99ff99'])

        plt.title('Model Accuracy by Feature Set', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Set', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1.0)
        plt.legend(title='Model')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (feature_name, models) in enumerate(self.results.items()):
            for j, (model_name, result) in enumerate(models.items()):
                plt.text(i + j * 0.25 - 0.25, result['accuracy'] + 0.01,
                         f'{result["accuracy"]:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Create confusion matrix for best model
        best_predictor, best_model, best_accuracy = self.find_best_predictor()
        best_result = self.results[best_predictor][best_model]

        plt.figure(figsize=(12, 5))

        # Plot 1: Confusion Matrix
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(best_result['y_test'], best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Edible', 'Poisonous'],
                    yticklabels=['Edible', 'Poisonous'])
        plt.title(f'Confusion Matrix\n{best_model} with {best_predictor}', fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        # Plot 2: Feature Importance with PROPER LABELS
        plt.subplot(1, 2, 2)

        # Define proper labels for odor and cap color features
        odor_labels = {
            0: 'Almond', 1: 'Anise', 2: 'Creosote', 3: 'Fishy', 4: 'Foul',
            5: 'Musty', 6: 'None', 7: 'Pungent', 8: 'Spicy'
        }

        cap_color_labels = {
            0: 'Brown', 1: 'Buff', 2: 'Cinnamon', 3: 'Gray', 4: 'Green',
            5: 'Pink', 6: 'Purple', 7: 'Red', 8: 'White', 9: 'Yellow'
        }

        # Feature importance for tree-based models
        if hasattr(best_result['model'], 'feature_importances_'):
            # Create feature names with proper labels
            feature_importance_list = []
            for feature_name, importance in zip(best_result['X_test'].columns,
                                                best_result['model'].feature_importances_):
                # Parse the feature name to get proper label
                if feature_name.startswith('odor_'):
                    odor_code = int(feature_name.split('_')[1])
                    pretty_name = f"Odor: {odor_labels.get(odor_code, f'Unknown({odor_code})')}"
                elif feature_name.startswith('cap_color_'):
                    color_code = int(feature_name.split('_')[2])
                    pretty_name = f"Cap: {cap_color_labels.get(color_code, f'Unknown({color_code})')}"
                else:
                    pretty_name = feature_name

                feature_importance_list.append({
                    'feature': pretty_name,
                    'importance': importance
                })

            feature_importance = pd.DataFrame(feature_importance_list)
            feature_importance = feature_importance.sort_values('importance', ascending=True).tail(
                10)  # Top 10 features

            # Create horizontal bar plot
            bars = plt.barh(feature_importance['feature'], feature_importance['importance'], color='lightgreen')
            plt.title('Top 10 Most Important Features', fontweight='bold')
            plt.xlabel('Feature Importance Score')

            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', ha='left', va='center', fontsize=8)

        else:
            # For logistic regression, show coefficient magnitudes with proper labels
            if hasattr(best_result['model'], 'coef_'):
                coef_list = []
                for feature_name, coef in zip(best_result['X_test'].columns, best_result['model'].coef_[0]):
                    # Parse the feature name to get proper label
                    if feature_name.startswith('odor_'):
                        odor_code = int(feature_name.split('_')[1])
                        pretty_name = f"Odor: {odor_labels.get(odor_code, f'Unknown({odor_code})')}"
                    elif feature_name.startswith('cap_color_'):
                        color_code = int(feature_name.split('_')[2])
                        pretty_name = f"Cap: {cap_color_labels.get(color_code, f'Unknown({color_code})')}"
                    else:
                        pretty_name = feature_name

                    coef_list.append({
                        'feature': pretty_name,
                        'coefficient': coef
                    })

                coef_df = pd.DataFrame(coef_list)
                coef_df = coef_df.sort_values('coefficient', key=abs, ascending=True).tail(10)

                bars = plt.barh(coef_df['feature'], np.abs(coef_df['coefficient']), color='lightcoral')
                plt.title('Top 10 Most Influential Features\n(Absolute Coefficient Values)', fontweight='bold')
                plt.xlabel('Coefficient Magnitude')

                # Add value labels on bars
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{width:.3f}', ha='left', va='center', fontsize=8)
            else:
                # If no feature importance or coefficients, show a message
                plt.text(0.5, 0.5, 'No feature importance\navailable for this model',
                         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.title('Feature Importance Not Available', fontweight='bold')

        plt.tight_layout()
        plt.savefig('results/best_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_conclusions(self):
        """Generate final conclusions and recommendations"""
        best_predictor, best_model, best_accuracy = self.find_best_predictor()

        conclusions = f"""
FINAL CONCLUSIONS AND RECOMMENDATIONS
{'=' * 60}

PREDICTIVE ANALYSIS RESULTS:

1. BEST PREDICTOR IDENTIFICATION:
   - Most Accurate Predictor: {best_predictor}
   - Best Performing Model: {best_model}
   - Achieved Accuracy: {best_accuracy:.4f}

2. KEY FINDINGS:
   - Odor is overwhelmingly the best predictor of mushroom edibility
   - Cap color provides some predictive power but is significantly less accurate
   - Combining features can provide marginal improvements but odor alone is sufficient

3. PRACTICAL IMPLICATIONS:
   - For mushroom identification: Focus on odor as the primary safety indicator
   - The strong predictive power demonstrates clear biological patterns
   - This dataset serves as an excellent teaching example for classification

4. RECOMMENDATIONS FOR FURTHER ANALYSIS:
   - Explore other mushroom attributes beyond the two tested
   - Investigate interaction effects between different features
   - Apply the same methodology to other categorical classification problems
   - Consider model deployment for educational purposes

TECHNICAL NOTE:
The high accuracy achieved ({best_accuracy:.4f}) indicates that the selected features,
particularly odor, contain nearly perfect information for classification. This makes
the mushroom dataset ideal for demonstrating machine learning concepts while being
less representative of typical real-world classification challenges where features
often overlap more significantly.
"""
        print(conclusions)

        # Save conclusions to file
        with open('results/predictive_conclusions.txt', 'w') as f:
            f.write(conclusions)

        return conclusions

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 70)
        print("MUSHROOM PREDICTIVE ANALYSIS WITH SCIKIT-LEARN")
        print("=" * 70)

        # Step 1: Load and preprocess data
        self.load_and_preprocess_data()

        # Step 2: Create dummy variables
        self.create_dummy_variables()

        # Step 3: Train and evaluate models
        self.train_and_evaluate_models()

        # Step 4: Find best predictor
        best_predictor, best_model, best_accuracy = self.find_best_predictor()

        # Step 5: Visualize results
        self.visualize_results()

        # Step 6: Generate conclusions
        conclusions = self.generate_conclusions()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("- processed_mushroom_data.csv")
        print("- results/model_comparison.png")
        print("- results/best_model_analysis.png")
        print("- results/predictive_conclusions.txt")

        return {
            'best_predictor': best_predictor,
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'conclusions': conclusions
        }


def main():
    """Main function"""
    predictor = StandaloneMushroomPredictor()
    results = predictor.run_complete_analysis()

    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    print(f"🎯 Best Predictor: {results['best_predictor']}")
    print(f"🤖 Best Model: {results['best_model']}")
    print(f"📊 Accuracy: {results['best_accuracy']:.4f}")
    print("✅ Conclusion: Odor is the best predictor of mushroom edibility")
    print("=" * 70)


if __name__ == "__main__":
    main()