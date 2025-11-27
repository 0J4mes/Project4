"""
Simplified Predictive Analysis for Quick Testing
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def simple_analysis():
    """Run a simplified version of the predictive analysis"""
    print("Running Simplified Predictive Analysis...")

    # Load data
    try:
        df = pd.read_csv('processed_mushroom_data.csv')
    except FileNotFoundError:
        print("Processed data not found. Please run preprocessing first.")
        return

    # Create dummy variables
    odor_dummies = pd.get_dummies(df['odor_type'], prefix='odor')
    cap_color_dummies = pd.get_dummies(df['cap_color_type'], prefix='cap_color')

    # Create feature sets
    X_odor = odor_dummies
    X_cap = cap_color_dummies
    X_combined = pd.concat([X_odor, X_cap], axis=1)
    y = df['edible']

    # Test each feature set
    feature_sets = {
        'Odor Only': X_odor,
        'Cap Color Only': X_cap,
        'Combined': X_combined
    }

    results = {}

    for name, X in feature_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = accuracy

        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Features: {X.shape[1]}")

    # Determine best predictor
    best_predictor = max(results, key=results.get)
    best_accuracy = results[best_predictor]

    print("\n" + "=" * 50)
    print("CONCLUSION:")
    print(f"Best predictor: {best_predictor}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print("=" * 50)

    if best_predictor == "Odor Only":
        print("✅ Odor is the best predictor of mushroom edibility!")
    elif best_predictor == "Combined":
        print("✅ Combined features work best, but odor is the primary driver!")
    else:
        print("❓ Unexpected result - odor should be the best predictor.")

    return results


if __name__ == "__main__":
    simple_analysis()