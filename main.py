#!/usr/bin/env python
# coding: utf-8

"""
Man of the Match Classifier
Interpretable and explainable ML using FIFA 2018 match statistics.
"""

import numpy as np
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt
from pdpbox import pdp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- Data ---

data = pd.read_csv('FIFA 2018 Statistics.csv')

y = data['Man of the Match'] == "Yes"
feature_names = [col for col in data.columns if data[col].dtype == np.int64]
X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# --- Models ---

tree_model = DecisionTreeClassifier(
    random_state=0, max_depth=5, min_samples_split=5
).fit(train_X, train_y)

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=0
).fit(train_X, train_y)

# --- Evaluation ---

def evaluate(name, model):
    y_pred = model.predict(val_X)
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(val_y, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(val_y, y_pred))

evaluate("Decision Tree", tree_model)
evaluate("Random Forest", rf_model)

# --- Permutation Importance ---

def show_permutation_importance(name, model):
    perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
    print(f"\n{'='*40}")
    print(f"Permutation Importance: {name}")
    print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=val_X.columns.tolist())))

show_permutation_importance("Random Forest", rf_model)
show_permutation_importance("Decision Tree", tree_model)

# --- Partial Dependence Plots ---

feature_to_plot = 'Distance Covered (Kms)'

def plot_pdp(name, model):
    pdp_dist = pdp.PDPIsolate(
        model=model,
        df=val_X,
        model_features=feature_names,
        feature=feature_to_plot,
        feature_name=feature_to_plot,
    )
    fig, _ = pdp_dist.plot(engine='matplotlib')
    fig.suptitle(f"PDP — {feature_to_plot} ({name})")
    plt.tight_layout()
    plt.show()

plot_pdp("Decision Tree", tree_model)
plot_pdp("Random Forest", rf_model)
