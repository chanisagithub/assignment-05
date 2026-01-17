import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Load the trained model and test data
best_model = joblib.load('lightgbm_best_model.pkl')  # Load the saved model
X_test = pd.read_csv('X_test.csv')  # Load your test data

# 1. Create the Explainer
# 'best_model' is the model you trained in the GridSearchCV step
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# 2. SUMMARY PLOT (The most important chart for your report)
# It shows which features matter most (e.g., Year, Engine CC)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Feature Importance (SHAP)", fontsize=16)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.show()

# 3. INDIVIDUAL PREDICTION EXPLANATION
# Let's explain the first car in the test set
# This shows exactly why the model predicted that specific price
shap.initjs() # Only needed if running in Jupyter
plt.figure(figsize=(10, 4))
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
plt.savefig('shap_individual_explanation.png')