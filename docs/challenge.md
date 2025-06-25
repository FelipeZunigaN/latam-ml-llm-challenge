# Part 1

## Review of Notebook - Exploration.ipynb

### Plot Fix
**Data Analyst: First Sight**

* ✅ Vertical space increased with figsize=(10, 5), improving readability.
* ✅ Switched to horizontal bar chart, which is much more readable when category names (airline names) are long.
* ✅ Cleaner and more concise: unnecessary styling and label rotation removed.
* ✅ Used keyword arguments (x and y) for clarity.
* ✅ Change X and Y title labels according to their data

### Features Generation

I refactored the get_period_day function, which classifies a flight's departure time into a period of the day (morning, afternoon, night). 

```python
from datetime import datetime

def get_period_day(date):
    """
    Determina el período del día basado en la hora.
    
    Períodos:
    - morning: 5:00 - 11:59
    - afternoon: 12:00 - 18:59  
    - night: 19:00 - 4:59 (incluye madrugada del día siguiente)
    
    Args:
        fecha: pandas Series con fechas y horas
    
    Returns:
        pandas Series con 'morning', 'afternoon', 'night'
    """
    dates = pd.to_datetime(date)
    
    # Extraer la hora (0-23)
    hour = dates.dt.hour

    # Crear condiciones para cada período
    morning_mask = (hour >= 5) & (hour <= 11)
    afternoon_mask = (hour >= 12) & (hour <= 18)
    night_mask = (hour >= 19) | (hour <= 4)

    # FORMA PROFESIONAL 1: Usar np.select (MÁS COMÚN Y ELEGANTE)
    conditions = [morning_mask, afternoon_mask, night_mask]
    choices = ['morning', 'afternoon', 'night']
    
    return np.select(conditions, choices, default='unknown')
```

* ✅ Uses pandas and numpy vectorized operations.
* ✅ Scales efficiently with large datasets.
* ✅ Provides standardized, English period labels for compatibility.
* ✅ Cleaner, shorter, and easier to maintain.

# Model Training - Part 4

After testing both **Logistic Regression** and **XGBoost Classifier** using the top 10 most relevant features, we found that both models achieved **nearly identical performance metrics**, particularly in recall and F1-score for the positive class (delayed flights).

Despite XGBoost being a powerful algorithm widely used in the industry, we decided to proceed with Logistic Regression for the following reasons:

- **Interpretability**  
  Logistic Regression provides easily interpretable coefficients, allowing us to clearly understand the effect of each feature on flight delay predictions.

- **Simplicity and Speed**  
  Logistic Regression is fast to train, easy to debug, and simpler to deploy. It also requires fewer computational resources.

- **Transparency for Stakeholders**  
  In real-world business contexts, it’s crucial to explain how and why a model makes certain predictions. Logistic Regression offers that transparency out of the box.

- **Performance Parity**  
  In our experiments, XGBoost offered no significant improvement in performance. Key metrics such as accuracy, recall, and F1-score were nearly identical between the two models.

Therefore, Logistic Regression is preferred as it achieves comparable results while being more transparent and production-friendly for this use case.

### 📊 Summary of Model Performance (Top 10 Features)

| Metric        | Logistic Regression | XGBoost Classifier |
|---------------|---------------------|---------------------|
| Accuracy      | 0.55                | 0.55                |
| Recall (class 1) | 0.69             | 0.69                |
| F1-score (class 1) | 0.36           | 0.37                |

### ✅ Final Decision

> Therefore **Logistic Regression** is selected as the final model because it offers comparable predictive performance with higher transparency, better explainability, and lower complexity — all critical aspects in real-world applications.


# 🧠 `DelayModel`: ML Pipeline Class for Flight Delay Prediction

The `DelayModel` class encapsulates the end-to-end logic for training and predicting flight delays using a `LogisticRegression` model. It is designed to be modular and reusable across different parts of the project.

### 🔧 Features

- Loads and saves the trained model using `joblib`.
- Computes the target label (`delay > 15 minutes`) from timestamp columns.
- Performs feature engineering (`min_diff` in minutes) and one-hot encoding.
- Ensures consistent input shape with a predefined set of top features.
- Trains a `LogisticRegression` model with class balancing.
- Predicts using the pre-trained model.

### 🧩 Key Methods

| Method                 | Description                                                   |
|------------------------|---------------------------------------------------------------|
| `fit(X, y)`            | Trains the logistic regression model and saves it to disk.    |
| `predict(X)`           | Returns predicted labels from the trained model.              |
| `preprocess(df)`       | Applies feature engineering and encoding to raw input data.   |
| `get_min_diff(df)`     | Computes the delay (in minutes) from timestamp columns.       |
| `_save_model()` / `_load_model()` | Handles model persistence.                         |