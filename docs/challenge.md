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

