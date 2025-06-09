# Car Price Prediction Web Application

A Flask-based web application that predicts car prices using machine learning. The application uses a Decision Tree Regressor to estimate vehicle prices based on key automotive specifications including fuel type, engine type, engine size, and horsepower, with results displayed in both original currency and converted to Indian Rupees.

## Features

- **Automotive Price Prediction**: AI-powered car valuation system
- **Multi-Currency Display**: Results shown in original currency and INR conversion
- **Regression Analysis**: Continuous price prediction (not just categories)
- **Automotive Specifications**: Uses key vehicle parameters for accurate pricing
- **Real-time Valuation**: Instant car price estimates

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn (Decision Tree Regressor)
- **Data Processing**: pandas, numpy
- **Frontend**: HTML templates (Jinja2)
- **Dataset**: Car specifications and pricing data (CSV format)

## Project Structure

```
car-price-predictor/
│
├── app.py                # Main Flask application
├── car.csv              # Car dataset with specifications and prices
├── templates/
│   └── car.html         # Main page with form and results
├── static/              # CSS, JS, images (if any)
└── README.md           # Project documentation
```

## Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd car-price-predictor
```

### Step 2: Install Dependencies

```bash
pip install flask pandas numpy scikit-learn
```

### Step 3: Prepare the Dataset

Ensure `car.csv` is in the root directory with 5 columns:
- Column 1: Fuel Type (encoded as integer)
- Column 2: Engine Type (encoded as integer)
- Column 3: Engine Size (float/numeric)
- Column 4: Horsepower (float/numeric)
- Column 5: Price (target variable in original currency)

### Step 4: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. **Access Application**: Navigate to `http://localhost:5000`
2. **Enter Car Specifications**: Fill in vehicle details
3. **Submit Form**: Click predict to get price estimation
4. **View Results**: See predicted price in original currency and INR

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page with car price prediction form |
| `/Car` | POST | Process car specifications and return price prediction |

## Model Details

- **Algorithm**: Decision Tree Regressor
- **Features**: Fuel Type, Engine Type, Engine Size, Horsepower
- **Target**: Car Price (Continuous Variable)
- **Currency Conversion**: Automatic conversion to INR (₹82.04 rate)
- **Training**: Model trains on entire dataset for each prediction

## Input Parameters

### Required Car Specifications

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| **Fuel Type** | Integer | Type of fuel system | 0=Petrol, 1=Diesel, 2=Electric, 3=Hybrid |
| **Engine Type** | Integer | Engine configuration | 0=4-Cylinder, 1=6-Cylinder, 2=V8, 3=Electric |
| **Engine Size** | Float | Engine displacement (Liters) | 1.0 - 6.0+ liters |
| **Horsepower** | Integer/Float | Engine power output | 100 - 500+ HP |

### Typical Value Ranges

#### Fuel Type Encoding
- **0**: Petrol/Gasoline
- **1**: Diesel
- **2**: Electric
- **3**: Hybrid
- **4**: Other (CNG, LPG, etc.)

#### Engine Type Encoding
- **0**: 4-Cylinder Inline
- **1**: 6-Cylinder Inline
- **2**: V6 Engine
- **3**: V8 Engine
- **4**: Electric Motor
- **5**: Rotary/Other

#### Engine Size Categories
- **Compact**: 1.0 - 1.6L
- **Mid-size**: 1.7 - 2.5L
- **Large**: 2.6 - 4.0L
- **Performance**: 4.0L+

#### Horsepower Categories
- **Economy**: 100-150 HP
- **Standard**: 150-250 HP
- **Performance**: 250-400 HP
- **High Performance**: 400+ HP

## Sample Input Examples

### Economy Car
```
Fuel Type: 0 (Petrol)
Engine Type: 0 (4-Cylinder)
Engine Size: 1.4
Horsepower: 120
Expected Price: $15,000 - $25,000
```

### Mid-Range Sedan
```
Fuel Type: 0 (Petrol)
Engine Type: 1 (6-Cylinder)
Engine Size: 2.5
Horsepower: 200
Expected Price: $25,000 - $40,000
```

### Luxury/Performance Car
```
Fuel Type: 0 (Petrol)
Engine Type: 2 (V8)
Engine Size: 4.0
Horsepower: 400
Expected Price: $60,000 - $100,000+
```

### Electric Vehicle
```
Fuel Type: 2 (Electric)
Engine Type: 4 (Electric Motor)
Engine Size: 0.0 (N/A for electric)
Horsepower: 300
Expected Price: $40,000 - $80,000
```

## Currency Conversion

The application displays results in two formats:
- **Original Currency**: Base prediction value
- **Indian Rupees**: Converted using rate of ₹82.04 per unit

```python
# Conversion formula used
inr_price = original_price * 82.04
```

## Dataset Format

The `car.csv` file should contain numerical data:

```csv
0,0,1.4,120,18500
1,0,2.0,150,22000
0,1,2.5,200,32000
2,4,0.0,300,45000
...
```

## Model Performance Considerations

### Advantages of Decision Tree Regressor
- **Non-linear Relationships**: Captures complex price patterns
- **Feature Interactions**: Handles combinations of specifications
- **Interpretability**: Can visualize decision paths
- **No Scaling Required**: Works with different feature ranges

### Potential Limitations
- **Overfitting Risk**: May memorize training data
- **Sensitive to Outliers**: Extreme values can skew predictions
- **Instability**: Small data changes affect tree structure

## Improvement Suggestions

### 1. Model Enhancement
```python
# Add Random Forest for better stability
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Add cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Feature importance analysis
importance = model.feature_importances_
```

### 2. Additional Features
- **Brand/Make**: Vehicle manufacturer
- **Model Year**: Age of the vehicle
- **Mileage**: Fuel efficiency rating
- **Transmission**: Manual vs Automatic
- **Body Type**: Sedan, SUV, Hatchback, etc.
- **Safety Rating**: NCAP or IIHS scores
- **Market Segment**: Luxury, Economy, Sports

### 3. Enhanced Validation
```python
def validate_car_specs(data):
    # Engine size validation
    if data['enginesize'] < 0 or data['enginesize'] > 8.0:
        return False, "Engine size must be between 0 and 8.0 liters"
    
    # Horsepower validation
    if data['horsepower'] < 50 or data['horsepower'] > 1000:
        return False, "Horsepower must be between 50 and 1000"
    
    return True, "Valid"
```

### 4. Security Improvements
⚠️ **Security Warning**: Replace `eval()` with safer alternatives:

```python
# Replace eval() with safe parsing
try:
    Enginesize = float(request.form.get("enginesize"))
    Horsepower = float(request.form.get("horsepower"))
except (ValueError, TypeError):
    return render_template("car.html", error="Invalid input format")
```

### 5. Advanced Features
- **Price Range Prediction**: Show confidence intervals
- **Market Comparison**: Compare with similar vehicles
- **Depreciation Analysis**: Predict value over time
- **Regional Pricing**: Adjust for local markets
- **Condition Factor**: Account for vehicle condition

## Business Applications

### Use Cases
- **Car Dealerships**: Quick vehicle valuation
- **Insurance Companies**: Claim assessment
- **Individual Buyers**: Purchase decision support
- **Fleet Management**: Asset valuation
- **Loan Assessment**: Collateral evaluation

### Market Integration
```python
# Real-time market data integration
import requests

def get_market_adjustment():
    # Fetch current market trends
    response = requests.get("api.carmarket.com/trends")
    adjustment_factor = response.json()['adjustment']
    return adjustment_factor

# Apply market adjustment
adjusted_price = base_price * get_market_adjustment()
```

## Performance Optimization

### Model Caching
```python
import joblib
from functools import lru_cache

# Save trained model
joblib.dump(model, "car_price_model.pkl")

# Load cached model
@lru_cache(maxsize=1)
def get_car_model():
    return joblib.load("car_price_model.pkl")
```

### Batch Processing
```python
def batch_predict_prices(car_specs_list):
    model = get_car_model()
    predictions = model.predict(car_specs_list)
    return predictions
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install flask pandas numpy scikit-learn
   ```

2. **CSV Format Issues**:
   - Ensure 5 columns with numerical data
   - Check for missing values
   - Verify proper encoding for categorical variables

3. **Eval() Security Risk**:
   - Replace with `float()` or `int()` conversion
   - Add try-catch for input validation

4. **Currency Conversion**:
   - Update exchange rate regularly
   - Consider using real-time currency APIs

### Data Quality Checks
```python
def validate_dataset(df):
    # Check for missing values
    missing_data = df.isnull().sum()
    
    # Check for unrealistic values
    unrealistic_prices = df[df['price'] < 1000]  # Too low
    expensive_cars = df[df['price'] > 500000]    # Very high
    
    return missing_data, unrealistic_prices, expensive_cars
```

## Deployment Considerations

### Production Setup
- **Environment Variables**: Store sensitive configurations
- **Database Integration**: Replace CSV with proper database
- **API Rate Limiting**: Prevent abuse
- **Caching Layer**: Redis for frequent predictions
- **Monitoring**: Track prediction accuracy and usage

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/automotive-enhancement`)
3. Commit changes (`git commit -am 'Add vehicle feature'`)
4. Push to branch (`git push origin feature/automotive-enhancement`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Automotive Resources

- **Kelley Blue Book**: https://kbb.com
- **Edmunds**: https://edmunds.com
- **AutoTrader**: https://autotrader.com
- **Cars.com**: https://cars.com

## Contact

For questions, automotive expertise, or technical support, please open an issue in the repository.

---

**Note**: Car prices are estimates based on specifications and historical data. Actual market values may vary based on condition, location, demand, and other factors not included in this model. Always consult multiple sources and professional appraisers for important purchasing decisions.
