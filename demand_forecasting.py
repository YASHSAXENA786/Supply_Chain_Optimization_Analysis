import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data for forecasting...")
sales = pd.read_csv('sales_data.csv')
products = pd.read_csv('products.csv')
inventory = pd.read_csv('inventory.csv')

sales['date'] = pd.to_datetime(sales['date'])

# DEMAND FORECASTING MODEL
print("\n" + "="*60)
print("DEMAND FORECASTING MODEL")
print("="*60)

def prepare_forecasting_data(sales_df, product_id):
    """Prepare time series data for a specific product"""
    product_sales = sales_df[sales_df['product_id'] == product_id].copy()
    
    if len(product_sales) < 30:  # Not enough data
        return None
    
    # Create daily time series
    product_sales = product_sales.set_index('date').resample('D').agg({
        'quantity_sold': 'sum'
    }).fillna(0)
    
    # Add time features
    product_sales['year'] = product_sales.index.year
    product_sales['month'] = product_sales.index.month
    product_sales['day_of_year'] = product_sales.index.dayofyear
    product_sales['day_of_week'] = product_sales.index.dayofweek
    product_sales['is_weekend'] = product_sales.index.dayofweek.isin([5, 6]).astype(int)
    product_sales['is_holiday_season'] = product_sales['month'].isin([11, 12]).astype(int)
    
    # Lag features
    for lag in [1, 7, 14, 30]:
        product_sales[f'lag_{lag}'] = product_sales['quantity_sold'].shift(lag)
    
    # Moving averages
    product_sales['ma_7'] = product_sales['quantity_sold'].rolling(7).mean()
    product_sales['ma_14'] = product_sales['quantity_sold'].rolling(14).mean()
    product_sales['ma_30'] = product_sales['quantity_sold'].rolling(30).mean()
    
    # Rolling statistics
    product_sales['rolling_std_7'] = product_sales['quantity_sold'].rolling(7).std()
    product_sales['rolling_max_7'] = product_sales['quantity_sold'].rolling(7).max()
    product_sales['rolling_min_7'] = product_sales['quantity_sold'].rolling(7).min()
    
    return product_sales.dropna()

def build_forecasting_model(data, target_col='quantity_sold'):
    """Build and train demand forecasting model"""
    feature_cols = [col for col in data.columns if col != target_col]
    
    X = data[feature_cols]
    y = data[target_col]
    
    # Split data (last 30 days for testing)
    split_date = data.index[-30]
    train_data = data[data.index < split_date]
    test_data = data[data.index >= split_date]
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'feature_importance': feature_importance,
        'test_data': test_data,
        'predictions': test_pred
    }

# Select top 10 products by revenue for forecasting
top_products = sales.groupby('product_id')['revenue'].sum().nlargest(10).index

forecasting_results = {}
print("Building forecasting models for top 10 products...")

for i, product_id in enumerate(top_products, 1):
    print(f"Processing product {i}/10: {product_id}")
    
    # Prepare data
    product_data = prepare_forecasting_data(sales, product_id)
    if product_data is None:
        continue
    
    # Build model
    result = build_forecasting_model(product_data)
    forecasting_results[product_id] = result
    
    print(f"  - Train MAE: {result['train_mae']:.2f}")
    print(f"  - Test MAE: {result['test_mae']:.2f}")
    print(f"  - Test RMSE: {result['test_rmse']:.2f}")

# OPTIMAL INVENTORY POLICIES
print("\n" + "="*60)
print("OPTIMAL INVENTORY POLICIES")
print("="*60)

def calculate_economic_order_quantity(annual_demand, ordering_cost, holding_cost_per_unit):
    """Calculate Economic Order Quantity (EOQ)"""
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
    return int(eoq)

def calculate_reorder_point(daily_demand, lead_time, safety_factor=1.64):  # 95% service level
    """Calculate optimal reorder point"""
    demand_during_lead_time = daily_demand * lead_time
    safety_stock = safety_factor * np.sqrt(lead_time) * daily_demand * 0.3  # Assume 30% demand variability
    return int(demand_during_lead_time + safety_stock)

# Merge sales data with inventory for optimization
product_demand = sales.groupby('product_id').agg({
    'quantity_sold': ['sum', 'mean'],
    'revenue': 'sum'
}).reset_index()

product_demand.columns = ['product_id', 'annual_demand', 'daily_demand', 'annual_revenue']
product_demand['daily_demand'] = product_demand['annual_demand'] / 365

# Merge with inventory and product data
optimization_data = inventory.merge(product_demand, on='product_id', how='left')
optimization_data = optimization_data.merge(products[['product_id', 'category', 'lead_time_days']], on='product_id')

# Fill missing values
optimization_data['annual_demand'] = optimization_data['annual_demand'].fillna(0)
optimization_data['daily_demand'] = optimization_data['daily_demand'].fillna(0)

# Calculate optimal policies
ordering_cost = 50  # Assume $50 per order
optimization_data['annual_holding_cost_rate'] = 0.25  # 25% of unit cost

optimization_data['optimal_eoq'] = optimization_data.apply(
    lambda row: calculate_economic_order_quantity(
        row['annual_demand'], 
        ordering_cost, 
        row['holding_cost_per_unit'] * 365
    ), axis=1
)

optimization_data['optimal_reorder_point'] = optimization_data.apply(
    lambda row: calculate_reorder_point(
        row['daily_demand'], 
        row['lead_time_days']
    ), axis=1
)

# Calculate current vs optimal costs
optimization_data['current_annual_holding_cost'] = optimization_data['current_stock'] * optimization_data['holding_cost_per_unit'] * 365
optimization_data['current_ordering_cost'] = (optimization_data['annual_demand'] / optimization_data['current_stock'].clip(lower=1)) * ordering_cost

optimization_data['optimal_annual_holding_cost'] = (optimization_data['optimal_eoq'] / 2) * optimization_data['holding_cost_per_unit'] * 365
optimization_data['optimal_ordering_cost'] = (optimization_data['annual_demand'] / optimization_data['optimal_eoq'].clip(lower=1)) * ordering_cost

optimization_data['current_total_cost'] = optimization_data['current_annual_holding_cost'] + optimization_data['current_ordering_cost']
optimization_data['optimal_total_cost'] = optimization_data['optimal_annual_holding_cost'] + optimization_data['optimal_ordering_cost']
optimization_data['potential_savings'] = optimization_data['current_total_cost'] - optimization_data['optimal_total_cost']

# Summary of optimization results
print("INVENTORY OPTIMIZATION RESULTS:")
print(f"Products analyzed: {len(optimization_data)}")
print(f"Current total inventory cost: ${optimization_data['current_total_cost'].sum():,.2f}")
print(f"Optimal total inventory cost: ${optimization_data['optimal_total_cost'].sum():,.2f}")
print(f"Potential annual savings: ${optimization_data['potential_savings'].sum():,.2f}")