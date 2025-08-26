import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the datasets
print("Loading datasets...")
products = pd.read_csv('products.csv')
suppliers = pd.read_csv('suppliers.csv')
sales = pd.read_csv('sales_data.csv')
inventory = pd.read_csv('inventory.csv')
purchase_orders = pd.read_csv('purchase_orders.csv')

# Convert date columns
sales['date'] = pd.to_datetime(sales['date'])
purchase_orders['order_date'] = pd.to_datetime(purchase_orders['order_date'])
purchase_orders['expected_delivery_date'] = pd.to_datetime(purchase_orders['expected_delivery_date'])
purchase_orders['actual_delivery_date'] = pd.to_datetime(purchase_orders['actual_delivery_date'])

print("Data loaded successfully!")

# 1. INVENTORY PERFORMANCE ANALYSIS
print("\n" + "="*60)
print("1. INVENTORY PERFORMANCE ANALYSIS")
print("="*60)

# Merge data for comprehensive analysis
inventory_analysis = inventory.merge(products[['product_id', 'category', 'unit_cost', 'selling_price']], on='product_id')

# Calculate inventory metrics
inventory_analysis['inventory_value'] = inventory_analysis['current_stock'] * inventory_analysis['unit_cost']
inventory_analysis['annual_holding_cost'] = inventory_analysis['inventory_value'] * 0.25
inventory_analysis['turnover_ratio'] = inventory_analysis['avg_daily_demand'] * 365 / (inventory_analysis['current_stock'] + 1)
inventory_analysis['stockout_risk'] = np.where(inventory_analysis['current_stock'] <= inventory_analysis['safety_stock'], 'High Risk', 'Safe')

# ABC Classification based on annual revenue
sales_by_product = sales.groupby('product_id').agg({
    'revenue': 'sum',
    'quantity_sold': 'sum'
}).reset_index()

inventory_analysis = inventory_analysis.merge(sales_by_product, on='product_id', how='left')
inventory_analysis['annual_revenue'] = inventory_analysis['revenue'].fillna(0)

# ABC Classification
inventory_analysis = inventory_analysis.sort_values('annual_revenue', ascending=False)
inventory_analysis['cumulative_revenue'] = inventory_analysis['annual_revenue'].cumsum()
total_revenue = inventory_analysis['annual_revenue'].sum()
inventory_analysis['cumulative_percentage'] = inventory_analysis['cumulative_revenue'] / total_revenue * 100

def classify_abc(cum_percent):
    if cum_percent <= 80:
        return 'A'
    elif cum_percent <= 95:
        return 'B'
    else:
        return 'C'

inventory_analysis['abc_category'] = inventory_analysis['cumulative_percentage'].apply(classify_abc)

print(f"Total Inventory Value: ${inventory_analysis['inventory_value'].sum():,.2f}")
print(f"Total Annual Holding Cost: ${inventory_analysis['annual_holding_cost'].sum():,.2f}")
print(f"Average Inventory Turnover: {inventory_analysis['turnover_ratio'].mean():.2f}")
print(f"Products at Stockout Risk: {(inventory_analysis['stockout_risk'] == 'High Risk').sum()}")

print("\nABC Classification Results:")
abc_summary = inventory_analysis.groupby('abc_category').agg({
    'product_id': 'count',
    'annual_revenue': 'sum',
    'inventory_value': 'sum'
}).round(2)
abc_summary.columns = ['Product Count', 'Annual Revenue', 'Inventory Value']
print(abc_summary)

# 2. DEMAND FORECASTING
print("\n" + "="*60)
print("2. DEMAND FORECASTING ANALYSIS")
print("="*60)

# Prepare data for forecasting
daily_sales = sales.groupby('date').agg({
    'quantity_sold': 'sum',
    'revenue': 'sum'
}).reset_index()

# Add time features
daily_sales['year'] = daily_sales['date'].dt.year
daily_sales['month'] = daily_sales['date'].dt.month
daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6])

# Calculate moving averages
daily_sales['ma_7'] = daily_sales['quantity_sold'].rolling(window=7).mean()
daily_sales['ma_30'] = daily_sales['quantity_sold'].rolling(window=30).mean()

# Seasonal analysis
seasonal_analysis = daily_sales.groupby('month').agg({
    'quantity_sold': ['mean', 'std'],
    'revenue': 'mean'
}).round(2)

print("Monthly Seasonal Patterns:")
seasonal_analysis.columns = ['Avg_Quantity', 'Std_Quantity', 'Avg_Revenue']
print(seasonal_analysis)

# 3. SUPPLIER PERFORMANCE ANALYSIS  
print("\n" + "="*60)
print("3. SUPPLIER PERFORMANCE ANALYSIS")
print("="*60)

# Analyze supplier performance
supplier_performance = purchase_orders.groupby('supplier_id').agg({
    'delivery_delay_days': ['mean', 'std', 'count'],
    'quality_score': 'mean',
    'total_cost': 'sum',
    'quantity_received': 'sum'
}).round(2)

supplier_performance.columns = ['Avg_Delay', 'Delay_Std', 'Order_Count', 'Avg_Quality', 'Total_Spend', 'Total_Quantity']

# Add supplier details
supplier_performance = supplier_performance.merge(
    suppliers[['supplier_id', 'supplier_name', 'country', 'reliability_score']], 
    on='supplier_id'
)

# Calculate performance scores
supplier_performance['on_time_delivery'] = np.where(supplier_performance['Avg_Delay'] <= 0, 1, 
                                                   1 - (supplier_performance['Avg_Delay'] / 10))
supplier_performance['performance_score'] = (
    supplier_performance['on_time_delivery'] * 0.4 + 
    supplier_performance['Avg_Quality'] * 0.6
)

print("Top 5 Suppliers by Performance Score:")
top_suppliers = supplier_performance.nlargest(5, 'performance_score')[
    ['supplier_name', 'performance_score', 'Avg_Delay', 'Avg_Quality', 'Total_Spend']
]
print(top_suppliers)

print("\nWorst 3 Suppliers by Performance Score:")
worst_suppliers = supplier_performance.nsmallest(3, 'performance_score')[
    ['supplier_name', 'performance_score', 'Avg_Delay', 'Avg_Quality', 'Total_Spend']
]
print(worst_suppliers)

# 4. COST OPTIMIZATION OPPORTUNITIES
print("\n" + "="*60)
print("4. COST OPTIMIZATION OPPORTUNITIES")
print("="*60)

# Identify overstock and understock situations
optimization_opportunities = inventory_analysis.copy()
optimization_opportunities['days_of_stock'] = optimization_opportunities['current_stock'] / (optimization_opportunities['avg_daily_demand'] + 0.01)

# Categorize inventory status
def categorize_stock_status(days_of_stock):
    if days_of_stock > 90:
        return 'Overstock'
    elif days_of_stock > 30:
        return 'Normal'
    elif days_of_stock > 7:
        return 'Low Stock'
    else:
        return 'Critical'

optimization_opportunities['stock_status'] = optimization_opportunities['days_of_stock'].apply(categorize_stock_status)

print("Inventory Status Summary:")
status_summary = optimization_opportunities.groupby('stock_status').agg({
    'product_id': 'count',
    'inventory_value': 'sum',
    'annual_holding_cost': 'sum'
}).round(2)
status_summary.columns = ['Product Count', 'Inventory Value', 'Annual Holding Cost']
print(status_summary)

# Calculate potential savings
overstock_items = optimization_opportunities[optimization_opportunities['stock_status'] == 'Overstock']
potential_savings = overstock_items['annual_holding_cost'].sum() * 0.5  # Assume 50% reduction possible

print(f"\nPotential Annual Savings from Overstock Reduction: ${potential_savings:,.2f}")

# 5. KEY INSIGHTS AND RECOMMENDATIONS
print("\n" + "="*60)
print("5. KEY INSIGHTS AND RECOMMENDATIONS")
print("="*60)

total_inventory_value = inventory_analysis['inventory_value'].sum()
high_risk_products = len(inventory_analysis[inventory_analysis['stockout_risk'] == 'High Risk'])
avg_delivery_delay = purchase_orders['delivery_delay_days'].mean()

print("EXECUTIVE SUMMARY:")
print(f"• Total Inventory Investment: ${total_inventory_value:,.2f}")
print(f"• Annual Holding Costs: ${inventory_analysis['annual_holding_cost'].sum():,.2f}")
print(f"• Products at Stockout Risk: {high_risk_products}")
print(f"• Average Supplier Delivery Delay: {avg_delivery_delay:.1f} days")
print(f"• Potential Cost Savings: ${potential_savings:,.2f}")

print("\nRECOMMendations:")
print("1. Focus on Class A products (80% of revenue) for tighter inventory control")
print("2. Reduce overstock on slow-moving items to save holding costs")
print("3. Improve supplier relationships with poor delivery performance")
print("4. Implement dynamic safety stock levels based on demand variability")
print("5. Consider vendor-managed inventory for critical suppliers")

# Save analysis results
inventory_analysis.to_csv('inventory_analysis_results.csv', index=False)
supplier_performance.to_csv('supplier_performance_results.csv', index=False)
optimization_opportunities.to_csv('optimization_opportunities.csv', index=False)

print(f"\nAnalysis complete! Results saved to CSV files.")
