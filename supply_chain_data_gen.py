import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate Product Master Data
def generate_products():
    categories = ['Laptops', 'Smartphones', 'Tablets', 'Accessories', 'Audio', 'Gaming']
    brands = ['TechCorp', 'InnoDevice', 'SmartTech', 'EliteElectronics', 'NextGen']
    
    products = []
    for i in range(100):  # 100 different products
        product_id = f"PROD_{i+1:03d}"
        category = random.choice(categories)
        brand = random.choice(brands)
        
        # Price ranges by category
        price_ranges = {
            'Laptops': (400, 2000),
            'Smartphones': (200, 1200),
            'Tablets': (150, 800),
            'Accessories': (10, 100),
            'Audio': (30, 500),
            'Gaming': (50, 600)
        }
        
        unit_cost = random.uniform(*price_ranges[category])
        selling_price = unit_cost * random.uniform(1.3, 2.0)  # 30-100% markup
        
        products.append({
            'product_id': product_id,
            'product_name': f"{brand} {category[:-1]} Model {i+1}",
            'category': category,
            'brand': brand,
            'unit_cost': round(unit_cost, 2),
            'selling_price': round(selling_price, 2),
            'weight_kg': random.uniform(0.1, 3.0),
            'lead_time_days': random.randint(7, 45)
        })
    
    return pd.DataFrame(products)

# Generate Supplier Data
def generate_suppliers():
    supplier_names = ['Global Supply Co', 'Asia Electronics', 'European Tech', 'American Components', 
                     'Pacific Trading', 'Metro Wholesale', 'Prime Distributors', 'Elite Suppliers']
    countries = ['China', 'USA', 'Germany', 'Japan', 'South Korea', 'Taiwan', 'Singapore']
    
    suppliers = []
    for i, name in enumerate(supplier_names):
        supplier_id = f"SUP_{i+1:03d}"
        suppliers.append({
            'supplier_id': supplier_id,
            'supplier_name': name,
            'country': random.choice(countries),
            'reliability_score': random.uniform(0.85, 0.99),  # 85-99% reliability
            'lead_time_variance': random.uniform(0.1, 0.3),   # 10-30% variance
            'minimum_order_qty': random.randint(10, 500)
        })
    
    return pd.DataFrame(suppliers)

# Generate Historical Sales Data (2 years)
def generate_sales_data(products):
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    sales_data = []
    current_date = start_date
    
    while current_date <= end_date:
        # Different sales patterns for different categories
        for _, product in products.iterrows():
            # Base demand influenced by category
            base_demand_map = {
                'Laptops': 15, 'Smartphones': 25, 'Tablets': 12,
                'Accessories': 50, 'Audio': 20, 'Gaming': 18
            }
            base_demand = base_demand_map[product['category']]
            
            # Seasonal effects
            month = current_date.month
            seasonal_factor = 1.0
            if month in [11, 12]:  # Holiday season
                seasonal_factor = 1.5
            elif month in [6, 7, 8]:  # Summer
                seasonal_factor = 1.2
            elif month in [1, 2]:  # Post-holiday
                seasonal_factor = 0.7
            
            # Weekend effect
            weekday = current_date.weekday()
            if weekday >= 5:  # Weekend
                seasonal_factor *= 0.6
            
            # Random demand with some products being more popular
            popularity_factor = random.uniform(0.5, 2.0)
            daily_demand = max(0, int(np.random.poisson(base_demand * seasonal_factor * popularity_factor)))
            
            if daily_demand > 0:
                sales_data.append({
                    'date': current_date,
                    'product_id': product['product_id'],
                    'quantity_sold': daily_demand,
                    'revenue': daily_demand * product['selling_price'],
                    'profit': daily_demand * (product['selling_price'] - product['unit_cost'])
                })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(sales_data)

# Generate Inventory Data
def generate_inventory_data(products, suppliers):
    inventory_data = []
    
    for _, product in products.iterrows():
        # Assign random supplier
        supplier = suppliers.sample(1).iloc[0]
        
        # Current inventory levels
        avg_daily_sales = random.uniform(5, 30)  # Estimated daily sales
        safety_stock = int(avg_daily_sales * random.uniform(3, 10))  # 3-10 days safety stock
        current_stock = random.randint(0, safety_stock * 3)
        
        inventory_data.append({
            'product_id': product['product_id'],
            'supplier_id': supplier['supplier_id'],
            'current_stock': current_stock,
            'safety_stock': safety_stock,
            'reorder_point': safety_stock + int(avg_daily_sales * product['lead_time_days']),
            'max_stock_level': safety_stock * 4,
            'avg_daily_demand': round(avg_daily_sales, 2),
            'holding_cost_per_unit': round(product['unit_cost'] * 0.25 / 365, 4),  # 25% annual holding cost
            'stockout_cost_per_unit': round(product['selling_price'] * 0.1, 2)  # 10% of selling price
        })
    
    return pd.DataFrame(inventory_data)

# Generate Purchase Order Data
def generate_purchase_orders(products, suppliers, start_date=datetime(2022, 1, 1)):
    po_data = []
    po_id = 1000
    
    for i in range(500):  # 500 purchase orders over 2 years
        product = products.sample(1).iloc[0]
        supplier = suppliers.sample(1).iloc[0]
        
        order_date = start_date + timedelta(days=int(random.randint(0, 730)))
        expected_delivery = order_date + timedelta(days=int(product['lead_time_days']))
        # Actual delivery with some variance
        delivery_variance = int(product['lead_time_days'] * supplier['lead_time_variance'])
        actual_delivery = expected_delivery + timedelta(days=int(random.randint(-delivery_variance, delivery_variance)))
        
        quantity = random.randint(20, 500)
        unit_cost = product['unit_cost']
        
        # Some quality issues
        quality_score = random.uniform(0.9, 1.0)
        
        po_data.append({
            'po_id': f"PO_{po_id}",
            'product_id': product['product_id'],
            'supplier_id': supplier['supplier_id'],
            'order_date': order_date,
            'expected_delivery_date': expected_delivery,
            'actual_delivery_date': actual_delivery,
            'quantity_ordered': quantity,
            'quantity_received': int(quantity * quality_score),
            'unit_cost': unit_cost,
            'total_cost': quantity * unit_cost,
            'delivery_delay_days': (actual_delivery - expected_delivery).days,
            'quality_score': round(quality_score, 3)
        })
        po_id += 1
    
    return pd.DataFrame(po_data)

# Generate all datasets
print("Generating supply chain datasets...")

products_df = generate_products()
suppliers_df = generate_suppliers()
sales_df = generate_sales_data(products_df)
inventory_df = generate_inventory_data(products_df, suppliers_df)
purchase_orders_df = generate_purchase_orders(products_df, suppliers_df)

# Save all datasets
products_df.to_csv('products.csv', index=False)
suppliers_df.to_csv('suppliers.csv', index=False)
sales_df.to_csv('sales_data.csv', index=False)
inventory_df.to_csv('inventory.csv', index=False)
purchase_orders_df.to_csv('purchase_orders.csv', index=False)

print("Dataset generation complete!")
print(f"Products: {len(products_df)} records")
print(f"Suppliers: {len(suppliers_df)} records") 
print(f"Sales transactions: {len(sales_df)} records")
print(f"Inventory items: {len(inventory_df)} records")
print(f"Purchase orders: {len(purchase_orders_df)} records")

# Display sample data
print("\n=== SAMPLE DATA ===")
print("\nProducts Sample:")
print(products_df.head())
print("\nSales Data Sample:")
print(sales_df.head())
print("\nInventory Sample:")
print(inventory_df.head())