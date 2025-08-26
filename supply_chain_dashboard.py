
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Supply Chain Analytics Dashboard", layout="wide")
st.title("Supply Chain Analytics Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", [
    "Data Overview", "Inventory Analysis", "Demand Forecasting", "Supplier Performance", "Cost Optimization", "Key Insights"
])

# Load data
@st.cache_data
def load_data():
    products = pd.read_csv('products.csv')
    suppliers = pd.read_csv('suppliers.csv')
    sales = pd.read_csv('sales_data.csv')
    inventory = pd.read_csv('inventory.csv')
    purchase_orders = pd.read_csv('purchase_orders.csv')
    return products, suppliers, sales, inventory, purchase_orders

products, suppliers, sales, inventory, purchase_orders = load_data()

# Data Overview
if section == "Data Overview":
    st.header("Data Overview")
    st.subheader("Products")
    st.dataframe(products)
    st.subheader("Suppliers")
    st.dataframe(suppliers)
    st.subheader("Sales (Sample)")
    st.dataframe(sales.head(1000))
    st.subheader("Inventory")
    st.dataframe(inventory)
    st.subheader("Purchase Orders")
    st.dataframe(purchase_orders)

# Inventory Analysis
elif section == "Inventory Analysis":
    st.header("Inventory Performance Analysis")
    inventory_analysis = inventory.merge(products[['product_id', 'category', 'unit_cost', 'selling_price']], on='product_id')
    inventory_analysis['inventory_value'] = inventory_analysis['current_stock'] * inventory_analysis['unit_cost']
    inventory_analysis['annual_holding_cost'] = inventory_analysis['inventory_value'] * 0.25
    inventory_analysis['turnover_ratio'] = inventory_analysis['avg_daily_demand'] * 365 / (inventory_analysis['current_stock'] + 1)
    inventory_analysis['stockout_risk'] = np.where(inventory_analysis['current_stock'] <= inventory_analysis['safety_stock'], 'High Risk', 'Safe')
    st.dataframe(inventory_analysis)
    st.subheader("Inventory Value by Category")
    st.bar_chart(inventory_analysis.groupby('category')['inventory_value'].sum())
    st.subheader("Stockout Risk Distribution")
    st.bar_chart(inventory_analysis['stockout_risk'].value_counts())

# Demand Forecasting
elif section == "Demand Forecasting":
    st.header("Demand Forecasting Analysis")
    sales['date'] = pd.to_datetime(sales['date'])
    daily_sales = sales.groupby('date').agg({'quantity_sold': 'sum', 'revenue': 'sum'}).reset_index()
    daily_sales['ma_7'] = daily_sales['quantity_sold'].rolling(window=7).mean()
    daily_sales['ma_30'] = daily_sales['quantity_sold'].rolling(window=30).mean()
    st.line_chart(daily_sales.set_index('date')[['quantity_sold', 'ma_7', 'ma_30']])
    st.subheader("Monthly Sales Patterns")
    monthly = daily_sales.copy()
    monthly['month'] = monthly['date'].dt.month
    st.bar_chart(monthly.groupby('month')['quantity_sold'].mean())
    st.subheader("Top 10 Products by Revenue")
    top_products = sales.groupby('product_id')['revenue'].sum().nlargest(10)
    st.bar_chart(top_products)

# Supplier Performance
elif section == "Supplier Performance":
    st.header("Supplier Performance Analysis")
    purchase_orders['order_date'] = pd.to_datetime(purchase_orders['order_date'])
    purchase_orders['expected_delivery_date'] = pd.to_datetime(purchase_orders['expected_delivery_date'])
    purchase_orders['actual_delivery_date'] = pd.to_datetime(purchase_orders['actual_delivery_date'])
    supplier_perf = purchase_orders.groupby('supplier_id').agg({
        'delivery_delay_days': ['mean', 'std', 'count'],
        'quality_score': 'mean',
        'total_cost': 'sum',
        'quantity_received': 'sum'
    })
    supplier_perf.columns = ['Avg_Delay', 'Delay_Std', 'Order_Count', 'Avg_Quality', 'Total_Spend', 'Total_Quantity']
    supplier_perf = supplier_perf.merge(suppliers.set_index('supplier_id'), left_index=True, right_index=True)
    st.dataframe(supplier_perf)
    st.subheader("Top 5 Suppliers by Performance Score")
    supplier_perf['Performance_Score'] = (np.where(supplier_perf['Avg_Delay'] <= 0, 1, 1 - (supplier_perf['Avg_Delay'] / 10))) * 0.4 + supplier_perf['Avg_Quality'] * 0.6
    st.dataframe(supplier_perf.sort_values('Performance_Score', ascending=False).head(5))

# Cost Optimization
elif section == "Cost Optimization":
    st.header("Cost Optimization Opportunities")
    optimization = inventory.merge(products[['product_id', 'category', 'unit_cost']], on='product_id')
    optimization['inventory_value'] = optimization['current_stock'] * optimization['unit_cost']
    optimization['annual_holding_cost'] = optimization['inventory_value'] * 0.25
    optimization['days_of_stock'] = optimization['current_stock'] / (optimization['avg_daily_demand'] + 0.01)
    def categorize_stock_status(days_of_stock):
        if days_of_stock > 90:
            return 'Overstock'
        elif days_of_stock > 30:
            return 'Normal'
        elif days_of_stock > 7:
            return 'Low Stock'
        else:
            return 'Critical'
    optimization['stock_status'] = optimization['days_of_stock'].apply(categorize_stock_status)
    st.bar_chart(optimization['stock_status'].value_counts())
    st.dataframe(optimization)

# Key Insights
elif section == "Key Insights":
    st.header("Key Insights and Recommendations")
    inventory_analysis = inventory.merge(products[['product_id', 'category', 'unit_cost', 'selling_price']], on='product_id')
    inventory_analysis['inventory_value'] = inventory_analysis['current_stock'] * inventory_analysis['unit_cost']
    inventory_analysis['annual_holding_cost'] = inventory_analysis['inventory_value'] * 0.25
    total_inventory_value = inventory_analysis['inventory_value'].sum()
    high_risk_products = len(inventory_analysis[inventory_analysis['stockout_risk'] == 'High Risk']) if 'stockout_risk' in inventory_analysis else 0
    avg_delivery_delay = purchase_orders['delivery_delay_days'].mean()
    st.markdown(f"**Total Inventory Investment:** ${total_inventory_value:,.2f}")
    st.markdown(f"**Annual Holding Costs:** ${inventory_analysis['annual_holding_cost'].sum():,.2f}")
    st.markdown(f"**Products at Stockout Risk:** {high_risk_products}")
    st.markdown(f"**Average Supplier Delivery Delay:** {avg_delivery_delay:.1f} days")
    st.markdown("**Recommendations:**")
    st.markdown("- Focus on Class A products for tighter inventory control\n- Reduce overstock on slow-moving items\n- Improve supplier relationships with poor delivery performance\n- Implement dynamic safety stock levels\n- Consider vendor-managed inventory for critical suppliers")
