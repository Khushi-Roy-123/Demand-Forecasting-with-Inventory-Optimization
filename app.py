import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# --- Page Config ---
st.set_page_config(
    page_title="Favorita Sales Intelligence", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Dark Look ---
st.markdown("""
<style>
    /* Global Dark Theme Overrides */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metrics Cards */
    div.css-1r6slb0, div.css-12w0qpk {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card {
        background-color: #1f2937; /* Dark Gray */
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #f3f4f6;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f2937;
        border-radius: 8px;
        color: #d1d5db;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_data():
    data_path = "train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}")
        return None
    try:
        # Load larger subset for better analytics
        df = pd.read_csv(data_path, nrows=200000, parse_dates=['date'])
        
        required_cols = {'date', 'store_nbr', 'item_nbr', 'unit_sales'}
        if not required_cols.issubset(df.columns):
            st.error(f"Dataset missing required columns: {required_cols - set(df.columns)}")
            return None
        
        # Ensure onpromotion boolean
        if 'onpromotion' in df.columns:
            df['onpromotion'] = df['onpromotion'].fillna(False).astype(bool)
        else:
            df['onpromotion'] = False
            
        return df
    except Exception as e:
        st.error(f"Error loading train.csv: {e}")
        return None

# --- Main App ---
st.title("Favorita Sales Intelligence")
st.markdown("### Demand Forecasting & Analytics Dashboard")

try:
    with st.spinner('Loading and processing data...'):
        df = load_data()
except Exception as e:
    st.error(f"Critical error: {e}")
    df = None

if df is not None:
    # --- Sidebar ---
    st.sidebar.markdown("## Control Panel")
    
    # Store Selection
    store_ids = sorted(df['store_nbr'].unique())
    selected_store = st.sidebar.selectbox("Select Store", store_ids)
    
    # Item Selection
    store_items = df[df['store_nbr'] == selected_store]['item_nbr'].unique()
    if len(store_items) > 0:
        item_ids = sorted(store_items)
        selected_item = st.sidebar.selectbox("Select Product", item_ids[:100])
    else:
        st.sidebar.warning("No items found.")
        selected_item = None

    # Date Selection
    min_date, max_date = df['date'].min(), df['date'].max()
    date_range = st.sidebar.date_input("Analysis Period", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    # Advanced Options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### View Options")
    show_trend = st.sidebar.checkbox("Show Trendline (OLS)", value=True)
    show_moving_avg = st.sidebar.checkbox("Show 7-Day Moving Avg", value=True)

    if selected_store and selected_item and len(date_range) == 2:
        # Filter Data
        mask = (df['store_nbr'] == selected_store) & \
               (df['item_nbr'] == selected_item) & \
               (df['date'] >= pd.to_datetime(date_range[0])) & \
               (df['date'] <= pd.to_datetime(date_range[1]))
        filtered_df = df[mask].sort_values('date')

        if not filtered_df.empty:
            filtered_df['DayOfWeek'] = filtered_df['date'].dt.day_name()
            filtered_df['Month'] = filtered_df['date'].dt.month_name()
            
            # --- Key Metrics Row ---
            col1, col2, col3, col4 = st.columns(4)
            
            total_sales = filtered_df['unit_sales'].sum()
            avg_sales = filtered_df['unit_sales'].mean()
            peak_day = filtered_df.loc[filtered_df['unit_sales'].idxmax(), 'date'].strftime('%d %b %Y')
            promo_days = filtered_df['onpromotion'].sum()
            
            col1.markdown(f'<div class="metric-card"><div class="metric-label">Total Volume</div><div class="metric-value">{total_sales:,.0f}</div></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-card"><div class="metric-label">Daily Avg</div><div class="metric-value">{avg_sales:,.1f}</div></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-card"><div class="metric-label">Peak Sales</div><div style="font-size: 20px; font-weight: bold; color: #f3f4f6; margin-top: 10px;">{peak_day}</div></div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="metric-card"><div class="metric-label">Promo Days</div><div class="metric-value" style="color: #10b981;">{promo_days}</div></div>', unsafe_allow_html=True)

            # --- Main Charts Area ---
            tab1, tab2, tab3 = st.tabs(["Sales Performance", "Deep Dive", "ML Forecast"])

            with tab1:
                col_chart, col_stat = st.columns([3, 1])
                with col_chart:
                    # Time Series Chart
                    fig = px.line(filtered_df, x='date', y='unit_sales', 
                                  title="Daily Sales Trajectory",
                                  labels={'unit_sales': 'Units Sold', 'date': 'Date'},
                                  template="plotly_dark")
                    
                    fig.update_traces(line=dict(color='#8b5cf6', width=3))
                    fig.update_layout(height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    
                    if show_trend:
                        fig.add_traces(px.scatter(filtered_df, x="date", y="unit_sales", trendline="ols").data[1])
                    if show_moving_avg:
                        filtered_df['MA7'] = filtered_df['unit_sales'].rolling(7).mean()
                        fig.add_scatter(x=filtered_df['date'], y=filtered_df['MA7'], mode='lines', name='7-Day MA', line=dict(color='#10b981', dash='dash'))

                    st.plotly_chart(fig, use_container_width=True)

                with col_stat:
                    st.markdown("##### Quick Insights")
                    st.write("Calculated statistics based on the selected period.")
                    try:
                        max_sales = filtered_df['unit_sales'].max()
                        min_sales = filtered_df[filtered_df['unit_sales']>0]['unit_sales'].min()
                        volatility = filtered_df['unit_sales'].std()
                        st.dataframe(pd.DataFrame({
                            'Metric': ['Max Sales', 'Min (Non-Zero)', 'Volatility (Std)'],
                            'Value': [f"{max_sales:,.0f}", f"{min_sales:,.0f}", f"{volatility:,.2f}"]
                        }), hide_index=True)
                    except:
                        st.info("Insufficient data for stats.")

            with tab2:
                col_d, col_p = st.columns(2)
                
                with col_d:
                    st.markdown("#### Seasonality: Sales by Day of Week")
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    daily_avg = filtered_df.groupby('DayOfWeek')['unit_sales'].mean().reindex(day_order)
                    
                    fig_day = px.bar(x=daily_avg.index, y=daily_avg.values, 
                                     labels={'x': 'Day', 'y': 'Avg Sales'},
                                     template="plotly_dark", color_discrete_sequence=['#3b82f6'])
                    fig_day.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_day, use_container_width=True)
                
                with col_p:
                    st.markdown("#### Impact of Promotions")
                    promo_avg = filtered_df.groupby('onpromotion')['unit_sales'].mean()
                    promo_df = pd.DataFrame({'Status': ['Off Promo', 'On Promo'], 
                                           'Avg Sales': [promo_avg.get(False, 0), promo_avg.get(True, 0)]})
                    
                    fig_promo = px.bar(promo_df, x='Status', y='Avg Sales', 
                                       color='Status',
                                       color_discrete_map={'Off Promo': '#6b7280', 'On Promo': '#ec4899'},
                                       template="plotly_dark")
                    fig_promo.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_promo, use_container_width=True)

            with tab3:
                model_path = "model.pkl"
                if os.path.exists(model_path):
                    col_m1, col_m2 = st.columns([1, 2])
                    with col_m1:
                        st.success("Model Trained")
                        st.markdown("""
                        **Model Specs:**
                        - Type: LightGBM Regressor
                        - Features: Lags, Rolling Stats, Date Parts
                        """)
                    with col_m2:
                        st.info("Prediction Module")
                        st.write("To generate forecasts for specific future dates, ensure `test.csv` is fully processed with the same feature engineering pipeline.")
                else:
                    st.warning("Model file not found. Please run the training pipeline first.")

        else:
            st.info("No data available for the selected range.")
    else:
        st.info("Please select filters from the sidebar to start analysis.")
