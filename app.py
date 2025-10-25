import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="GoldSense AI - Gold Price Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {color: #FF6B35; font-weight: 700;}
    h2 {color: #004E89; font-weight: 600;}
    </style>
    """, unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('gold_processed_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Calculate Metrics
def calculate_metrics(df):
    current_price = df['Price'].iloc[-1]
    prev_price = df['Price'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    week_ago = df['Price'].iloc[-7] if len(df) >= 7 else prev_price
    week_change = current_price - week_ago
    week_change_pct = (week_change / week_ago) * 100
    
    month_ago = df['Price'].iloc[-30] if len(df) >= 30 else prev_price
    month_change = current_price - month_ago
    month_change_pct = (month_change / month_ago) * 100
    
    return {
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'week_change': week_change,
        'week_change_pct': week_change_pct,
        'month_change': month_change,
        'month_change_pct': month_change_pct,
        'year_high': df['Price'].max(),
        'year_low': df['Price'].min()
    }

# Filter by Time Range
def filter_by_time_range(df, time_range):
    if time_range == "ALL":
        return df
    days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095, "5Y": 1825}
    days = days_map.get(time_range, 365)
    end_date = df['Date'].max()
    start_date = end_date - timedelta(days=days)
    return df[df['Date'] >= start_date]

# Create Main Chart
def create_main_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Price'],
        mode='lines', name='Gold Price',
        line=dict(color='gold', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 215, 0, 0.1)'
    ))
    
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA_20'],
            mode='lines', name='MA 20',
            line=dict(color='blue', width=1)
        ))
    
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA_50'],
            mode='lines', name='MA 50',
            line=dict(color='red', width=1)
        ))
    
    fig.update_layout(
        title='Gold Price Movement',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        height=500
    )
    return fig

# Dashboard Page
def show_dashboard(df, metrics):
    st.title("💰 Gold Price Analytics Dashboard")
    st.markdown("### Real-time Gold Market Intelligence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Current Price", f"₹{metrics['current_price']:,.2f}", 
                 f"{metrics['price_change_pct']:.2f}%")
    with col2:
        st.metric("📅 7-Day Change", f"₹{metrics['week_change']:,.2f}", 
                 f"{metrics['week_change_pct']:.2f}%")
    with col3:
        st.metric("📆 30-Day Change", f"₹{metrics['month_change']:,.2f}", 
                 f"{metrics['month_change_pct']:.2f}%")
    with col4:
        volatility = df['Price'].pct_change().std() * np.sqrt(252) * 100
        st.metric("📊 Volatility", f"{volatility:.2f}%")
    
    st.markdown("---")
    st.markdown("### 📈 Price Movement")
    
    time_range = st.select_slider("Select Time Range", 
                                   options=["1M", "3M", "6M", "1Y", "3Y", "5Y", "ALL"], 
                                   value="1Y")
    
    df_filtered = filter_by_time_range(df, time_range)
    fig = create_main_chart(df_filtered)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Price Distribution")
        fig_dist = px.histogram(df_filtered, x='Price', nbins=50,
                               color_discrete_sequence=['gold'])
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Daily Returns")
        returns = df_filtered['Daily_Return'].dropna() * 100
        fig_returns = px.histogram(returns, nbins=50, color_discrete_sequence=['blue'])
        fig_returns.update_layout(showlegend=False)
        st.plotly_chart(fig_returns, use_container_width=True)
    
    st.markdown("### 📋 Market Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**🎯 52-Week High**\n\n₹{metrics['year_high']:,.2f}")
    with col2:
        st.info(f"**📉 52-Week Low**\n\n₹{metrics['year_low']:,.2f}")
    with col3:
        range_val = ((metrics['current_price'] - metrics['year_low']) / 
                    (metrics['year_high'] - metrics['year_low']) * 100)
        st.info(f"**📊 52-Week Range**\n\n{range_val:.1f}% from low")

# Technical Analysis Page
def show_technical_analysis(df):
    st.title("📊 Technical Analysis")
    
    time_range = st.selectbox("Select Time Range", 
                             ["1M", "3M", "6M", "1Y", "3Y", "5Y", "ALL"], 
                             index=3)
    df_filtered = filter_by_time_range(df, time_range)
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Price with Moving Averages', 'Volume', 'RSI', 'MACD'),
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Price'],
                            name='Price', line=dict(color='gold', width=2)),
                 row=1, col=1)
    
    for ma, color in [('MA_20', 'blue'), ('MA_50', 'red'), ('MA_200', 'green')]:
        if ma in df_filtered.columns:
            fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered[ma],
                                    name=ma, line=dict(color=color, width=1)),
                         row=1, col=1)
    
    if 'Volume' in df_filtered.columns:
        fig.add_trace(go.Bar(x=df_filtered['Date'], y=df_filtered['Volume'],
                            name='Volume', marker_color='lightblue'),
                     row=2, col=1)
    
    if 'RSI' in df_filtered.columns:
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['RSI'],
                                name='RSI', line=dict(color='purple', width=1.5)),
                     row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    if all(col in df_filtered.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['MACD'],
                                name='MACD', line=dict(color='blue', width=1)),
                     row=4, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['MACD_Signal'],
                                name='Signal', line=dict(color='red', width=1)),
                     row=4, col=1)
        colors = ['green' if val >= 0 else 'red' for val in df_filtered['MACD_Histogram']]
        fig.add_trace(go.Bar(x=df_filtered['Date'], y=df_filtered['MACD_Histogram'],
                            name='Histogram', marker_color=colors),
                     row=4, col=1)
    
    fig.update_layout(height=1000, showlegend=True, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# Forecasting Page
def show_forecasting(df):
    st.title("🔮 Gold Price Forecasting")
    st.info("⚠️ Simplified demo. Use ARIMA/Prophet/LSTM for production.")
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
    with col2:
        model_type = st.selectbox("Model", ["Moving Average", "Linear Trend", "Exponential Smoothing"])
    
    if st.button("🚀 Generate Forecast"):
        last_date = df['Date'].max()
        last_price = df['Price'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                    periods=forecast_days, freq='D')
        
        if model_type == "Moving Average":
            avg_change = df['Price'].tail(30).pct_change().mean()
            forecast = [last_price]
            for _ in range(forecast_days-1):
                forecast.append(forecast[-1] * (1 + avg_change))
        elif model_type == "Linear Trend":
            recent = df.tail(90)
            x = np.arange(len(recent))
            y = recent['Price'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            forecast = [p(len(recent) + i) for i in range(forecast_days)]
        else:
            alpha = 0.3
            forecast = [last_price]
            recent_avg = df['Price'].tail(30).mean()
            for _ in range(forecast_days-1):
                forecast.append(alpha * forecast[-1] + (1 - alpha) * recent_avg)
        
        std = df['Price'].tail(30).std()
        upper = [f + 1.96 * std for f in forecast]
        lower = [f - 1.96 * std for f in forecast]
        
        fig = go.Figure()
        historical = df.tail(90)
        fig.add_trace(go.Scatter(x=historical['Date'], y=historical['Price'],
                                name='Historical', line=dict(color='gold', width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast,
                                name='Forecast', line=dict(color='blue', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=future_dates, y=upper, name='Upper',
                                line=dict(color='lightblue', width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_dates, y=lower, name='Lower',
                                fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)',
                                line=dict(color='lightblue', width=0)))
        
        fig.update_layout(title=f"{forecast_days}-Day Forecast", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            end_price = forecast[-1]
            change = ((end_price - last_price) / last_price) * 100
            st.metric(f"Price in {forecast_days} days", f"₹{end_price:,.2f}", f"{change:+.2f}%")
        with col2:
            st.metric("Average Forecast", f"₹{np.mean(forecast):,.2f}")
        with col3:
            st.metric("Predicted High", f"₹{max(forecast):,.2f}")

# Statistics Page
def show_statistics(df):
    st.title("📈 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Price Statistics")
        stats = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
            'Value': [
                f"₹{df['Price'].mean():,.2f}",
                f"₹{df['Price'].median():,.2f}",
                f"₹{df['Price'].std():,.2f}",
                f"₹{df['Price'].min():,.2f}",
                f"₹{df['Price'].max():,.2f}",
                f"₹{df['Price'].max() - df['Price'].min():,.2f}"
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Returns Statistics")
        returns = df['Daily_Return'].dropna() * 100
        returns_stats = pd.DataFrame({
            'Metric': ['Mean Return', 'Std Dev', 'Sharpe Ratio', 'Best Day', 'Worst Day'],
            'Value': [
                f"{returns.mean():.4f}%",
                f"{returns.std():.4f}%",
                f"{returns.mean()/returns.std():.4f}",
                f"{returns.max():.2f}%",
                f"{returns.min():.2f}%"
            ]
        })
        st.dataframe(returns_stats, use_container_width=True, hide_index=True)
    
    st.markdown("### 📅 Monthly Performance")
    df_monthly = df.copy()
    df_monthly['Year'] = df_monthly['Date'].dt.year
    df_monthly['Month'] = df_monthly['Date'].dt.month
    monthly_avg = df_monthly.groupby(['Year', 'Month'])['Price'].mean().reset_index()
    monthly_avg['Month_Name'] = pd.to_datetime(monthly_avg['Month'], format='%m').dt.month_name()
    
    pivot = monthly_avg.pivot(index='Month_Name', columns='Year', values='Price')
    fig = px.imshow(pivot, labels=dict(x="Year", y="Month", color="Price"),
                    color_continuous_scale='RdYlGn', aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Trading Signals Page
def show_trading_signals(df):
    st.title("💹 Trading Signals")
    st.info("📊 Educational purposes only!")
    
    current = df['Price'].iloc[-1]
    ma20 = df['MA_20'].iloc[-1] if 'MA_20' in df.columns else current
    ma50 = df['MA_50'].iloc[-1] if 'MA_50' in df.columns else current
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
    
    score = 0
    if current > ma20: score += 1
    if current > ma50: score += 1
    if ma20 > ma50: score += 1
    if 30 < rsi < 70: score += 1
    elif rsi < 30: score += 2
    if macd > 0: score += 1
    
    if score >= 5:
        signal, color = "🟢 STRONG BUY", "green"
    elif score >= 3:
        signal, color = "🟡 BUY", "blue"
    elif score >= 2:
        signal, color = "🟠 HOLD", "orange"
    else:
        signal, color = "🔴 SELL", "red"
    
    st.markdown(f"### Overall: <span style='color:{color}; font-size:32px; font-weight:bold'>{signal}</span>", 
               unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📈 Trend Signals")
        st.write(f"**Price vs MA20:** {'🟢 Bullish' if current > ma20 else '🔴 Bearish'}")
        st.write(f"**Price vs MA50:** {'🟢 Bullish' if current > ma50 else '🔴 Bearish'}")
    with col2:
        st.markdown("#### 🎯 Momentum")
        rsi_sig = "🟢 Oversold" if rsi < 30 else "🔴 Overbought" if rsi > 70 else "🟡 Neutral"
        st.write(f"**RSI ({rsi:.2f}):** {rsi_sig}")
        st.write(f"**MACD:** {'🟢 Bullish' if macd > 0 else '🔴 Bearish'}")
    with col3:
        st.markdown("#### 📊 Support/Resistance")
        high = df['Price'].tail(30).max()
        low = df['Price'].tail(30).min()
        st.write(f"**30D High:** ₹{high:,.2f}")
        st.write(f"**30D Low:** ₹{low:,.2f}")

# Chatbot Page
def show_chatbot(df):
    st.title("🤖 AI Gold Price Assistant")
    st.info("💡 Ask me about gold prices!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Current Price?"):
            st.session_state.messages.append({"role": "user", "content": "Current price?"})
    with col2:
        if st.button("📈 Trend?"):
            st.session_state.messages.append({"role": "user", "content": "What's the trend?"})
    with col3:
        if st.button("🎯 Buy?"):
            st.session_state.messages.append({"role": "user", "content": "Should I buy?"})
    
    if prompt := st.chat_input("Ask me..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        current = df['Price'].iloc[-1]
        prev = df['Price'].iloc[-2]
        change = ((current - prev) / prev) * 100
        
        q = prompt.lower()
        if 'price' in q:
            response = f"💰 Current: **₹{current:,.2f}** (Change: {change:+.2f}%)"
        elif 'trend' in q:
            response = f"📊 {'📈 Upward' if change > 0 else '📉 Downward'} trend. Change: {change:+.2f}%"
        elif 'buy' in q:
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            response = f"{'🟢 BUY' if rsi < 40 else '🔴 WAIT' if rsi > 60 else '🟡 HOLD'} (RSI: {rsi:.2f})"
        else:
            response = "🤖 Ask about: price, trend, or buy/sell signals!"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Main App
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2916/2916687.png", width=100)
    st.sidebar.title("🏆 GoldSense AI")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox("📍 Navigate",
        ["🏠 Dashboard", "📊 Technical Analysis", "🔮 Forecasting",
         "📈 Statistics", "💹 Trading Signals", "🤖 AI Chatbot"])
    
    df = load_data()
    
    if df is None:
        st.error("⚠️ Upload 'gold_processed_data.csv'")
        return
    
    metrics = calculate_metrics(df)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Records:** {len(df):,}\n**Range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    if page == "🏠 Dashboard":
        show_dashboard(df, metrics)
    elif page == "📊 Technical Analysis":
        show_technical_analysis(df)
    elif page == "🔮 Forecasting":
        show_forecasting(df)
    elif page == "📈 Statistics":
        show_statistics(df)
    elif page == "💹 Trading Signals":
        show_trading_signals(df)
    elif page == "🤖 AI Chatbot":
        show_chatbot(df)

if __name__ == "__main__":
    main()
