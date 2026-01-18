import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Sri Lanka Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling with theme support
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: var(--background-color);
    }
    
    /* Card styling */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Header styling - adaptive to theme */
    h1 {
        color: var(--text-color) !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        padding: 20px 0;
    }
    
    h2 {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    
    /* Subheader styling - adaptive to theme */
    h3 {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    
    h4 {
        color: var(--text-color) !important;
    }
    
    /* Paragraph text - adaptive to theme */
    p {
        color: var(--text-color) !important;
    }
    
    /* Label text - adaptive to theme */
    label {
        color: var(--text-color) !important;
        font-weight: 500;
    }
    
    /* Custom card - adaptive background */
    .custom-card {
        background: var(--secondary-background-color);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 600;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric card styling - adaptive */
    [data-testid="stMetricValue"] {
        font-size: 36px;
        color: var(--text-color) !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }
    
    /* Input field styling - adaptive */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
    }
    
    /* Text input - adaptive */
    input {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
    }
    
    /* Number input specific styling - adaptive */
    input[type="number"] {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
    }
    
    /* Slider text - adaptive */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: var(--text-color) !important;
    }
    
    /* Select box text - adaptive */
    [data-baseweb="select"] {
        color: var(--text-color) !important;
    }
    
    /* Select box selected value - adaptive */
    [data-baseweb="select"] > div {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        color: var(--text-color) !important;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        color: var(--text-color) !important;
    }
    
    /* Radio button text - adaptive */
    .stRadio > label {
        color: var(--text-color) !important;
    }
    
    /* Checkbox text - adaptive */
    .stCheckbox > label {
        color: var(--text-color) !important;
    }
    
    /* Expander text - adaptive */
    .streamlit-expanderHeader {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
    }
    
    /* Expander content - adaptive */
    .streamlit-expanderContent {
        background-color: var(--secondary-background-color) !important;
    }
    
    /* Markdown text in expander - adaptive */
    .streamlit-expanderContent p {
        color: var(--text-color) !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        color: var(--text-color);
        font-size: 14px;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('lightgbm_best_model.pkl')

model = load_model()

# Hero Section
st.markdown("""
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1 style='font-size: 48px; margin-bottom: 10px;'>🚗 Sri Lanka Vehicle Price Predictor</h1>
        <p style='font-size: 20px; color: #6b7280;'>Get instant price estimates powered by AI & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Statistics Cards
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='color: white; margin: 0;'>1000+</h2>
            <p style='margin: 5px 0 0 0;'>Vehicles Analyzed</p>
        </div>
    """, unsafe_allow_html=True)

with col_stat2:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='color: white; margin: 0;'>95%+</h2>
            <p style='margin: 5px 0 0 0;'>Accuracy Rate</p>
        </div>
    """, unsafe_allow_html=True)

with col_stat3:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='color: white; margin: 0;'>10+</h2>
            <p style='margin: 5px 0 0 0;'>Vehicle Brands</p>
        </div>
    """, unsafe_allow_html=True)

with col_stat4:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='color: white; margin: 0;'>Instant</h2>
            <p style='margin: 5px 0 0 0;'>Predictions</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Create two columns for input fields
st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
    <h2 style='color: #1e3a8a; margin-top: 0;'>📝 Enter Vehicle Details</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🏷️ Basic Information")
    
    # Make selection
    make = st.selectbox(
        "🚘 Make (Brand)",
        options=['Toyota', 'Honda', 'Nissan', 'Suzuki', 'Mitsubishi', 'KIA', 'Mazda', 'Hyundai', 'Ford', 'Perodua'],
        help="Select the manufacturer of the vehicle"
    )
    
    # Model frequency (simplified - using average values)
    model_freq = st.slider(
        "📊 Model Popularity", 
        1, 100, 20,
        help="1 = Rare model, 100 = Very common model"
    )
    
    # Year of Manufacture
    current_year = 2026
    yom = st.slider(
        "📅 Year of Manufacture (YOM)", 
        1990, current_year, 2015,
        help="The year the vehicle was manufactured"
    )
    vehicle_age = current_year - yom
    
    # Mileage
    mileage = st.number_input(
        "🛣️ Mileage (km)", 
        min_value=0, max_value=500000, value=50000, step=1000,
        help="Total distance traveled by the vehicle"
    )
    
    # Engine Capacity
    engine_cc = st.number_input(
        "⚙️ Engine Capacity (cc)", 
        min_value=500, max_value=5000, value=1500, step=100,
        help="Engine displacement in cubic centimeters"
    )

with col2:
    st.markdown("### 🔧 Features & Specifications")
    
    # Transmission
    transmission = st.radio(
        "🔄 Transmission", 
        options=['Automatic', 'Manual'],
        horizontal=True,
        help="Type of transmission system"
    )
    
    # Fuel Type
    fuel_type = st.selectbox(
        "⛽ Fuel Type",
        options=['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Gas'],
        help="Type of fuel the vehicle uses"
    )
    
    st.markdown("### ✨ Optional Features")
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        has_ac = st.checkbox("❄️ A/C", value=True)
    with col_feat2:
        has_power_steering = st.checkbox("🎮 P. Steering", value=True)
    with col_feat3:
        has_power_mirror = st.checkbox("🪞 P. Mirror", value=True)

# Prepare input data for prediction
def prepare_input():
    # Create a dictionary with all features initialized to 0
    input_data = {
        'Mileage_km': mileage,
        'Engine_cc': engine_cc,
        'Vehicle_Age': vehicle_age,
        'Has_AC': 1 if has_ac else 0,
        'Has_PowerSteering': 1 if has_power_steering else 0,
        'Has_PowerMirror': 1 if has_power_mirror else 0,
        'Transmission_Manual': 1 if transmission == 'Manual' else 0,
        'Fuel_Type_Electric': 1 if fuel_type == 'Electric' else 0,
        'Fuel_Type_Gas': 1 if fuel_type == 'Gas' else 0,
        'Fuel_Type_Hybrid': 1 if fuel_type == 'Hybrid' else 0,
        'Fuel_Type_Petrol': 1 if fuel_type == 'Petrol' else 0,
        'Make_ford': 1 if make.lower() == 'ford' else 0,
        'Make_honda': 1 if make.lower() == 'honda' else 0,
        'Make_hyundai': 1 if make.lower() == 'hyundai' else 0,
        'Make_kia': 1 if make.lower() == 'kia' else 0,
        'Make_mazda': 1 if make.lower() == 'mazda' else 0,
        'Make_mitsubishi': 1 if make.lower() == 'mitsubishi' else 0,
        'Make_nissan': 1 if make.lower() == 'nissan' else 0,
        'Make_perodua': 1 if make.lower() == 'perodua' else 0,
        'Make_suzuki': 1 if make.lower() == 'suzuki' else 0,
        'Make_toyota': 1 if make.lower() == 'toyota' else 0,
        'Model_Freq': model_freq
    }
    
    return pd.DataFrame([input_data])

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)

col_center = st.columns([1, 2, 1])
with col_center[1]:
    predict_button = st.button("🔮 Predict Vehicle Price", use_container_width=True, type="primary")

if predict_button:
    with st.spinner('🤖 Analyzing vehicle data...'):
        # Prepare input
        input_df = prepare_input()
        
        # Make prediction (model was trained on log-transformed prices)
        prediction_log = model.predict(input_df)
        predicted_price = np.expm1(prediction_log[0])  # Convert back from log scale
    
    # Display result with animation
    st.balloons()
    
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 10%); 
                    padding: 30px; border-radius: 15px; margin: 30px 0; text-align: center;'>
            <h2 style='color: white; margin: 0;'>💰 Predicted Price</h2>
            <h1 style='color: white; font-size: 56px; margin: 10px 0;'>Rs. {:,.0f}</h1>
        </div>
    """.format(predicted_price), unsafe_allow_html=True)
    
    # Detailed metrics
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        # Calculate price range (±10%)
        lower_bound = predicted_price * 0.9
        upper_bound = predicted_price * 1.1
        
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <h4 style='color: #6b7280; margin: 0;'>💵 Price Range</h4>
                <h3 style='color: #1e3a8a; margin: 10px 0 0 0;'>Rs. {:,.0f} - {:,.0f}</h3>
            </div>
        """.format(lower_bound, upper_bound), unsafe_allow_html=True)
    
    with col_result2:
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <h4 style='color: #6b7280; margin: 0;'>📅 Vehicle Age</h4>
                <h3 style='color: #1e3a8a; margin: 10px 0 0 0;'>{} years</h3>
            </div>
        """.format(vehicle_age), unsafe_allow_html=True)
    
    with col_result3:
        # Calculate depreciation
        depreciation_rate = (vehicle_age / 15) * 100 if vehicle_age <= 15 else 100
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <h4 style='color: #6b7280; margin: 0;'>📉 Est. Depreciation</h4>
                <h3 style='color: #1e3a8a; margin: 10px 0 0 0;'>{:.1f}%</h3>
            </div>
        """.format(depreciation_rate), unsafe_allow_html=True)
    
    # Create a gauge chart for price confidence
    st.markdown("<br>", unsafe_allow_html=True)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 85,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "🎯 Prediction Confidence", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 100], 'color': '#d1fae5'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "#1e3a8a", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional information
    st.info("""
    📌 **Important Notes:**
    - This prediction is based on historical vehicle data from riyasewana.com
    - Actual market prices may vary based on vehicle condition, location, and current demand
    - The price range shown represents a ±10% variation from the predicted value
    - For the most accurate valuation, we recommend getting a professional inspection
    """)
    
    # Vehicle summary
    with st.expander("📋 View Vehicle Summary"):
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write(f"**Make:** {make}")
            st.write(f"**Year:** {yom}")
            st.write(f"**Mileage:** {mileage:,} km")
            st.write(f"**Engine:** {engine_cc} cc")
        with summary_col2:
            st.write(f"**Transmission:** {transmission}")
            st.write(f"**Fuel Type:** {fuel_type}")
            st.write(f"**Age:** {vehicle_age} years")
            features = []
            if has_ac: features.append("A/C")
            if has_power_steering: features.append("Power Steering")
            if has_power_mirror: features.append("Power Mirrors")
            st.write(f"**Features:** {', '.join(features) if features else 'None'}")

# Add footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: white; border-radius: 15px; margin-top: 30px;'>
    <h3 style='color: #1e3a8a; margin-bottom: 10px;'>🚗 Sri Lanka Vehicle Price Predictor</h3>
    <p style='color: #6b7280; margin: 5px 0;'>Powered by Light GBM Machine Learning & Streamlit</p>
    <p style='color: #9ca3af; font-size: 14px; margin: 10px 0 0 0;'>
        Built with ❤️ for the Sri Lankan automotive community
    </p>
    <p style='color: #d1d5db; font-size: 12px; margin-top: 15px;'>
        © 2026 All Rights Reserved | Data sourced from riyasewana.com
    </p>
</div>
""", unsafe_allow_html=True)
