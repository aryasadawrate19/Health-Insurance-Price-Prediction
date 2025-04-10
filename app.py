import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and metrics
model = joblib.load('insurance_model.pkl')

# Metrics
MAE = 1983.70
MSE = 19060953.44
R2 = 0.88

# Set page config with a favicon
st.set_page_config(
    page_title="Insurance Charge Predictor",
    page_icon="💉",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #3B82F6;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #ECFDF5;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 20px;
        text-align: center;
    }
    .risk-high {
        color: #B91C1C;
        font-weight: bold;
    }
    .risk-medium {
        color: #D97706;
        font-weight: bold;
    }
    .risk-low {
        color: #047857;
        font-weight: bold;
    }
    /* Added styles for number inputs */
    .stNumberInput {
        width: 100%;
    }
    .slider-text-pair {
        display: flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">💸 Medical Insurance Charge Predictor</p>', unsafe_allow_html=True)

# Introduction
with st.container():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <style>
            .info-box {
                color: black;
                font-size: 16px;
                background-color: #f2f2f2;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ccc;
            }
        </style>

        <div class="info-box">
            This application uses machine learning to predict annual medical insurance charges based on customer details.
            Fill in the form to receive a personalized cost estimation and risk assessment.
        </div>
        """, unsafe_allow_html=True)

# BMI Calculator outside the form
with st.expander("📏 Calculate BMI"):
    weight_height_cols = st.columns(2)
    with weight_height_cols[0]:
        weight = st.number_input("Weight (kg)", min_value=20, max_value=250, value=70)
    with weight_height_cols[1]:
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    
    if st.button("Calculate BMI"):
        calculated_bmi = weight / ((height/100) ** 2)
        st.write(f"Calculated BMI: **{calculated_bmi:.1f}**")
        
        # BMI category
        bmi_category = ""
        if calculated_bmi < 18.5:
            bmi_category = "Underweight"
        elif calculated_bmi < 25:
            bmi_category = "Normal weight"
        elif calculated_bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        
        st.write(f"Category: **{bmi_category}**")
        st.session_state['calculated_bmi'] = calculated_bmi

# Create two columns for the layout
col1, col2 = st.columns([1, 1.5])

# Input form in the left column
with col1:
    st.markdown('<p class="subheader">Customer Details</p>', unsafe_allow_html=True)
    
    with st.form("input_form"):
        # Personal information section
        st.markdown("##### Personal Information")
        
        # Age with both slider and text input
        st.write("Age")
        age_col1, age_col2 = st.columns([3, 1])
        with age_col1:
            age = st.slider("Age slider", 18, 100, 30, label_visibility="collapsed", 
                          help="Age of the customer in years")
        with age_col2:
            age_input = st.number_input("Age input", min_value=18, max_value=100, value=age, 
                                      label_visibility="collapsed")
            # Sync slider and text input
            if age_input != age:
                age = age_input
        
        sex = st.selectbox("Sex", ["male", "female"])
        
        # BMI with both slider and text input
        st.write("BMI (Body Mass Index)")
        bmi_col1, bmi_col2 = st.columns([3, 1])
        with bmi_col1:
            # Check if we have a calculated BMI from the calculator
            initial_bmi = st.session_state.get('calculated_bmi', 25.0)
            if initial_bmi < 10.0:
                initial_bmi = 10.0
            elif initial_bmi > 50.0:
                initial_bmi = 50.0
                
            bmi = st.slider("BMI slider", 10.0, 50.0, float(initial_bmi), 0.1, 
                          label_visibility="collapsed",
                          help="BMI is weight(kg)/height²(m). Normal range: 18.5-24.9")
        with bmi_col2:
            bmi_input = st.number_input("BMI input", min_value=10.0, max_value=50.0, 
                                      value=float(bmi), step=0.1, format="%.1f",
                                      label_visibility="collapsed")
            # Sync slider and text input
            if bmi_input != bmi:
                bmi = bmi_input
        
        # Use calculated BMI checkbox
        if 'calculated_bmi' in st.session_state:
            use_calculated = st.checkbox("Use my calculated BMI", value=False)
            if use_calculated:
                bmi = min(max(st.session_state['calculated_bmi'], 10.0), 50.0)
        
        # Health status
        st.markdown("##### Health Status")
        smoker = st.selectbox("Smoking Status", ["no", "yes"], 
                             help="Smoking significantly increases insurance costs")
        
        # Additional factors
        st.markdown("##### Family & Location")
        
        # Children with both slider and text input
        st.write("Number of Dependents")
        children_col1, children_col2 = st.columns([3, 1])
        with children_col1:
            children = st.slider("Children slider", 0, 5, 0, 
                               label_visibility="collapsed",
                               help="Number of children covered by insurance")
        with children_col2:
            children_input = st.number_input("Children input", min_value=0, max_value=5, 
                                           value=children, label_visibility="collapsed")
            # Sync slider and text input
            if children_input != children:
                children = children_input
        
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"],
                             help="Geographic region in India")
        
        # Form submit button
        submitted = st.form_submit_button("Calculate Premium")

# Results display in right column
with col2:
    st.markdown('<p class="subheader">Premium Estimation</p>', unsafe_allow_html=True)
    
    if submitted:
        # Create engineered features
        bmi_over_30 = int(bmi > 30)
        age_bmi_interaction = age * bmi
        smoker_age_interaction = age if smoker == "yes" else 0
        children_over_2 = int(children > 2)

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
            'bmi_over_30': bmi_over_30,
            'age_bmi_interaction': age_bmi_interaction,
            'smoker_age_interaction': smoker_age_interaction,
            'children_over_2': children_over_2
        }])

        # Make prediction
        pred_log = model.predict(input_data)[0]
        pred = np.expm1(pred_log)
        
        # Define risk categories
        if pred > 25000:
            risk_category = "high"
            risk_class = "risk-high"
        elif pred > 12000:
            risk_category = "medium"
            risk_class = "risk-medium"
        else:
            risk_category = "low"
            risk_class = "risk-low"
        
        # Display prediction
        st.markdown(f"""
            <style>
                .prediction-box {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: black;
                    font-family: 'Segoe UI', sans-serif;
                }}
                .prediction-box h3 {{
                    margin-bottom: 10px;
                    font-size: 22px;
                }}
                .prediction-box h2 {{
                    margin-top: 0;
                    font-size: 32px;
                    color: black;
                }}
                .prediction-box span {{
                    font-weight: bold;
                }}
            </style>

            <div class="prediction-box">
                <h3>Estimated Annual Premium</h3>
                <h2>₹{pred:,.2f}</h2>
                <p>Risk Category: <span class="{risk_class}">{risk_category.upper()}</span></p>
            </div>
        """, unsafe_allow_html=True)

        
        # Key factors affecting the prediction
        st.subheader("Key Factors Influencing Your Premium")
        
        # Create a basic factor impact visualization
        factors = ['Age', 'BMI', 'Smoking', 'Children', 'Region']
        
        # Simplified impact calculation (would be better with SHAP values)
        impacts = [
            age / 30 * 0.3,  # Age impact
            (bmi - 20) / 10 * 0.2 if bmi > 20 else 0,  # BMI impact
            0.8 if smoker == "yes" else 0,  # Smoking impact
            children * 0.1,  # Children impact
            0.1 if region in ["northeast", "northwest"] else 0.05  # Region impact
        ]
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#3B82F6' if i != 2 else '#EF4444' if smoker == "yes" else '#3B82F6' for i in range(len(impacts))]
        bars = ax.barh(factors, impacts, color=colors)
        ax.set_xlabel('Relative Impact on Premium')
        ax.set_title('Factors Influencing Your Insurance Premium')
        st.pyplot(fig)
        
        # Recommendations section
        st.subheader("Recommendations")
        
        recommendations = []
        if smoker == "yes":
            recommendations.append("• Consider smoking cessation programs to significantly reduce your premium")
        if bmi > 30:
            recommendations.append("• Lowering your BMI below 30 could lead to premium reductions")
        if age > 50:
            recommendations.append("• Consider comprehensive health check-ups to maintain good health records")
        
        if recommendations:
            st.markdown("\n".join(recommendations))
        else:
            st.markdown("• Your profile already shows favorable factors for lower premiums")
    
    else:
        st.info("👈 Fill in the customer details and click 'Calculate Premium' to see the prediction")
    
    # Model performance metrics
    with st.expander("📊 Model Accuracy Information"):
        st.markdown(f"""
        **How Reliable is This Prediction?**
        
        - 💡 **Typical Difference from Actual Cost:** Around ₹{MAE:,.0f}  
        - 📈 **Prediction Accuracy:** About {R2:.0%} — meaning the model gets it mostly right  
        - 📉 **Most Common Fluctuation:** Your actual cost may swing by about ₹{np.sqrt(MSE):,.0f} either way
        
        *This prediction is based on real-world insurance data and generally works well for similar customers.*
        """)


# FAQ Section at the bottom
st.markdown("---")
with st.expander("❓ Frequently Asked Questions"):
    st.markdown("""
    **What factors most influence insurance charges?**
    
    Smoking status has the largest impact, often increasing premiums by 3-4 times. Age, BMI, and number of dependents also significantly affect costs.
    
    **How accurate is this prediction?**
    
    The model explains about 88% of the variation in insurance charges, which is quite good for this type of prediction. However, individual results may vary.
    
    **What can I do to lower my premium?**
    
    Maintaining a healthy lifestyle is key - avoiding smoking, keeping a healthy BMI, and preventive healthcare can all contribute to lower insurance costs.
    
    **How is BMI calculated?**
    
    BMI = Weight(kg) / Height²(m). A BMI between 18.5-24.9 is considered normal.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Insurance Premium Predictor | Data from Medical Insurance Records")