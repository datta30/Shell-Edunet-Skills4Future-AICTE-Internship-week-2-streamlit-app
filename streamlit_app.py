import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Supply Chain Emissions Predictor",
    page_icon="🌱",
    layout="wide"
)

# Title
st.title("🌱 Supply Chain Emissions Predictor")
st.markdown("### Predict emission factors for industries and commodities using AI")

# Add explanation section at the top
with st.expander("ℹ️ How to Use This App - Click to Read First!", expanded=False):
    st.markdown("""
    **Welcome! This app helps you predict environmental emissions from supply chain activities.**
    
    **Simple Steps:**
    1. **Choose your scenario** using the dropdowns on the left
    2. **Set the base emission level** and **rate data quality** using sliders
    3. **Click "Predict"** to see the environmental impact
    
    **What you'll get:** A prediction of how much pollution (in kg) is created per $1 of economic activity.
    
    **Example:** If the app predicts 0.25, it means every $1 spent creates 0.25 kg of greenhouse gas emissions.
    """)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/final_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False

def get_substance_explanation(substance):
    explanations = {
        'Carbon Dioxide': "💨 The most common greenhouse gas. Comes from burning fossil fuels, transportation, and manufacturing.",
        'Methane': "🔥 25x more potent than CO2! Comes from agriculture, landfills, and natural gas operations.",
        'Nitrous Oxide': "⚡ 300x more potent than CO2! Comes from fertilizers, combustion, and industrial activities.",
        'Other GHGs': "🌪️ Includes various industrial gases that are extremely potent greenhouse gases."
    }
    return explanations.get(substance, "")

def get_unit_explanation(unit):
    explanations = {
        'kg/2018 USD, purchaser price': "📊 Regular weight measurement per dollar of economic activity",
        'kg CO2e/2018 USD, purchaser price': "🌍 CO2 equivalent - accounts for how potent different gases are compared to CO2"
    }
    return explanations.get(unit, "")

def get_source_explanation(source):
    explanations = {
        'Commodity': "🌾 Raw materials like crops, oil, metals, lumber - the basic building blocks of the economy",
        'Industry': "🏭 Manufacturing, processing, services - businesses that transform raw materials into products"
    }
    return explanations.get(source, "")

def interpret_prediction(prediction, substance, source, base_factor):
    """Generate human-readable interpretation of the prediction"""
    
    # Economic scale examples
    scenarios = [
        ("a small purchase", 100),
        ("a medium business order", 10000),
        ("a large corporate contract", 1000000)
    ]
    
    interpretation = f"""
    **🎯 What This Means in Plain English:**
    
    **Input Summary:** {substance} emissions from {source.lower()} activities, with base emission level of {base_factor:.3f}
    
    **Result:** {prediction:.4f} kg of greenhouse gas emissions per $1 of economic activity
    
    **Real-World Examples:**
    """
    
    for scenario_name, amount in scenarios:
        total_emissions = prediction * amount
        interpretation += f"\n- **{scenario_name}** (${amount:,}): {total_emissions:.2f} kg of emissions"
        
        # Add context for larger amounts
        if total_emissions > 1000:
            interpretation += f" ({total_emissions/1000:.1f} metric tons)"
    
    # Add environmental context
    if prediction < 0.1:
        interpretation += "\n\n🟢 **Environmental Impact: LOW** - This is relatively clean economic activity!"
    elif prediction < 0.5:
        interpretation += "\n\n🟡 **Environmental Impact: MEDIUM** - Moderate emissions, room for improvement."
    else:
        interpretation += "\n\n🔴 **Environmental Impact: HIGH** - This activity has significant environmental impact."
    
    return interpretation

def main():
    # Load model
    model, scaler, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ Model files not found! Please run the Jupyter notebook first to train and save the model.")
        st.info("Make sure these files exist:\n- models/final_model.pkl\n- models/scaler.pkl")
        return
    else:
        st.success("✅ Model loaded successfully!")

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📝 Input Parameters")
        
        # Substance selection with explanation
        st.subheader("🧪 What Type of Greenhouse Gas?")
        substance_options = {
            'Carbon Dioxide': 0,
            'Methane': 1, 
            'Nitrous Oxide': 2,
            'Other GHGs': 3
        }
        substance = st.selectbox(
            "Choose the main greenhouse gas:", 
            list(substance_options.keys()),
            help="Different gases have different environmental impacts"
        )
        st.caption(get_substance_explanation(substance))
        
        # Unit selection with explanation
        st.subheader("📏 How Should We Measure Emissions?")
        unit_options = {
            'kg/2018 USD, purchaser price': 0,
            'kg CO2e/2018 USD, purchaser price': 1
        }
        unit = st.selectbox(
            "Choose measurement type:", 
            list(unit_options.keys()),
            help="CO2e accounts for different gas potencies"
        )
        st.caption(get_unit_explanation(unit))
        
        # Source selection with explanation
        st.subheader("🏭 What Type of Economic Activity?")
        source_options = {
            'Commodity': 0,
            'Industry': 1
        }
        source = st.selectbox(
            "Choose activity type:", 
            list(source_options.keys()),
            help="Different economic sectors have different emission patterns"
        )
        st.caption(get_source_explanation(source))
        
        # Base factor with explanation
        st.subheader("🔢 Base Emission Level")
        base_factor = st.number_input(
            "Set the baseline emission factor:", 
            min_value=0.0, 
            max_value=5.0, 
            value=0.5, 
            step=0.01,
            help="Higher values = more polluting processes"
        )
        st.caption("💡 **What this means:** This is the starting point for emissions before considering data quality. Lower = cleaner, Higher = more polluting")
        
        # Data quality metrics with explanations
        st.subheader("🎯 Data Quality Assessment")
        st.markdown("*Rate each aspect from 1 (Poor) to 5 (Excellent). Better data quality usually means more accurate predictions.*")
        
        reliability = st.slider(
            "🔍 How reliable is your data?", 
            min_value=1.0, 
            max_value=5.0, 
            value=3.0, 
            step=0.1,
            help="1 = Very unreliable, 5 = Highly trustworthy"
        )
        st.caption("📊 **Reliability:** How much can you trust this data? High reliability = more accurate predictions")
        
        temporal = st.slider(
            "⏰ How consistent is data over time?", 
            min_value=1.0, 
            max_value=5.0, 
            value=3.0, 
            step=0.1,
            help="1 = Very inconsistent over time, 5 = Very stable over time"
        )
        st.caption("📈 **Temporal:** Do emissions stay consistent year to year? High = stable patterns")
        
        geographical = st.slider(
            "🌍 How consistent across different locations?", 
            min_value=1.0, 
            max_value=5.0, 
            value=3.0, 
            step=0.1,
            help="1 = Very different by location, 5 = Same everywhere"
        )
        st.caption("🗺️ **Geographical:** Are emissions similar in different places? High = consistent globally")
        
        technological = st.slider(
            "⚙️ How well does data match current technology?", 
            min_value=1.0, 
            max_value=5.0, 
            value=3.0, 
            step=0.1,
            help="1 = Very outdated technology, 5 = Cutting-edge technology"
        )
        st.caption("🔧 **Technological:** Is the data based on modern technology? High = latest tech")
        
        data_collection = st.slider(
            "📋 How good was the data collection method?", 
            min_value=1.0, 
            max_value=5.0, 
            value=3.0, 
            step=0.1,
            help="1 = Poor collection methods, 5 = Excellent methods"
        )
        st.caption("📝 **Collection Quality:** How carefully was data gathered? High = scientific methods")
        
        # Prediction button
        predict_button = st.button("🔮 Predict Emission Factor", type="primary", use_container_width=True)
    
    with col2:
        st.header("📊 Results & Analysis")
        
        if predict_button:
            # Prepare input data
            input_data = np.array([
                [
                    substance_options[substance],
                    unit_options[unit],
                    base_factor,
                    reliability,
                    temporal,
                    geographical,
                    technological,
                    data_collection,
                    source_options[source]
                ]
            ])
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Display main result
            st.subheader("🎯 Prediction Result")
            
            # Create metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Predicted Emission Factor", 
                    value=f"{prediction:.4f}",
                    help="Kilograms of greenhouse gas per $1 of economic activity"
                )
            
            with metric_col2:
                confidence = 88.9
                st.metric(
                    label="Model Confidence", 
                    value=f"{confidence}%",
                    help="How accurate the AI model is on average"
                )
            
            with metric_col3:
                if prediction < 0.1:
                    impact = "🟢 Low"
                    impact_color = "green"
                elif prediction < 0.5:
                    impact = "🟡 Medium" 
                    impact_color = "orange"
                else:
                    impact = "🔴 High"
                    impact_color = "red"
                    
                st.metric(
                    label="Environmental Impact", 
                    value=impact,
                    help="Overall environmental impact level"
                )
            
            # Add detailed interpretation
            st.subheader("💡 What Does This Mean?")
            interpretation = interpret_prediction(prediction, substance, source, base_factor)
            st.markdown(interpretation)
            
            # Create gauge chart
            st.subheader("📈 Visual Impact Scale")
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Emission Factor (kg per $1)"},
                delta = {'reference': 0.5},
                gauge = {
                    'axis': {'range': [None, 2]},
                    'bar': {'color': impact_color},
                    'steps': [
                        {'range': [0, 0.1], 'color': "lightgreen"},
                        {'range': [0.1, 0.5], 'color': "yellow"},
                        {'range': [0.5, 2], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Input summary table
            st.subheader("📋 Your Input Summary")
            summary_data = {
                'Parameter': [
                    'Substance Type', 'Unit Type', 'Source Type', 'Base Factor',
                    'Reliability', 'Temporal', 'Geographical', 'Technological', 'Data Collection'
                ],
                'Your Selection': [
                    substance, unit, source, f"{base_factor:.3f}",
                    f"{reliability:.1f}/5.0", f"{temporal:.1f}/5.0", f"{geographical:.1f}/5.0", 
                    f"{technological:.1f}/5.0", f"{data_collection:.1f}/5.0"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Model information section
    st.markdown("---")
    st.header("🤖 About This AI Model")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info("""
        **📊 Technical Details**
        - **Algorithm**: Random Forest (100+ decision trees)
        - **Training Data**: 2010-2016 US supply chain data
        - **Accuracy**: 88.9% (Very Good!)
        - **Input Features**: 9 different parameters
        - **Purpose**: Predict environmental emissions
        """)
    
    with info_col2:
        st.info("""
        **🌍 Real-World Applications**
        - **Business Planning**: Choose greener suppliers
        - **Environmental Assessment**: Measure carbon footprint
        - **Investment Decisions**: Compare environmental impact
        - **Sustainability Reporting**: Get data-driven estimates
        - **Policy Research**: Understand emission patterns
        """)
    
    # Add tips section
    with st.expander("💡 Tips for Better Predictions"):
        st.markdown("""
        **🎯 For More Accurate Results:**
        - Use **higher data quality scores** (4-5) when you have reliable, recent data
        - **CO2e units** are often more comprehensive than regular kg measurements
        - **Industry activities** typically have different patterns than **commodity activities**
        - **Base factors around 0.3-0.8** are most common in real-world scenarios
        
        **🧪 Try Different Scenarios:**
        - Compare **methane vs CO2** - methane is much more potent!
        - See how **data quality affects predictions** - better data often gives different results
        - Test **industry vs commodity** - manufacturing often differs from raw materials
        """)

# Run the app
if __name__ == "__main__":
    main()
