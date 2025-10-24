import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data import load_raw, basic_clean
from src.predict import score_dataframe, load_model, make_arrow_compatible
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Telco Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .insight-box h3 {
        margin-top: 0;
    }
    .risk-high { color: #ef5350; font-weight: bold; }
    .risk-medium { color: #ff9800; font-weight: bold; }
    .risk-low { color: #66bb6a; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Telco Customer Churn Analytics Dashboard</div>', unsafe_allow_html=True)

# Load and prepare data
@st.cache_resource
def get_model(path: str = None):
    """Load model (cached per session)."""
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def load_and_process_data(model, uploaded_file=None):
    """Load, clean, and score data (no caching)."""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = load_raw()
        df_clean = basic_clean(df)
        if model is None:
            st.error("No model available.")
            return None
        scored = score_dataframe(df_clean, model=model)
        return scored
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        return None


# Sidebar controls
st.sidebar.header("🔧 Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your telco dataset (CSV)", 
    type=['csv'],
    help="Upload a CSV file with customer data to analyze churn patterns"
)

# Reset filters button
if st.sidebar.button("🔄 Reset Filters"):
    st.rerun()

model = get_model()
with st.spinner("🔄 Loading and processing data..."):
    data = load_and_process_data(model, uploaded_file)

if data is None:
    st.error("Model or data could not be loaded. Check your model path or input file.")
    st.stop()

# Sidebar filters
st.sidebar.subheader("🔍 Data Filters")
data['risk_label'] = data['risk_label'].astype(str).str.strip().str.lower()
risk_options = sorted(data['risk_label'].dropna().unique())

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=risk_options,
    default=risk_options
)

contract_options = sorted(data['Contract'].dropna().unique()) if 'Contract' in data.columns else []
contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=contract_options,
    default=contract_options
) if contract_options else []

if 'Internet Type' in data.columns:
    data['Internet Type'] = data['Internet Type'].astype(str).str.strip().replace({'': 'No Internet', 'nan': 'No Internet'})
    internet_service_options = sorted(data['Internet Type'].unique())
    selected_internet = st.sidebar.selectbox("Internet Service Type:", ["All"] + internet_service_options, index=0)
else:
    selected_internet = "All"

filtered_data = data.copy()
filtered_data = filtered_data[filtered_data['risk_label'].isin(risk_filter)]
if contract_options:
    filtered_data = filtered_data[filtered_data['Contract'].isin(contract_filter)]
if selected_internet != "All":
    filtered_data = filtered_data[filtered_data['Internet Type'] == selected_internet]

st.sidebar.subheader("📈 Dataset Info")
st.sidebar.info(f"""
**Total Customers:** {len(data):,}  
**Filtered View:** {len(filtered_data):,}  
**Features:** {len([col for col in data.columns if col not in ['pred_prob', 'risk_label']])}
""")

# Standardized Plotly colors
COLOR_MAP = {'low': '#2e7d32', 'medium': '#f57c00', 'high': '#c62828'}
PLOT_COLORS = px.colors.qualitative.Safe

# Main dashboard layout
col1, col2, col3, col4, col5 = st.columns(5)

# Key Metrics
total_customers = len(data)
high_risk_customers = len(data[data['risk_label'] == 'high'])
churn_rate = None
avg_monthly_revenue = None
revenue_at_risk = None
avg_tenure = None
customer_lifetime_value = None
actual_churn_available = False

# Calculate actual churn rate if available
if 'Churn Label' in data.columns:
    try:
        unique_values = data['Churn Label'].astype(str).str.strip().str.lower().unique()
        if 'yes' in unique_values and 'no' in unique_values:
            actual_churned = data['Churn Label'].astype(str).str.lower().eq('yes').sum()
            churn_rate = (actual_churned / total_customers) * 100
            actual_churn_available = True
        elif set(unique_values).issubset({'1', '0', '1.0', '0.0'}):
            actual_churned = pd.to_numeric(data['Churn Label'], errors='coerce').fillna(0).sum()
            churn_rate = (actual_churned / total_customers) * 100
            actual_churn_available = True
    except Exception:
        pass

elif 'Churn Value' in data.columns:
    try:
        actual_churned = pd.to_numeric(data['Churn Value'], errors='coerce').fillna(0).sum()
        churn_rate = (actual_churned / total_customers) * 100
        actual_churn_available = True
    except Exception:
        pass

elif 'Customer Status' in data.columns:
    try:
        unique_status = data['Customer Status'].astype(str).str.lower().unique()
        if 'churned' in unique_status:
            actual_churned = data['Customer Status'].astype(str).str.lower().eq('churned').sum()
            churn_rate = (actual_churned / total_customers) * 100
            actual_churn_available = True
    except Exception:
        pass

# Revenue & CLV calculations
if 'Monthly Charge' in data.columns:
    avg_monthly_revenue = data['Monthly Charge'].mean()
    revenue_at_risk = data[data['risk_label'] == 'high']['Monthly Charge'].sum()

if 'Tenure in Months' in data.columns and avg_monthly_revenue is not None:
    avg_tenure = data['Tenure in Months'].mean()
    customer_lifetime_value = avg_monthly_revenue * avg_tenure

# --- Metrics Display ---
with col1:
    st.metric(
        label="📉 Churn Rate",
        value=f"{churn_rate:.1f}%" if churn_rate is not None else "N/A",
        help="Percentage of customers who have churned"
    )

with col2:
    st.metric(
        label="⚠️ High Risk Customers",
        value=f"{high_risk_customers:,}"
    )

with col3:
    st.metric(
        label="💰 Avg CLTV",
        value=f"${customer_lifetime_value:,.0f}" if customer_lifetime_value is not None else "N/A",
        help="Average customer lifetime value (Avg Monthly Revenue × Avg Tenure)"
    )

with col4:
    st.metric(
        label="🚨 Revenue at Risk",
        value=f"${revenue_at_risk:,.0f}" if revenue_at_risk is not None else "N/A",
        help="Estimated monthly revenue from high-risk customers"
    )

with col5:
    st.metric(
        label="📆 Avg Tenure",
        value=f"{avg_tenure:.1f} months" if avg_tenure is not None else "N/A",
        help="Average number of months customers stay subscribed"
    )

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Risk Analysis", 
    "📊 Customer Insights", 
    "🔍 Individual Lookup", 
    "📈 Model Performance",
    "💡 Business Recommendations"
])

with tab1:
    st.header("Customer Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        risk_counts = filtered_data['risk_label'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Customer Risk Distribution",
            color_discrete_map={
                'low': '#2e7d32',
                'medium': '#f57c00',
                'high': '#c62828'
            }
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk probability distribution
        fig_hist = px.histogram(
            filtered_data,
            x='pred_prob',
            nbins=30,
            title='Churn Probability Distribution',
            labels={'pred_prob': 'Predicted Churn Probability', 'count': 'Number of Customers'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.add_vline(x=0.2, line_dash="dash", line_color="green", annotation_text="Low Risk Threshold")
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="High Risk Threshold")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # High-risk customer table
    st.subheader("🚨 Top 20 High-Risk Customers")
    high_risk_table = filtered_data.nlargest(20, 'pred_prob')
    
    # Select relevant columns for display
    display_cols = ['pred_prob', 'risk_label']
    if 'Customer ID' in high_risk_table.columns:
        display_cols.insert(0, 'Customer ID')
    if 'Monthly Charge' in high_risk_table.columns:
        display_cols.append('Monthly Charge')
    if 'Contract' in high_risk_table.columns:
        display_cols.append('Contract')
    if 'Tenure in Months' in high_risk_table.columns:
        display_cols.append('Tenure in Months')
    
    # Format the dataframe for better display
    display_df = high_risk_table[display_cols].copy()
    if 'pred_prob' in display_df.columns:
        display_df['Churn Probability'] = display_df['pred_prob'].apply(lambda x: f"{x:.1%}")
        display_df = display_df.drop('pred_prob', axis=1)
    
    st.dataframe(
        make_arrow_compatible(display_df),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.header("Customer Segmentation & Patterns")
    
    # Create visualizations based on available columns
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Contract' in filtered_data.columns:
            # Churn by contract type
            contract_risk = filtered_data.groupby('Contract')['risk_label'].apply(
                lambda x: (x == 'high').sum() / len(x) * 100
            ).reset_index()
            contract_risk.columns = ['Contract', 'High_Risk_Rate']
            
            fig_contract = px.bar(
                contract_risk,
                x='Contract',
                y='High_Risk_Rate',
                title='High Risk Rate by Contract Type',
                labels={'High_Risk_Rate': 'High Risk Rate (%)'},
                color='High_Risk_Rate',
                color_continuous_scale='Reds'
            )
            fig_contract.update_layout(height=400)
            st.plotly_chart(fig_contract, use_container_width=True)
        else:
            # Fallback: Risk vs a numeric feature
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                feature = st.selectbox("Select feature to analyze:", numeric_cols)
                fig_box = px.box(
                    filtered_data,
                    x='risk_label',
                    y=feature,
                    title=f'{feature} by Risk Level',
                    color='risk_label',
                    color_discrete_map={
                        'low': '#2e7d32',
                        'medium': '#f57c00',
                        'high': '#c62828'
                    }
                )
                st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        if 'Tenure in Months' in filtered_data.columns and 'Monthly Charge' in filtered_data.columns:
            # Scatter plot: Tenure vs Monthly Charge colored by risk
            fig_scatter = px.scatter(
                filtered_data.sample(min(1000, len(filtered_data))),  # Sample for performance
                x='Tenure in Months',
                y='Monthly Charge',
                color='risk_label',
                size='pred_prob',
                title='Customer Tenure vs Monthly Charge',
                color_discrete_map={
                    'low': '#2e7d32',
                    'medium': '#f57c00',
                    'high': '#c62828'
                },
                opacity=0.7
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Feature correlation heatmap (if we have numeric features)
    st.subheader("📊 Feature Analysis")
    numeric_features = filtered_data.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['pred_prob']]
    
    if len(numeric_features) > 3:
        corr_data = filtered_data[numeric_features + ['pred_prob']].corr()['pred_prob'].drop('pred_prob').sort_values(key=abs, ascending=False)
        
        fig_corr = px.bar(
            x=corr_data.values,
            y=corr_data.index,
            orientation='h',
            title='Feature Correlation with Churn Probability',
            labels={'x': 'Correlation Coefficient', 'y': 'Features'},
            color=corr_data.values,
            color_continuous_scale='RdBu_r'
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # Churn Map Visualization
    st.subheader("🗺️ Churn Map by Geography")
    geo_cols = [col for col in ['State', 'City', 'Zip Code', 'Latitude', 'Longitude'] if col in filtered_data.columns]
    if 'Latitude' in geo_cols and 'Longitude' in geo_cols:
        fig_map = px.scatter_mapbox(
            filtered_data,
            lat='Latitude',
            lon='Longitude',
            color='risk_label',
            size='pred_prob',
            hover_name='Customer ID' if 'Customer ID' in filtered_data.columns else None,
            mapbox_style='carto-positron',
            zoom=4,
            title='Churn Risk by Location',
            color_discrete_map={'low': '#2e7d32', 'medium': '#f57c00', 'high': '#c62828'}
        )
        fig_map.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    elif 'State' in geo_cols:
        state_churn = filtered_data.groupby('State')['pred_prob'].mean().reset_index()
        fig_choropleth = px.choropleth(
            state_churn,
            locations='State',
            locationmode='USA-states',
            color='pred_prob',
            color_continuous_scale='Reds',
            scope='usa',
            title='Average Churn Probability by State'
        )
        fig_choropleth.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_choropleth, use_container_width=True)
    else:
        st.info("No geographic columns found for mapping. Please upload data with location info.")

with tab3:
    st.header("🔍 Individual Customer Analysis")
    # Use filtered_data for all lookups
    if 'Customer ID' in filtered_data.columns and not filtered_data.empty:
        col1, col2 = st.columns([1, 1])
        with col1:
            risk_level_filter = st.selectbox(
                "Filter by Risk Level:",
                options=['All'] + list(filtered_data['risk_label'].unique()),
                help="Filter customers by their risk level"
            )
        # Filter customers based on risk level selection
        if risk_level_filter == 'All':
            available_customers = filtered_data['Customer ID'].unique()
        else:
            available_customers = filtered_data[filtered_data['risk_label'] == risk_level_filter]['Customer ID'].unique()
        with col2:
            customer_id = st.selectbox(
                "Select Customer ID:",
                options=available_customers[:100],  # Limit for performance
                help="Search for a specific customer to view their risk profile"
            )
        customer_data = filtered_data[filtered_data['Customer ID'] == customer_id].iloc[0]
        col1, col2 = st.columns([1, 2])
        with col1:
            risk_color = {'low': 'risk-low', 'medium': 'risk-medium', 'high': 'risk-high'}[customer_data['risk_label']]
            st.markdown(f"""
            <div class="insight-box">
                <h3>Customer Profile</h3>
                <p><strong>Customer ID:</strong> {customer_data['Customer ID']}</p>
                <p><strong>Risk Level:</strong> <span class="{risk_color}">{customer_data['risk_label'].upper()}</span></p>
                <p><strong>Churn Probability:</strong> {customer_data['pred_prob']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            if 'Monthly Charge' in customer_data:
                st.metric("Monthly Charge", f"${customer_data['Monthly Charge']:.2f}")
            if 'Tenure in Months' in customer_data:
                st.metric("Tenure", f"{customer_data['Tenure in Months']} months")
            if 'Total Charges' in customer_data:
                st.metric("Total Charges", f"${customer_data['Total Charges']:.2f}")
        with col2:
            st.subheader("How does this customer compare?")
            numeric_cols = [col for col in filtered_data.select_dtypes(include=[np.number]).columns 
                          if col not in ['pred_prob', 'Customer ID'] and col in customer_data.index]
            if numeric_cols:
                comparison_data = []
                for col in numeric_cols[:6]:
                    customer_val = customer_data[col]
                    avg_val = filtered_data[col].mean()
                    comparison_data.append({
                        'Feature': col,
                        'Customer': customer_val,
                        'Average': avg_val,
                        'Difference': customer_val - avg_val
                    })
                comparison_df = pd.DataFrame(comparison_data)
                fig_comparison = px.bar(
                    comparison_df,
                    x='Feature',
                    y=['Customer', 'Average'],
                    title='Customer vs Average Values',
                    barmode='group'
                )
                fig_comparison.update_layout(height=400)
                st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.info("Customer ID column not found or no customers after filtering.")

with tab4:
    st.header("📈 Model Performance & Analytics")
    # Use filtered_data for performance evaluation
    has_actual_labels = False
    y_true = None
    label_column = None
    for col in filtered_data.columns:
        if col in ['Churn Label', 'Churn Value', 'Customer Status']:
            try:
                if col == 'Churn Label':
                    temp_labels = filtered_data[col].astype(str).str.strip().str.lower()
                    if temp_labels.isin(['yes', 'no']).all():
                        y_true = temp_labels.eq('yes').astype(int)
                        has_actual_labels = True
                        label_column = col
                        break
                elif col == 'Churn Value':
                    temp_labels = pd.to_numeric(filtered_data[col], errors='coerce')
                    if temp_labels.isin([0, 1]).all():
                        y_true = temp_labels.astype(int)
                        has_actual_labels = True
                        label_column = col
                        break
                elif col == 'Customer Status':
                    temp_labels = filtered_data[col].astype(str).str.strip().str.lower()
                    if 'churned' in temp_labels.values:
                        y_true = temp_labels.eq('churned').astype(int)
                        has_actual_labels = True
                        label_column = col
                        break
            except Exception as e:
                continue
    if has_actual_labels and y_true is not None and not y_true.isna().all():
        valid_mask = ~(y_true.isna() | pd.isna(filtered_data['pred_prob']))
        y_true_clean = y_true[valid_mask]
        y_pred_probs_clean = filtered_data['pred_prob'][valid_mask]
        if len(y_true_clean) > 0 and len(y_true_clean.unique()) > 1:
            y_pred_binary = (y_pred_probs_clean > 0.5).astype(int)
            col1, col2 = st.columns(2)
            with col1:
                try:
                    cm = confusion_matrix(y_true_clean, y_pred_binary)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Not Churn', 'Churn'],
                        y=['Not Churn', 'Churn'],
                        color_continuous_scale='Blues'
                    )
                    fig_cm.update_layout(height=400)
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating confusion matrix: {str(e)}")
            with col2:
                try:
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(y_true_clean, y_pred_probs_clean)
                    roc_auc = auc(fpr, tpr)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_auc:.3f})',
                        line=dict(color='blue', width=2)
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='red', width=1, dash='dash')
                    ))
                    fig_roc.update_layout(
                        title='ROC Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        height=400
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating ROC curve: {str(e)}")
                    roc_auc = np.nan
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                precision = precision_score(y_true_clean, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_clean, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_clean, y_pred_binary, zero_division=0)
                accuracy = accuracy_score(y_true_clean, y_pred_binary)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Precision", f"{precision:.3f}" if not np.isnan(precision) else "N/A")
                with col2:
                    st.metric("📊 Recall", f"{recall:.3f}" if not np.isnan(recall) else "N/A")
                with col3:
                    st.metric("⚖️ F1-Score", f"{f1:.3f}" if not np.isnan(f1) else "N/A")
                with col4:
                    st.metric("📈 ROC-AUC", f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "N/A")
                st.subheader("📊 Detailed Performance Analysis")
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    'Value': [accuracy, precision, recall, f1, roc_auc],
                    'Interpretation': [
                        'Overall correctness of predictions',
                        'Accuracy of positive predictions',
                        'Ability to find all positive cases',
                        'Balance between precision and recall',
                        'Overall discrimination ability'
                    ]
                })
                st.dataframe(make_arrow_compatible(metrics_df), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error calculating performance metrics: {str(e)}")
        else:
            st.warning("Insufficient valid label data for performance evaluation")
    else:
        st.warning("📝 No valid churn labels found for performance evaluation")
    st.subheader("🤖 Model Insights")
    col1, col2 = st.columns(2)
    with col1:
        pred_stats = {
            'Total Predictions': len(filtered_data),
            'High Risk (>50%)': len(filtered_data[filtered_data['pred_prob'] > 0.5]),
            'Medium Risk (20-50%)': len(filtered_data[(filtered_data['pred_prob'] >= 0.2) & (filtered_data['pred_prob'] <= 0.5)]),
            'Low Risk (<20%)': len(filtered_data[filtered_data['pred_prob'] < 0.2]),
            'Average Probability': filtered_data['pred_prob'].mean(),
            'Max Probability': filtered_data['pred_prob'].max(),
            'Min Probability': filtered_data['pred_prob'].min()
        }
        pred_df = pd.DataFrame(list(pred_stats.items()), columns=['Metric', 'Value'])
        st.dataframe(make_arrow_compatible(pred_df), use_container_width=True, hide_index=True)
    with col2:
        if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(8)
        elif hasattr(model, 'coef_') and hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False).head(8)
        else:
            importance_df = None

        if importance_df is not None:
            fig_importance = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Top Feature Importance',
                color='importance',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            model_info = {
                'Model Type': type(model).__name__,
                'Model Features': len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'Unknown',
                'Prediction Range': f"{filtered_data['pred_prob'].min():.3f} - {filtered_data['pred_prob'].max():.3f}"
            }
            model_df = pd.DataFrame(list(model_info.items()), columns=['Property', 'Value'])
            st.dataframe(make_arrow_compatible(model_df), use_container_width=True, hide_index=True)

with tab5:
    st.header("💡 Business Recommendations")
    # Use filtered_data for business insights
    high_risk_count = len(filtered_data[filtered_data['risk_label'] == 'high'])
    medium_risk_count = len(filtered_data[filtered_data['risk_label'] == 'medium'])
    revenue_at_risk = 0
    if 'Monthly Charge' in filtered_data.columns:
        revenue_at_risk = filtered_data[filtered_data['risk_label'] == 'high']['Monthly Charge'].sum()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h3>🎯 Immediate Actions</h3>
            <ul>
                <li><strong>Priority Retention:</strong> Contact high-risk customers immediately</li>
                <li><strong>Targeted Offers:</strong> Deploy retention campaigns for medium-risk segment</li>
                <li><strong>Contract Optimization:</strong> Focus on month-to-month customers</li>
                <li><strong>Service Enhancement:</strong> Address common pain points</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Financial Impact box (always render with fallbacks)
        fin_monthly = f"${revenue_at_risk:,.0f}" if revenue_at_risk and revenue_at_risk > 0 else "N/A"
        fin_annual = f"${revenue_at_risk * 12:,.0f}" if revenue_at_risk and revenue_at_risk > 0 else "N/A"
        fin_just = f"${revenue_at_risk * 0.2:,.0f}/month" if revenue_at_risk and revenue_at_risk > 0 else "N/A"
        st.markdown(f"""
        <div class="insight-box">
            <h3>💰 Financial Impact</h3>
            <p><strong>Monthly Revenue at Risk:</strong> {fin_monthly}</p>
            <p><strong>Annual Impact:</strong> {fin_annual}</p>
            <p><strong>Retention Investment Justification:</strong> {fin_just}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="insight-box">
            <h3>📊 Key Insights</h3>
            <ul>
                <li><strong>High Risk Customers:</strong> {high_risk_count:,} customers need immediate attention</li>
                <li><strong>Prevention Opportunity:</strong> {medium_risk_count:,} customers in medium risk</li>
                <li><strong>Success Metric:</strong> Target 25% reduction in high-risk segment</li>
                <li><strong>ROI Potential:</strong> Every 1% churn reduction = significant revenue impact</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # Top Risk Factors: show feature_importances_ or aggregated coef_ or a helpful fallback
        try:
            items_html = ""
            if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
                top_features = pd.Series(
                    model.feature_importances_, 
                    index=model.feature_names_in_
                ).nlargest(5)
                for feature, importance in top_features.items():
                    items_html += f"<li><strong>{feature}</strong>: {importance:.3f} importance</li>"
            elif hasattr(model, 'coef_') and hasattr(model, 'feature_names_in_'):
                # handle multiclass by aggregating absolute coefficients across classes
                coef = np.array(model.coef_)
                if coef.ndim == 1:
                    agg = np.abs(coef)
                else:
                    agg = np.abs(coef).mean(axis=0)
                top_features = pd.Series(agg, index=model.feature_names_in_).nlargest(5)
                for feature, importance in top_features.items():
                    items_html += f"<li><strong>{feature}</strong>: {importance:.3f} (avg abs coef)</li>"
            else:
                items_html = "<li>No model importances available. Ensure a trained model with feature metadata is present.</li>"
        except Exception as e:
            items_html = f"<li>Could not compute top risk factors: {e}</li>"

        st.markdown(f"""
        <div class="insight-box">
            <h3>🔍 Top Risk Factors</h3>
            <ul>
        {items_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("📋 Recommended Action Plan")
    action_plan = pd.DataFrame({
        'Priority': ['High', 'High', 'Medium', 'Medium', 'Low'],
        'Action': [
            'Contact top 50 high-risk customers within 48 hours',
            'Deploy automated retention email campaign to high-risk segment',
            'Offer contract upgrade incentives to month-to-month customers',
            'Implement customer satisfaction survey for medium-risk customers',
            'Monitor and update model performance monthly'
        ],
        'Timeline': ['48 hours', '1 week', '2 weeks', '1 month', 'Ongoing'],
        'Owner': ['Sales Team', 'Marketing', 'Sales Team', 'Customer Success', 'Data Team']
    })
    st.dataframe(make_arrow_compatible(action_plan), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Telco Churn Analytics Dashboard | Built with Streamlit & Machine Learning</p>
    <p><a href="https://github.com/jshchng/telco-churn-dashboard" target="_blank">GitHub Repository</a> | © 2025 Joshua Chang</p>
</div>
""", unsafe_allow_html=True)