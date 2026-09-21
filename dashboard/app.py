import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import custom modules
try:
    from src.data_preprocessing import load_and_clean_data, get_data_quality_report
    from src.feature_engineering import compute_features, compute_mtbf_trend, compute_rolling_failure_rate
    from src.modeling import (
        fit_weibull, fit_exponential, fit_lognormal, WeibullModel,
        weibull_survival, weibull_failure_prob, weibull_conditional_failure_prob,
        weibull_expected_ttf, weibull_median_ttf, weibull_hazard, weibull_percentile,
        compare_distributions, kaplan_meier_estimate, save_model, load_model,
        build_transition_matrix, predict_next_category, get_category_overall_distribution
    )
    from src.evaluation import (
        chronological_split, evaluate_weibull_predictions, baseline_metrics,
        rolling_validation, brier_score_windows, compute_concordance_index
    )
    from src.predictions import (
        predict_next_breakdown, risk_windows, current_risk_level,
        generate_survival_curve, generate_conditional_survival_curve,
        simulate_future_breakdowns
    )
    from src.pdf_generator import generate_3_month_pdf
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configuration
st.set_page_config(page_title='HGG Predictive Maintenance', page_icon='🏭', layout='wide')

# Custom CSS for dark industrial theme and KPI cards
st.markdown("""
<style>
    .reportview-container {
        background-color: #121212;
        color: #e0e0e0;
    }
    .kpi-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border-left: 5px solid #0078D7;
        margin-bottom: 20px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: bold;
        color: #ffffff;
        margin: 10px 0;
    }
    .kpi-label {
        font-size: 14px;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-high { border-left-color: #ff4b4b; }
    .risk-medium { border-left-color: #ffa500; }
    .risk-low { border-left-color: #00cc96; }
    
    div.stMetric > div > div > div > div {
        color: #e0e0e0;
    }
    div.stMetric > div > div > div > div > span {
        color: #a0a0a0;
    }
</style>
""", unsafe_allow_html=True)

# Data loading
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'HGG_downtime.xlsx')
    if not os.path.exists(file_path):
        st.error(f"Data file not found at {file_path}")
        st.stop()
    df = load_and_clean_data(file_path)
    df = compute_features(df)
    return df

@st.cache_resource
def fit_models(df):
    ift = df['inter_failure_days'].dropna().values
    if len(ift) < 5:
        st.error("Not enough data to fit models.")
        st.stop()
    
    weibull_model = fit_weibull(ift)
    comp_df = compare_distributions(ift)
    km_df = kaplan_meier_estimate(ift)
    transition_matrix = build_transition_matrix(df['category'])
    
    return weibull_model, comp_df, km_df, transition_matrix

# Main execution
def main():
    try:
        df_raw = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    weibull_model, comp_df, km_df, transition_matrix = fit_models(df_raw)
    
    # Sidebar Filters
    st.sidebar.image("https://img.icons8.com/color/96/000000/factory.png", width=64)
    st.sidebar.title("🏭 HGG Predictive Maintenance")
    st.sidebar.markdown("---")
    
    min_date = df_raw['actual_start'].min().date()
    max_date = df_raw['actual_start'].max().date()
    
    st.sidebar.subheader("Filters")
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
        
    categories = df_raw['category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
    
    work_types = df_raw['work_type'].dropna().unique().tolist()
    selected_work_types = st.sidebar.multiselect("Work Type", work_types, default=work_types)
    
    min_dur, max_dur = float(df_raw['duration_hours'].min()), float(df_raw['duration_hours'].max())
    duration_range = st.sidebar.slider("Duration (Hours)", min_dur, max_dur, (min_dur, max_dur))
    
    # Apply filters
    mask = (
        (df_raw['actual_start'].dt.date >= start_date) & 
        (df_raw['actual_start'].dt.date <= end_date) &
        (df_raw['category'].isin(selected_categories)) &
        (df_raw['work_type'].isin(selected_work_types)) &
        (df_raw['duration_hours'] >= duration_range[0]) &
        (df_raw['duration_hours'] <= duration_range[1])
    )
    df = df_raw[mask].copy()
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Data loaded: {len(df)} events")
    st.sidebar.info(f"Unfiltered events: {len(df_raw)} events")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Reports")
    
    # Generate PDF logic
    last_cat = df_raw.iloc[-1]['category']
    last_date = df_raw['actual_start'].max().to_pydatetime()
    mtbf = df_raw['inter_failure_days'].mean()
    
    with st.sidebar:
        if st.button("Generate 3-Month PDF Report"):
            with st.spinner("Generating PDF..."):
                schedule_df = simulate_future_breakdowns(weibull_model, transition_matrix, last_date, last_cat, horizon_days=90, use_median=True)
                if not schedule_df.empty:
                    next_days = schedule_df.iloc[0]['days_from_now']
                    most_common = schedule_df['predicted_category'].value_counts().index[0]
                    pdf_path = generate_3_month_pdf(schedule_df, mtbf, next_days, most_common)
                    
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_file,
                            file_name=f"HGG_3Month_Forecast_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.warning("No predictions found for the next 90 days.")
    
    if df.empty:
        st.warning("No data matches the current filters.")
        return
        
    # Pre-calculate global metrics using UNFILTERED data for risk models, but FILTERED for stats
    total_events = len(df)
    total_downtime = df['duration_hours'].sum()
    avg_downtime = df['duration_hours'].mean()
    mtbf = df['inter_failure_days'].mean()
    last_breakdown_date = df_raw['actual_start'].max()
    
    # Next Breakdown Prediction
    prediction = predict_next_breakdown(weibull_model, last_breakdown_date)
    risks = risk_windows(weibull_model, prediction['elapsed_days'])
    risk_7d = risks[7] * 100
    risk_level, risk_color = current_risk_level(risks[7])
    
    # Create Tabs
    tabs = st.tabs([
        "📊 Overview", 
        "📅 Future Schedule",
        "📈 Frequency Analysis", 
        "⏱️ Downtime Analysis", 
        "🔮 Prediction & Survival",
        "✅ Model Performance", 
        "📋 Data Table"
    ])
    
    # ----------------------------------------
    # TAB 1: Overview
    # ----------------------------------------
    with tabs[0]:
        st.header("System Overview")
        st.markdown("This dashboard provides a proactive view of machine health for the **HGG CNC Profile Coping Machine**. Utilizing statistical **Weibull survival models** and **Markov Chains**, we aim to shift from reactive firefighting to planned maintenance.")
        
        # KPI Row
        cols = st.columns(7)
        
        with cols[0]:
            st.markdown(f'''
                <div class="kpi-card" title="Total number of maintenance events recorded in the filtered date range.">
                    <div class="kpi-label">Events</div>
                    <div class="kpi-value">{total_events}</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with cols[1]:
            st.markdown(f'''
                <div class="kpi-card" title="Sum of all downtime hours recorded.">
                    <div class="kpi-label">Total Downtime (h)</div>
                    <div class="kpi-value">{total_downtime:.1f}</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with cols[2]:
            st.markdown(f'''
                <div class="kpi-card" title="Average duration of a downtime event in hours.">
                    <div class="kpi-label">Avg Downtime (h)</div>
                    <div class="kpi-value">{avg_downtime:.1f}</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with cols[3]:
            st.markdown(f'''
                <div class="kpi-card" title="Mean Time Between Failures: Average number of days between machine breakdowns.">
                    <div class="kpi-label">MTBF (Days)</div>
                    <div class="kpi-value">{mtbf:.1f}</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with cols[4]:
            st.markdown(f'''
                <div class="kpi-card" title="The date of the most recent recorded failure.">
                    <div class="kpi-label">Last Breakdown</div>
                    <div class="kpi-value" style="font-size:18px;">{last_breakdown_date.strftime('%Y-%m-%d')}</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with cols[5]:
            pred_date = prediction['predicted_date_median'].strftime('%Y-%m-%d') if pd.notnull(prediction['predicted_date_median']) else 'N/A'
            st.markdown(f'''
                <div class="kpi-card" title="The statistically most likely date for the next breakdown based on the Weibull survival model.">
                    <div class="kpi-label">Next Expected</div>
                    <div class="kpi-value" style="font-size:18px;">{pred_date}</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with cols[6]:
            risk_class = f"risk-{risk_level.lower()}"
            st.markdown(f'''
                <div class="kpi-card {risk_class}" title="The probability that the machine will fail within the next 7 days, given that it has survived up until today.">
                    <div class="kpi-label">7-Day Risk</div>
                    <div class="kpi-value" style="color:{risk_color};">{risk_7d:.1f}%</div>
                </div>
            ''', unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Timeline and Risk sections
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Breakdown Timeline")
            fig = px.scatter(
                df, x='actual_start', y='duration_hours', color='category',
                hover_data=['directive', 'work_details', 'duration_hours'],
                labels={'actual_start': 'Date', 'duration_hours': 'Duration (h)'},
                title='Historical Breakdowns',
                template='plotly_dark'
            )
            fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Predictive Maintenance Risk")
            
            # Risk Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_7d,
                title={'text': "7-Day Failure Risk (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "gold"},
                        {'range': [70, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), template='plotly_dark')
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Details
            st.markdown(f"**Elapsed Days Since Last Failure:** {prediction['elapsed_days']:.1f}")
            st.markdown(f"**Expected Days Remaining:** {prediction['remaining_expected_days']:.1f}")
            
            # Risk Table
            risk_df = pd.DataFrame([
                {"Window": "1 Day", "Probability": f"{risks[1]*100:.1f}%"},
                {"Window": "3 Days", "Probability": f"{risks[3]*100:.1f}%"},
                {"Window": "7 Days", "Probability": f"{risks[7]*100:.1f}%"},
                {"Window": "14 Days", "Probability": f"{risks[14]*100:.1f}%"},
                {"Window": "30 Days", "Probability": f"{risks[30]*100:.1f}%"},
            ])
            st.table(risk_df)
            
            # Predicted Fault Type
            st.markdown("---")
            st.subheader("🔧 Predicted Fault Type")
            last_category = df_raw.iloc[-1]['category']
            cat_predictions = predict_next_category(transition_matrix, last_category, top_n=5)
            
            if cat_predictions:
                st.markdown(f"Based on the last breakdown being **{last_category}**, the most likely next failure:")
                for i, pred in enumerate(cat_predictions):
                    bar_width = int(pred['probability'] * 100)
                    emoji = '🔴' if i == 0 else '🟠' if i == 1 else '🟡'
                    bar_color = '#e74c3c' if i == 0 else '#f39c12' if i == 1 else '#2ecc71'
                    st.markdown(f"""
                    {emoji} **{pred['category']}** — {pred['probability']*100:.0f}%
                    <div style="background-color:#333;border-radius:4px;height:20px;width:100%;">
                        <div style="background-color:{bar_color};height:100%;width:{bar_width}%;border-radius:4px;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                st.caption("⚠️ Prediction based on Markov Chain transition probabilities from historical failure sequence. Not a guarantee.")
            else:
                st.info("Not enough historical data to predict fault type.")
            
    # ----------------------------------------
    # TAB 2: Future Schedule
    # ----------------------------------------
    with tabs[1]:
        st.header("📅 Predicted Maintenance Schedule")
        st.markdown("""
        This section simulates future breakdowns using the Weibull survival model and Markov Chain 
        category predictions. It chains predictions forward from the last known breakdown to build 
        a maintenance planning calendar.
        
        > ⚠️ **Important**: These are statistical projections, not certainties. Actual breakdowns depend on 
        > operating conditions, maintenance quality, and factors not captured in historical data alone.
        """)
        
        # Controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            horizon = st.selectbox("Forecast Horizon", [90, 180, 270, 365], index=1, format_func=lambda x: f"{x} days ({x//30} months)")
        with col_ctrl2:
            use_median = st.checkbox("Use Median TTF (more conservative)", value=True, help="Median is more robust to outliers. Uncheck to use Mean TTF.")
        
        last_category = df_raw.iloc[-1]['category']
        last_date = df_raw['actual_start'].max()
        
        schedule_df = simulate_future_breakdowns(
            model=weibull_model,
            transition_matrix=transition_matrix,
            last_breakdown_date=last_date.to_pydatetime(),
            last_category=last_category,
            horizon_days=horizon,
            use_median=use_median
        )
        
        if not schedule_df.empty:
            # Summary metrics
            st.markdown("---")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Predicted Breakdowns", len(schedule_df))
            with col_s2:
                next_pred = schedule_df.iloc[0]
                days_until = next_pred['days_from_now']
                label = f"{days_until:.0f} days" if days_until > 0 else "OVERDUE"
                st.metric("Next Breakdown In", label)
            with col_s3:
                most_common = schedule_df['predicted_category'].value_counts().index[0]
                st.metric("Most Common Predicted Fault", most_common)
            
            st.markdown("---")
            
            # Timeline visualization
            st.subheader("Predicted Breakdown Timeline")
            fig_timeline = go.Figure()
            
            # Add historical events
            fig_timeline.add_trace(go.Scatter(
                x=df_raw['actual_start'],
                y=[0.5] * len(df_raw),
                mode='markers',
                name='Historical (Actual)',
                marker=dict(size=8, color='#17becf', symbol='circle'),
                text=df_raw['directive'],
                hovertemplate='<b>%{text}</b><br>Date: %{x}<extra>Historical</extra>'
            ))
            
            # Add predicted events
            colors_map = {
                'Plasma/Torch': '#e74c3c',
                'Conveyor/Roller': '#3498db',
                'Laser/Sensor': '#9b59b6',
                'Dust Collector': '#95a5a6',
                'Hydraulic': '#e67e22',
                'Control/Electrical': '#f1c40f',
                'Mechanical': '#2ecc71',
                'Other': '#1abc9c'
            }
            
            for _, row in schedule_df.iterrows():
                color = colors_map.get(row['predicted_category'], '#ffffff')
                fig_timeline.add_trace(go.Scatter(
                    x=[row['predicted_date']],
                    y=[1.5],
                    mode='markers',
                    name=f"Predicted #{row['event_number']}",
                    marker=dict(size=14, color=color, symbol='diamond', line=dict(width=2, color='white')),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Predicted Breakdown #{row['event_number']}</b><br>"
                        f"Date: {row['predicted_date'].strftime('%Y-%m-%d')}<br>"
                        f"Category: {row['predicted_category']} ({row['category_probability']}%)<br>"
                        f"Top 3: {row['top_3_categories']}<br>"
                        f"<extra></extra>"
                    )
                ))
                
                # Add range bar for earliest-latest
                fig_timeline.add_trace(go.Scatter(
                    x=[row['earliest_estimate'], row['latest_estimate']],
                    y=[1.5, 1.5],
                    mode='lines',
                    line=dict(color=color, width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add vertical line for today
            fig_timeline.add_vline(x=datetime.now(), line_dash="dash", line_color="yellow", annotation_text="Today")
            
            fig_timeline.update_layout(
                title="Historical + Predicted Breakdowns",
                xaxis_title="Date",
                yaxis=dict(visible=False),
                template='plotly_dark',
                height=350,
                showlegend=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Schedule Table
            st.subheader("Predicted Schedule Table")
            display_schedule = schedule_df[['event_number', 'predicted_date', 'days_from_now', 
                                            'predicted_category', 'category_probability', 
                                            'top_3_categories', 'earliest_estimate', 'latest_estimate']].copy()
            display_schedule.columns = ['#', 'Predicted Date', 'Days From Now', 
                                        'Most Likely Cause', 'Confidence (%)', 
                                        'Top 3 Possible Causes', 'Earliest Estimate', 'Latest Estimate']
            display_schedule['Predicted Date'] = display_schedule['Predicted Date'].dt.strftime('%Y-%m-%d')
            display_schedule['Earliest Estimate'] = display_schedule['Earliest Estimate'].dt.strftime('%Y-%m-%d')
            display_schedule['Latest Estimate'] = display_schedule['Latest Estimate'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_schedule, use_container_width=True, hide_index=True)
            
            # Category distribution in predictions
            st.subheader("Predicted Fault Distribution")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pred_cat_counts = schedule_df['predicted_category'].value_counts().reset_index()
                pred_cat_counts.columns = ['category', 'count']
                fig_pred_donut = px.pie(pred_cat_counts, values='count', names='category', 
                                       hole=0.4, title='Predicted Breakdown Categories',
                                       template='plotly_dark')
                st.plotly_chart(fig_pred_donut, use_container_width=True)
            with col_p2:
                # Monthly predicted breakdown count
                schedule_df['pred_month'] = schedule_df['predicted_date'].dt.to_period('M').astype(str)
                monthly_pred = schedule_df.groupby('pred_month').size().reset_index(name='count')
                fig_monthly_pred = px.bar(monthly_pred, x='pred_month', y='count',
                                         title='Predicted Breakdowns per Month',
                                         template='plotly_dark',
                                         color_discrete_sequence=['#e74c3c'])
                fig_monthly_pred.update_layout(xaxis_title='Month', yaxis_title='Predicted Breakdowns')
                st.plotly_chart(fig_monthly_pred, use_container_width=True)
            
            # Transition matrix visualization
            st.subheader("Fault Category Transition Probabilities")
            st.markdown("This heatmap shows the probability of transitioning from one fault type to another. For example, if a machine just had a Hydraulic fault, read across the Hydraulic row to see the probability of what the *next* fault will be. This forms the basis of our predictive categorizations.")
            fig_trans = px.imshow(
                transition_matrix.values,
                labels=dict(x="Next Fault", y="Current Fault", color="Probability"),
                x=transition_matrix.columns.tolist(),
                y=transition_matrix.index.tolist(),
                color_continuous_scale='RdYlGn_r',
                template='plotly_dark',
                text_auto='.0%'
            )
            fig_trans.update_layout(height=500)
            st.plotly_chart(fig_trans, use_container_width=True)
        else:
            st.warning("Unable to generate future schedule. Check model parameters.")

    # ----------------------------------------
    # TAB 3: Frequency Analysis
    # ----------------------------------------
    with tabs[2]:
        st.header("Frequency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly Breakdown Frequency
            df['month_year'] = df['actual_start'].dt.to_period('M').astype(str)
            monthly_counts = df.groupby('month_year').size().reset_index(name='count')
            monthly_counts['rolling_avg'] = monthly_counts['count'].rolling(window=3, min_periods=1).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly_counts['month_year'], y=monthly_counts['count'], name='Count', marker_color='#1f77b4'))
            fig.add_trace(go.Scatter(x=monthly_counts['month_year'], y=monthly_counts['rolling_avg'], name='3-Mo Rolling Avg', line=dict(color='#ff7f0e', width=3)))
            fig.update_layout(title="Monthly Breakdown Frequency", xaxis_title="Month", yaxis_title="Number of Breakdowns", template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
            # Day of week x Month Heatmap
            heatmap_data = df.groupby(['day_of_week', 'month']).size().unstack(fill_value=0)
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            heatmap_data = heatmap_data.reindex(range(7), fill_value=0)
            
            fig_heat = px.imshow(
                heatmap_data, 
                labels=dict(x="Month", y="Day of Week", color="Count"),
                x=[str(i) for i in heatmap_data.columns],
                y=days,
                title="Breakdowns Heatmap: Month vs Day of Week",
                color_continuous_scale="Viridis",
                template='plotly_dark'
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with col2:
            # MTBF Trend
            mtbf_df = compute_mtbf_trend(df, window=10)
            if not mtbf_df.empty:
                fig_mtbf = px.line(mtbf_df, x='actual_start', y='mtbf_rolling', title="MTBF Trend (10-event rolling)", template='plotly_dark')
                fig_mtbf.add_hline(y=mtbf, line_dash="dash", line_color="red", annotation_text="Overall MTBF")
                fig_mtbf.update_traces(line=dict(color='#2ca02c', width=3))
                st.plotly_chart(fig_mtbf, use_container_width=True)
            else:
                st.info("Not enough data for MTBF trend.")
                
            # Rolling Failure Rate
            fail_df = compute_rolling_failure_rate(df, window_days=30)
            if not fail_df.empty:
                fig_fail = px.line(fail_df, x='date', y='failure_rate', title="Rolling Failure Rate (30 Days)", template='plotly_dark')
                fig_fail.update_traces(line=dict(color='#d62728', width=3))
                st.plotly_chart(fig_fail, use_container_width=True)
            else:
                st.info("Not enough data for Rolling Failure Rate.")
                
    # ----------------------------------------
    # TAB 4: Downtime Analysis
    # ----------------------------------------
    with tabs[3]:
        st.header("Downtime Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dt = px.scatter(df, x='actual_start', y='duration_hours', title="Downtime Duration Over Time", template='plotly_dark', color_discrete_sequence=['#17becf'])
            fig_dt.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig_dt, use_container_width=True)
            
            # Category donut
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            fig_donut = px.pie(cat_counts, values='count', names='category', hole=0.4, title="Breakdowns by Category", template='plotly_dark')
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col2:
            # Histogram and Boxplot
            fig_hist = px.histogram(df, x="duration_hours", marginal="box", title="Duration Distribution", template='plotly_dark', color_discrete_sequence=['#9467bd'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Total Hours by Category
            cat_hours = df.groupby('category')['duration_hours'].sum().reset_index().sort_values('duration_hours', ascending=True)
            fig_bar = px.bar(cat_hours, x='duration_hours', y='category', orientation='h', title="Total Downtime Hours by Category", template='plotly_dark', color_discrete_sequence=['#8c564b'])
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Stats Table
            stats = {
                "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
                "Value (Hours)": [
                    f"{df['duration_hours'].mean():.2f}",
                    f"{df['duration_hours'].median():.2f}",
                    f"{df['duration_hours'].min():.2f}",
                    f"{df['duration_hours'].max():.2f}",
                    f"{df['duration_hours'].std():.2f}"
                ]
            }
            st.table(pd.DataFrame(stats))

    # ----------------------------------------
    # TAB 5: Prediction & Survival
    # ----------------------------------------
    with tabs[4]:
        st.header("Prediction & Survival Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Survival Curves")
            surv_df = generate_survival_curve(weibull_model, max_days=int(df_raw['inter_failure_days'].max()*1.2))
            
            fig_surv = go.Figure()
            # Kaplan-Meier
            fig_surv.add_trace(go.Scatter(x=km_df['time'], y=km_df['survival_prob'], mode='lines', line=dict(shape='hv', color='white', width=2), name='Kaplan-Meier (Empirical)'))
            fig_surv.add_trace(go.Scatter(x=km_df['time'], y=km_df['ci_upper'], mode='lines', line=dict(shape='hv', color='rgba(255,255,255,0.2)', width=1), name='KM 95% CI', showlegend=False))
            fig_surv.add_trace(go.Scatter(x=km_df['time'], y=km_df['ci_lower'], mode='lines', line=dict(shape='hv', color='rgba(255,255,255,0.2)', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.1)', showlegend=False))
            
            # Weibull
            fig_surv.add_trace(go.Scatter(x=surv_df['days'], y=surv_df['survival_prob'], mode='lines', line=dict(color='#00d2ff', width=3), name='Weibull (Parametric)'))
            
            fig_surv.update_layout(title="Overall Survival Probability", xaxis_title="Days", yaxis_title="Probability", template='plotly_dark')
            st.plotly_chart(fig_surv, use_container_width=True)
            
            st.subheader("Conditional Survival (Given Current State)")
            cond_surv_df = generate_conditional_survival_curve(weibull_model, prediction['elapsed_days'])
            fig_cond = px.line(cond_surv_df, x='additional_days', y='survival_prob', title=f"Survival Probability given {prediction['elapsed_days']:.1f} days elapsed", template='plotly_dark')
            fig_cond.update_traces(line=dict(color='#ff007f', width=3))
            st.plotly_chart(fig_cond, use_container_width=True)
            
        with col2:
            st.subheader("Model Interpretation")
            shape = weibull_model.shape
            scale = weibull_model.scale
            
            st.markdown(f"**Shape Parameter (β):** {shape:.3f}")
            if shape < 1:
                st.warning("β < 1: Decreasing failure rate (infant mortality / early failures).")
            elif abs(shape - 1) < 0.1:
                st.info("β ≈ 1: Constant failure rate (random failures).")
            else:
                st.error("β > 1: Increasing failure rate (wear-out / aging).")
                
            st.markdown(f"**Scale Parameter (η):** {scale:.3f} days (characteristic life)")
            st.markdown(f"**Expected TTF:** {weibull_expected_ttf(weibull_model):.2f} days")
            st.markdown(f"**Median TTF:** {weibull_median_ttf(weibull_model):.2f} days")
            
            st.markdown("---")
            st.subheader("Risk Windows Breakdown")
            risk_bar_df = pd.DataFrame({
                'Window': [f"{w} Days" for w in risks.keys()],
                'Probability': [p for p in risks.values()]
            })
            fig_risk_bar = px.bar(risk_bar_df, x='Window', y='Probability', text_auto='.1%', title="Failure Probability in Next X Days", template='plotly_dark', color='Probability', color_continuous_scale='Reds')
            fig_risk_bar.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_risk_bar, use_container_width=True)

    # ----------------------------------------
    # TAB 6: Model Performance
    # ----------------------------------------
    with tabs[5]:
        st.header("Model Performance & Validation")
        
        # Prepare eval data
        train_df, test_df = chronological_split(df_raw, train_frac=0.8)
        train_ift = train_df['inter_failure_days'].dropna().values
        test_ift = test_df['inter_failure_days'].dropna().values
        
        if len(test_ift) > 0 and len(train_ift) > 0:
            eval_model = fit_weibull(train_ift)
            metrics = evaluate_weibull_predictions(eval_model, test_ift)
            base_metrics = baseline_metrics(train_ift, test_ift)
            c_index = compute_concordance_index([weibull_median_ttf(eval_model)]*len(test_ift), test_ift)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution Comparison (Full Data)")
                st.dataframe(comp_df, use_container_width=True)
                
                st.subheader("Metrics (Test Set)")
                met_df = pd.DataFrame([
                    {"Metric": "MAE", "Weibull": metrics['mae'], "Baseline Mean": base_metrics['mae_mean'], "Baseline Median": base_metrics['mae_median']},
                    {"Metric": "RMSE", "Weibull": metrics['rmse'], "Baseline Mean": base_metrics['rmse_mean'], "Baseline Median": base_metrics['rmse_median']}
                ])
                st.table(met_df)
                st.info(f"Concordance Index: {c_index:.3f}")
                
            with col2:
                st.subheader("Predictions vs Actuals (Test Set)")
                pred_median = weibull_median_ttf(eval_model)
                pred_mean = weibull_expected_ttf(eval_model)
                
                fig_act_pred = go.Figure()
                fig_act_pred.add_trace(go.Scatter(y=test_ift, mode='markers', name='Actual IFT', marker=dict(color='yellow')))
                fig_act_pred.add_trace(go.Scatter(y=[pred_median]*len(test_ift), mode='lines', name='Predicted Median', line=dict(color='green', dash='dash')))
                fig_act_pred.add_trace(go.Scatter(y=[pred_mean]*len(test_ift), mode='lines', name='Predicted Mean', line=dict(color='blue', dash='dot')))
                fig_act_pred.update_layout(title="Test Set Inter-Failure Times vs Predictions", xaxis_title="Test Sample Index", yaxis_title="Days", template='plotly_dark')
                st.plotly_chart(fig_act_pred, use_container_width=True)
                
                # Brier scores
                bs = brier_score_windows(eval_model, test_ift)
                bs_df = pd.DataFrame({"Window (Days)": list(bs.keys()), "Brier Score": list(bs.values())})
                fig_bs = px.bar(bs_df, x="Window (Days)", y="Brier Score", title="Brier Score by Window (Lower is Better)", template='plotly_dark')
                st.plotly_chart(fig_bs, use_container_width=True)
                
            # Rolling Validation
            st.subheader("Rolling Validation (Expanding Window)")
            roll_df = rolling_validation(df_raw['inter_failure_days'].dropna().values)
            if not roll_df.empty:
                fig_roll = px.line(roll_df, x='train_size', y='abs_error', title="Absolute Error over Expanding Training Window", template='plotly_dark')
                fig_roll.add_trace(go.Scatter(x=roll_df['train_size'], y=roll_df['abs_error'].rolling(5).mean(), name='5-period Moving Avg', line=dict(color='orange')))
                st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.warning("Not enough data to perform train/test validation.")

    # ----------------------------------------
    # TAB 7: Data Table
    # ----------------------------------------
    with tabs[6]:
        st.header("Data Inspector")
        
        # Data Quality Report
        with st.expander("Data Quality Report", expanded=False):
            dq = get_data_quality_report(df_raw)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Records", dq['total_records'])
                st.metric("Zero Duration Count", dq['zero_duration_count'])
                st.metric("Extreme Duration Count", dq['extreme_duration_count'])
            with col_b:
                st.write("Null Counts by Column:")
                st.json(dq['null_counts'])
            st.write("Category Distribution:")
            st.json(dq['category_distribution'])
        
        st.subheader("Filtered Dataset")
        display_cols = ['wo_no', 'actual_start', 'actual_finish', 'category', 'work_type', 'duration_hours', 'inter_failure_days', 'work_details']
        st.dataframe(df[display_cols].sort_values('actual_start', ascending=False), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name='hgg_filtered_downtime.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()
