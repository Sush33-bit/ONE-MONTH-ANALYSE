import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Employee Performance Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #28a745, #20c997);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .top-performer {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_and_process_data(uploaded_file):
    """Load and process the Excel file"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Convert datetime columns
        datetime_cols = ['ActivityDate', 'StartTravelTime', 'EndTravelTime', 'GTV1PunchInTime', 'GTV1PunchOutTime', 'GTV2PunchInTime', 'GTV2PunchOutTime']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate work duration in hours
        df['TravelDuration'] = (df['EndTravelTime'] - df['StartTravelTime']).dt.total_seconds() / 3600
        df['TravelDuration'] = df['TravelDuration'].fillna(0)
        df['TravelDuration'] = df['TravelDuration'].apply(lambda x: max(0, x) if pd.notna(x) else 0)
        
        # Calculate GTV work durations
        if 'GTV1PunchInTime' in df.columns and 'GTV1PunchOutTime' in df.columns:
            df['GTV1Duration'] = (df['GTV1PunchOutTime'] - df['GTV1PunchInTime']).dt.total_seconds() / 3600
            df['GTV1Duration'] = df['GTV1Duration'].fillna(0)
            df['GTV1Duration'] = df['GTV1Duration'].apply(lambda x: max(0, x) if pd.notna(x) else 0)
        
        if 'GTV2PunchInTime' in df.columns and 'GTV2PunchOutTime' in df.columns:
            df['GTV2Duration'] = (df['GTV2PunchOutTime'] - df['GTV2PunchInTime']).dt.total_seconds() / 3600
            df['GTV2Duration'] = df['GTV2Duration'].fillna(0)
            df['GTV2Duration'] = df['GTV2Duration'].apply(lambda x: max(0, x) if pd.notna(x) else 0)
        
        # Calculate total work duration
        travel_duration = df['TravelDuration'].fillna(0)
        gtv1_duration = df.get('GTV1Duration', pd.Series([0] * len(df))).fillna(0)
        gtv2_duration = df.get('GTV2Duration', pd.Series([0] * len(df))).fillna(0)
        
        df['TotalWorkDuration'] = travel_duration + gtv1_duration + gtv2_duration
        
        # Calculate total village activities
        gtv1_act = df.get('GTV1VilActCount', pd.Series([0] * len(df))).fillna(0)
        gtv2_act = df.get('GTV2VilActCount', pd.Series([0] * len(df))).fillna(0)
        market_act = df.get('MarketActCount', pd.Series([0] * len(df))).fillna(0)
        
        df['TotalActivities'] = gtv1_act + gtv2_act + market_act
        
        # Calculate distance traveled
        df['DistanceTraveled'] = df['EndKM'] - df['StartKM']
        df['DistanceTraveled'] = df['DistanceTraveled'].apply(lambda x: max(0, x) if pd.notna(x) else 0)
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def analyze_employee_performance(df):
    """Analyze employee performance data with distinct days calculation"""
    
    # Filter out rows without employee names
    df_filtered = df[df['KAName'].notna() & (df['KAName'] != 'NULL')].copy()
    
    if df_filtered.empty:
        return {}
    
    # Group by employee
    employee_analysis = df_filtered.groupby('KAName').agg({
        'TotalWorkDuration': 'sum',  # Total hours
        'TotalActivities': 'sum',     # Total activities
        'DistanceTraveled': 'sum',    # Total distance
        'KmByKA': 'sum',              # Total km by KA
        'KMReimbursement': 'sum',     # Total reimbursement
        'DA': 'sum',                  # Total DA
        'ActivityDate': lambda x: x.nunique(),  # Distinct days count
        'IsSunday': lambda x: (x == 'Y').sum()  # Sunday work days
    }).round(2)
    
    # Reset index and rename columns
    employee_analysis = employee_analysis.reset_index()
    employee_analysis.columns = [
        'KAName', 'Total_Hours', 'Total_Activities', 'Distance_Traveled',
        'KmByKA_Total', 'Total_Reimbursement', 'Total_DA', 
        'Days_Worked', 'Sunday_Work_Days'
    ]
    
    # Calculate average hours per day
    employee_analysis['Avg_Hours_Per_Day'] = np.where(
        employee_analysis['Days_Worked'] > 0,
        employee_analysis['Total_Hours'] / employee_analysis['Days_Worked'],
        0
    ).round(2)
    
    # Calculate efficiency metrics
    employee_analysis['Activities_Per_Hour'] = (employee_analysis['Total_Activities'] / 
                                              (employee_analysis['Total_Hours'] + 0.1)).round(2)
    employee_analysis['Activities_Per_KM'] = (employee_analysis['Total_Activities'] / 
                                            (employee_analysis['Distance_Traveled'] + 0.1)).round(2)
    employee_analysis['Efficiency_Score'] = (employee_analysis['Activities_Per_Hour'] * 0.7 + 
                                           employee_analysis['Activities_Per_KM'] * 0.3).round(2)
    
    # Sort by efficiency score
    employee_analysis = employee_analysis.sort_values('Efficiency_Score', ascending=False)
    
    return employee_analysis

def create_performance_metrics(df, employee_analysis):
    """Create performance metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_employees = len(employee_analysis) if not employee_analysis.empty else 0
    total_activities = employee_analysis['Total_Activities'].sum() if not employee_analysis.empty else 0
    total_distance = employee_analysis['Distance_Traveled'].sum() if not employee_analysis.empty else 0
    total_hours = employee_analysis['Total_Hours'].sum() if not employee_analysis.empty else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Active Employees</h3>
            <h2>{total_employees}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏘️ Total Activities</h3>
            <h2>{int(total_activities)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚗 Distance (KM)</h3>
            <h2>{total_distance:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏰ Total Hours</h3>
            <h2>{total_hours:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)

def create_top_performers_analysis(employee_analysis):
    """Create top performers visualization and analysis"""
    if employee_analysis.empty:
        return None, None, None
    
    # Top performers based on efficiency (high activities, less travel)
    top_performers = employee_analysis.head(10).copy()
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=top_performers['Distance_Traveled'],
        y=top_performers['Total_Activities'],
        mode='markers+text',
        marker=dict(
            size=top_performers['Total_Hours'],
            color=top_performers['Efficiency_Score'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="Efficiency Score"),
            sizemode='diameter',
            sizeref=2.*max(top_performers['Total_Hours'])/(15.**2),
            sizemin=4
        ),
        text=top_performers['KAName'].str.split().str[0],  # First name only for readability
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>' +
                     'Distance: %{x:.1f} KM<br>' +
                     'Activities: %{y}<br>' +
                     'Hours: %{marker.size:.1f}<br>' +
                     'Efficiency: %{marker.color:.2f}' +
                     '<extra></extra>',
        name='Employees'
    ))
    
    fig.update_layout(
        title="🏆 Top Performers: Activities vs Distance (Bubble size = Hours worked)",
        xaxis_title="Distance Traveled (KM)",
        yaxis_title="Total Activities Completed",
        height=500,
        showlegend=False
    )
    
    # Top 5 most efficient (traveled less, worked more)
    efficient_workers = employee_analysis.nlargest(5, 'Activities_Per_KM')[['KAName', 'Total_Activities', 'Distance_Traveled', 'Activities_Per_KM', 'Total_Hours']]
    
    # Top 5 hardest workers (most activities)
    hard_workers = employee_analysis.nlargest(5, 'Total_Activities')[['KAName', 'Total_Activities', 'Total_Hours', 'Days_Worked', 'Activities_Per_Hour']]
    
    return fig, efficient_workers, hard_workers

def create_time_distribution_chart(df):
    """Create time distribution bar chart"""
    if df.empty or 'TotalWorkDuration' not in df.columns:
        return None
    
    # Filter valid work durations
    work_data = df[df['TotalWorkDuration'] > 0]['TotalWorkDuration'] * 60  # Convert to minutes
    
    if work_data.empty:
        return None
    
    # Create time bins
    bins = [0, 30, 60, 90, 120, 180, 300, 480, float('inf')]
    labels = ['0-30min', '30-60min', '60-90min', '90-120min', '120-180min', '180-300min', '300-480min', '480+min']
    
    work_data_binned = pd.cut(work_data, bins=bins, labels=labels, include_lowest=True)
    time_distribution = work_data_binned.value_counts().sort_index()
    
    fig = px.bar(
        x=time_distribution.index,
        y=time_distribution.values,
        title="📊 Work Duration Distribution",
        labels={'x': 'Time Ranges', 'y': 'Number of Work Sessions'},
        color=time_distribution.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def analyze_low_performers(employee_analysis):
    """Analyze employees working less than 1.5 hours"""
    if employee_analysis.empty:
        return pd.DataFrame()
    
    low_performers = employee_analysis[employee_analysis['Avg_Hours_Per_Day'] < 1.5].copy()
    return low_performers[['KAName', 'Total_Hours', 'Avg_Hours_Per_Day', 'Days_Worked', 'Total_Activities']]

def create_daily_activity_heatmap(df):
    """Create daily activity heatmap"""
    if df.empty or 'ActivityDate' not in df.columns:
        return None
    
    # Filter valid data
    df_valid = df[df['KAName'].notna() & (df['KAName'] != 'NULL')].copy()
    
    if df_valid.empty:
        return None
    
    # Extract day and month
    df_valid['Day'] = df_valid['ActivityDate'].dt.day
    df_valid['Month'] = df_valid['ActivityDate'].dt.month
    df_valid['DayName'] = df_valid['ActivityDate'].dt.day_name()
    
    # Create pivot table
    daily_activities = df_valid.groupby(['DayName', 'Month'])['TotalActivities'].sum().unstack(fill_value=0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_activities = daily_activities.reindex(day_order)
    
    fig = px.imshow(
        daily_activities,
        title="📅 Daily Activity Heatmap (by Month)",
        color_continuous_scale='Blues',
        aspect='auto'
    )
    
    fig.update_layout(height=400)
    return fig

def create_village_coverage_analysis(df):
    """Analyze village coverage"""
    if df.empty:
        return None, None
    
    # Get unique villages visited
    villages = []
    for col in ['GTV1VillageName', 'GTV2VillageName']:
        if col in df.columns:
            village_list = df[col].dropna()
            village_list = village_list[village_list != 'NULL']
            villages.extend(village_list.tolist())
    
    if not villages:
        return None, None
    
    village_counts = pd.Series(villages).value_counts().head(10)
    
    # Village coverage chart
    fig1 = px.bar(
        x=village_counts.values,
        y=village_counts.index,
        orientation='h',
        title="🏘️ Top 10 Most Visited Villages",
        labels={'x': 'Number of Visits', 'y': 'Village Name'}
    )
    fig1.update_layout(height=400)
    
    # Employee-wise village coverage
    df_filtered = df[df['KAName'].notna() & (df['KAName'] != 'NULL')].copy()
    
    if not df_filtered.empty:
        employee_villages = df_filtered.groupby('KAName')['TotalActivities'].sum().nlargest(10)
        
        fig2 = px.bar(
            x=employee_villages.values,
            y=employee_villages.index,
            orientation='h',
            title="👤 Top 10 Employees by Village Activities",
            labels={'x': 'Total Activities', 'y': 'Employee Name'}
        )
        fig2.update_layout(height=400)
        
        return fig1, fig2
    
    return fig1, None

def main():
    st.markdown('<h1 class="main-header">🚀 Employee Performance Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            df = load_and_process_data(uploaded_file)
        
        if df is not None:
            # Analyze employee performance
            employee_analysis = analyze_employee_performance(df)
            
            # Display metrics
            st.subheader("📊 Key Performance Metrics")
            create_performance_metrics(df, employee_analysis)
            
            # Top Performers Analysis
            st.subheader("🏆 Top Performers Analysis")
            if not employee_analysis.empty:
                top_perf_fig, efficient_workers, hard_workers = create_top_performers_analysis(employee_analysis)
                
                if top_perf_fig:
                    st.plotly_chart(top_perf_fig, use_container_width=True)
                
                # Display top performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Most Efficient Workers (Less Travel, More Work)")
                    if efficient_workers is not None and not efficient_workers.empty:
                        st.dataframe(efficient_workers, use_container_width=True)
                
                with col2:
                    st.markdown("### 💪 Hardest Workers (Most Activities)")
                    if hard_workers is not None and not hard_workers.empty:
                        st.dataframe(hard_workers, use_container_width=True)
            
            # Time Distribution Analysis
            st.subheader("⏰ Work Duration Distribution")
            time_dist_fig = create_time_distribution_chart(df)
            if time_dist_fig:
                st.plotly_chart(time_dist_fig, use_container_width=True)
            
            # Low Performers Analysis
            st.subheader("⚠️ Employees Working Less Than 1.5 Hours Per Day")
            low_performers = analyze_low_performers(employee_analysis)
            if not low_performers.empty:
                st.dataframe(low_performers, use_container_width=True)
                st.warning(f"⚠️ {len(low_performers)} employees are working less than 1.5 hours per day on average")
            else:
                st.success("✅ All employees are working more than 1.5 hours per day!")
            
            # Village Coverage Analysis
            st.subheader("🏘️ Village Coverage Analysis")
            village_fig1, village_fig2 = create_village_coverage_analysis(df)
            
            col1, col2 = st.columns(2)
            with col1:
                if village_fig1:
                    st.plotly_chart(village_fig1, use_container_width=True)
            
            with col2:
                if village_fig2:
                    st.plotly_chart(village_fig2, use_container_width=True)
            
            # Daily Activity Heatmap
            st.subheader("📅 Daily Activity Pattern")
            heatmap_fig = create_daily_activity_heatmap(df)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Additional Insights
            st.subheader("💡 Additional Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distance vs Activities scatter
                if not employee_analysis.empty:
                    fig = px.scatter(
                        employee_analysis,
                        x='Distance_Traveled',
                        y='Total_Activities',
                        size='Total_Hours',
                        color='Efficiency_Score',
                        hover_data=['KAName'],
                        title="🚗 Distance vs Activities Relationship"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sunday work analysis
                if not employee_analysis.empty and 'Sunday_Work_Days' in employee_analysis.columns:
                    sunday_workers = employee_analysis[employee_analysis['Sunday_Work_Days'] > 0]
                    if not sunday_workers.empty:
                        fig = px.bar(
                            sunday_workers.head(10),
                            x='Sunday_Work_Days',
                            y='KAName',
                            orientation='h',
                            title="📅 Sunday Work Champions"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Comprehensive Employee Summary
            st.subheader("👥 Complete Employee Performance Summary")
            if not employee_analysis.empty:
                # Add rank column
                employee_analysis['Rank'] = range(1, len(employee_analysis) + 1)
                
                # Reorder columns for better presentation
                display_cols = ['Rank', 'KAName', 'Total_Activities', 'Total_Hours', 'Days_Worked', 
                              'Distance_Traveled', 'Activities_Per_Hour', 'Activities_Per_KM', 
                              'Efficiency_Score', 'Total_Reimbursement', 'Sunday_Work_Days']
                
                summary_df = employee_analysis[display_cols].copy()
                st.dataframe(summary_df, use_container_width=True, height=400)
                
                # Download button
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Performance Report (CSV)",
                    data=csv,
                    file_name="employee_performance_report.csv",
                    mime="text/csv"
                )
            
            # Key Insights Summary
            st.subheader("🔍 Key Insights for HR")
            
            if not employee_analysis.empty:
                top_performer = employee_analysis.iloc[0]
                total_employees = len(employee_analysis)
                avg_efficiency = employee_analysis['Efficiency_Score'].mean()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="top-performer">
                        <h4>🥇 Top Performer</h4>
                        <p><strong>{top_performer['KAName']}</strong></p>
                        <p>Activities: {int(top_performer['Total_Activities'])}</p>
                        <p>Efficiency: {top_performer['Efficiency_Score']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    high_performers = len(employee_analysis[employee_analysis['Efficiency_Score'] > avg_efficiency])
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>📈 Performance Distribution</h4>
                        <p><strong>{high_performers}/{total_employees}</strong> employees are above average efficiency</p>
                        <p>Average Efficiency Score: <strong>{avg_efficiency:.2f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    sunday_workers_count = len(employee_analysis[employee_analysis['Sunday_Work_Days'] > 0])
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>📅 Dedication Level</h4>
                        <p><strong>{sunday_workers_count}</strong> employees worked on Sundays</p>
                        <p>Shows high commitment to work</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.info("👆 Please upload your Excel file to begin analysis")
        
        # Show expected data structure
        st.subheader("📋 Expected Data Structure")
        st.write("Your Excel file should contain these columns:")
        
        expected_columns = [
            "ActivityDate", "KACode", "StartTravelTime", "EndTravelTime", "KmByKA", 
            "KmBySystem", "Attendance", "GTV1PunchInTime", "GTV1PunchOutTime", 
            "GTV1VillageName", "GTV1VilActCount", "GTV1Attendance", "GTV2PunchInTime", 
            "GTV2PunchOutTime", "GTV2VillageName", "GTV2VilActCount", "GTV2Attendance", 
            "MarketActCount", "KMReimbursement", "DA", "StartKM", "EndKM", "IsSunday", "KAName"
        ]
        
        cols_df = pd.DataFrame({"Column Names": expected_columns})
        st.dataframe(cols_df, use_container_width=True)

if __name__ == "__main__":
    main()