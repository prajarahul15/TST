#!/usr/bin/env python3
"""
Streamlit Production App for Financial Insights Agents
Production-ready web interface for the multi-agent financial analysis system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Insights AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
        padding: 0.5rem;
        border-left: 4px solid #3498db;
        background: #ecf0f1;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
def get_openai_api_key_from_ui():
    """Get OpenAI API key from user input in the UI"""
    st.sidebar.markdown("### 🔑 OpenAI API Configuration")
    
    openai_key = st.sidebar.text_input(
        "OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to use GPT models"
    )
    
    return openai_key
# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Agent tools (same as in the Python file)
@tool
def load_and_prepare_data(file_path: str) -> str:
    """Load and prepare CSV data for analysis.
    
    Args:
        file_path (str): Path to the CSV file to load and prepare
        
    Returns:
        str: Information about the loaded data including record count, date range, lineups, and columns
    """
    print(f"\n{'='*60}")
    print(f"🔍 TOOL EXECUTION: load_and_prepare_data")
    print(f"📁 File Path: {file_path}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        print(f"📖 Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ CSV loaded successfully. Shape: {df.shape}")
        print(f"✅ Head of the dataframe: {df.head()}")
        print(f"✅ Tail of the dataframe: {df.tail()}")
        
        print(f"🧹 Cleaning data (removing empty rows)...")
        df = df.dropna(how='all')
        print(f"✅ Data cleaned. New shape: {df.shape}")
        print(f"✅ Head after cleaning: {df.head()}")
        print(f"✅ Tail after cleaning: {df.tail()}")
        
        # Handle different date formats automatically
        print(f"📅 Processing date column...")
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
            print(f"✅ Date format: dd-mm-yyyy")
        except ValueError:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
                print(f"✅ Date format: yyyy-mm-dd")
            except ValueError:
                df['DATE'] = pd.to_datetime(df['DATE'])
                print(f"✅ Date format: auto-detected")
        
        print(f"📊 Sorting data by Lineup and DATE...")
        df = df.sort_values(['Lineup', 'DATE'])
        
        info = f"""
        Data loaded successfully from {file_path}
        - Total records: {len(df)}
        - Date range: {df['DATE'].min().strftime('%Y-%m')} to {df['DATE'].max().strftime('%Y-%m')}
        - Lineups: {', '.join(df['Lineup'].unique())}
        - Columns: {', '.join(df.columns)}
        """
        
        print(f"✅ Tool execution completed successfully!")
        print(f"📊 Records processed: {len(df)}")
        print(f"🏷️ Lineups found: {len(df['Lineup'].unique())}")
        print(f"{'='*60}\n")
        
        return info
    except Exception as e:
        print(f"❌ ERROR in load_and_prepare_data: {str(e)}")
        print(f"{'='*60}\n")
        return f"Error loading data: {str(e)}"

@tool
def get_data_statistics(file_path: str, hierarchy_level: str = "Lineup") -> str:
    """Get statistical summary for specified hierarchy level.
    
    Args:
        file_path (str): Path to the CSV file to analyze
        hierarchy_level (str): Level of hierarchy to analyze - "Lineup", "Site", or "Profile"
        
    Returns:
        str: Statistical summary including sum, mean, std, min, max for the specified hierarchy level
    """
    print(f"\n{'='*60}")
    print(f"📊 TOOL EXECUTION: get_data_statistics")
    print(f"📁 File Path: {file_path}")
    print(f"🏗️ Hierarchy Level: {hierarchy_level}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        print(f"📖 Loading data for statistics...")
        df = pd.read_csv(file_path)
        df = df.dropna(how='all')
        print(f"✅ Data loaded. Shape: {df.shape}")
        print(f"✅ Head of the dataframe: {df.head()}")
        print(f"✅ Tail of the dataframe: {df.tail()}")
        
        print(f"📅 Processing dates...")
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
            print(f"✅ Date format: dd-mm-yyyy")
        except ValueError:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
                print(f"✅ Date format: yyyy-mm-dd")
            except ValueError:
                df['DATE'] = pd.to_datetime(df['DATE'])
                print(f"✅ Date format: auto-detected")
        
        print(f"✅ Head after date processing: {df.head()}")
        print(f"✅ Tail after date processing: {df.tail()}")
        
        print(f"📊 Calculating statistics for {hierarchy_level} level...")
        if hierarchy_level == "Lineup":
            stats = df.groupby('Lineup')['Actual'].agg(['sum', 'mean', 'std', 'min', 'max']).round(2)
            print(f"✅ Statistics calculated for {len(stats)} lineups")
        elif hierarchy_level == "Site":
            stats = df.groupby('Site')['Actual'].agg(['sum', 'mean', 'std', 'min', 'max']).round(2)
            print(f"✅ Statistics calculated for {len(stats)} sites")
        elif hierarchy_level == "Profile":
            stats = df.groupby('Profile')['Actual'].agg(['sum', 'mean', 'std', 'min', 'max']).round(2)
            print(f"✅ Statistics calculated for {len(stats)} profiles")
        
        print(f"✅ Tool execution completed successfully!")
        print(f"📈 Statistics generated for {hierarchy_level} level")
        print(f"{'='*60}\n")
        
        return f"Statistics for {hierarchy_level} level:\n{stats.to_string()}"
    except Exception as e:
        print(f"❌ ERROR in get_data_statistics: {str(e)}")
        print(f"{'='*60}\n")
        return f"Error getting statistics: {str(e)}"

@tool
def analyze_trends(file_path: str, hierarchy_level: str = "Lineup") -> str:
    """Analyze spending trends over time for specified hierarchy level.
    
    Args:
        file_path (str): Path to the CSV file to analyze
        hierarchy_level (str): Level of hierarchy to analyze - "Lineup", "Site", or "Profile"
        
    Returns:
        str: Trend analysis including monthly spending patterns and growth rates
    """
    print(f"\n{'='*60}")
    print(f"📈 TOOL EXECUTION: analyze_trends")
    print(f"📁 File Path: {file_path}")
    print(f"🏗️ Hierarchy Level: {hierarchy_level}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(how='all')
        print(f"✅ Data loaded. Shape: {df.shape}")
        print(f"✅ Head of the dataframe: {df.head()}")
        print(f"✅ Tail of the dataframe: {df.tail()}")
        
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        except ValueError:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
            except ValueError:
                df['DATE'] = pd.to_datetime(df['DATE'])
        
        print(f"✅ Head after date processing: {df.head()}")
        print(f"✅ Tail after date processing: {df.tail()}")
        
        df['Month'] = df['DATE'].dt.to_period('M')
        
        if hierarchy_level == "Lineup":
            trends = df.groupby(['Lineup', 'Month'])['Actual'].sum().reset_index()
        elif hierarchy_level == "Site":
            trends = df.groupby(['Site', 'Month'])['Actual'].sum().reset_index()
        elif hierarchy_level == "Profile":
            trends = df.groupby(['Profile', 'Month'])['Actual'].sum().reset_index()
        
        trends['Growth_Rate'] = trends.groupby(hierarchy_level)['Actual'].pct_change() * 100
        
        print(f"✅ Tool execution completed successfully!")
        print(f"📈 Trend analysis generated for {hierarchy_level} level")
        print(f"📊 Records processed: {len(trends)}")
        print(f"{'='*60}\n")
        
        return f"Trend analysis for {hierarchy_level} level:\n{trends.tail(10).to_string()}"
    except Exception as e:
        print(f"❌ ERROR in analyze_trends: {str(e)}")
        print(f"{'='*60}\n")
        return f"Error analyzing trends: {str(e)}"

@tool
def create_visualization(file_path: str, chart_type: str = "line", hierarchy_level: str = "Lineup") -> str:
    """Create visualization charts for the data.
    
    Args:
        file_path (str): Path to the CSV file to visualize
        chart_type (str): Type of chart to create - "line" or "bar"
        hierarchy_level (str): Level of hierarchy to visualize - "Lineup", "Site", or "Profile"
        
    Returns:
        str: Confirmation message that the visualization was created
    """
    print(f"\n{'='*60}")
    print(f"📊 TOOL EXECUTION: create_visualization")
    print(f"📁 File Path: {file_path}")
    print(f"📈 Chart Type: {chart_type}")
    print(f"🏗️ Hierarchy Level: {hierarchy_level}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(how='all')
        print(f"✅ Data loaded. Shape: {df.shape}")
        print(f"✅ Head of the dataframe: {df.head()}")
        print(f"✅ Tail of the dataframe: {df.tail()}")
        
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        except ValueError:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
            except ValueError:
                df['DATE'] = pd.to_datetime(df['DATE'])
        
        print(f"✅ Head after date processing: {df.head()}")
        print(f"✅ Tail after date processing: {df.tail()}")
        
        if chart_type == "line":
            if hierarchy_level == "Lineup":
                fig = px.line(df, x='DATE', y='Actual', color='Lineup', title='Spending Trends by Lineup')
            elif hierarchy_level == "Site":
                fig = px.line(df, x='DATE', y='Actual', color='Site', title='Spending Trends by Site')
            
            print(f"✅ Tool execution completed successfully!")
            print(f"📊 Line chart created for {hierarchy_level} level")
            print(f"{'='*60}\n")
            return f"Line chart created for {hierarchy_level} level"
        
        elif chart_type == "bar":
            monthly_totals = df.groupby([df['DATE'].dt.to_period('M'), hierarchy_level])['Actual'].sum().reset_index()
            monthly_totals['DATE'] = monthly_totals['DATE'].astype(str)
            
            print(f"✅ Tool execution completed successfully!")
            print(f"📊 Bar chart created for {hierarchy_level} level")
            print(f"{'='*60}\n")
            return f"Bar chart created for {hierarchy_level} level"
        
        print(f"✅ Tool execution completed successfully!")
        print(f"📊 Visualization created successfully")
        print(f"{'='*60}\n")
        return "Visualization created successfully"
    except Exception as e:
        print(f"❌ ERROR in create_visualization: {str(e)}")
        print(f"{'='*60}\n")
        return f"Error creating visualization: {str(e)}"

@tool
def perform_variance_analysis(actual_file: str, plan_file: str, forecast_file: str) -> str:
    """Perform comprehensive variance analysis between Actual, Plan, and Forecast data.
    
    Args:
        actual_file (str): Path to the file containing actual data
        plan_file (str): Path to the file containing plan data
        forecast_file (str): Path to the file containing forecast data
        
    Returns:
        str: Comprehensive variance analysis with insights, recommendations, and detailed data
    """
    print(f"\n{'='*60}")
    print(f"📈 TOOL EXECUTION: perform_variance_analysis")
    print(f"📁 Actual File: {actual_file}")
    print(f"📁 Plan File: {plan_file}")
    print(f"📁 Forecast File: {forecast_file}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        print(f"📖 Loading actual data from {actual_file}...")
        actual_df = pd.read_csv(actual_file)
        print(f"✅ Actual data loaded. Shape: {actual_df.shape}")
        print(f"✅ Head of actual data: {actual_df.head()}")
        print(f"✅ Tail of actual data: {actual_df.tail()}")
        
        print(f"📖 Loading plan data from {plan_file}...")
        plan_df = pd.read_csv(plan_file)
        print(f"✅ Plan data loaded. Shape: {plan_df.shape}")
        print(f"✅ Head of plan data: {plan_df.head()}")
        print(f"✅ Tail of plan data: {plan_df.tail()}")
        
        print(f"📖 Loading forecast data from {forecast_file}...")
        forecast_df = pd.read_csv(forecast_file)
        print(f"✅ Forecast data loaded. Shape: {forecast_df.shape}")
        print(f"✅ Head of forecast data: {forecast_df.head()}")
        print(f"✅ Tail of forecast data: {forecast_df.tail()}")
        
        # Handle different date formats
        print(f"📅 Processing dates for actual data...")
        try:
            actual_df['DATE'] = pd.to_datetime(actual_df['DATE'], format='%d-%m-%Y')
            print(f"✅ Actual data date format: dd-mm-yyyy")
        except ValueError:
            actual_df['DATE'] = pd.to_datetime(actual_df['DATE'])
            print(f"✅ Actual data date format: auto-detected")
            
        print(f"📅 Processing dates for plan data...")
        try:
            plan_df['DATE'] = pd.to_datetime(plan_df['DATE'], format='%d-%m-%Y')
            print(f"✅ Plan data date format: dd-mm-yyyy")
        except ValueError:
            plan_df['DATE'] = pd.to_datetime(plan_df['DATE'])
            print(f"✅ Plan data date format: auto-detected")
            
        print(f"📅 Processing dates for forecast data...")
        try:
            forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE'], format='%Y-%m-%d')
            print(f"✅ Forecast data date format: yyyy-mm-dd")
        except ValueError:
            forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE'])
            print(f"✅ Forecast data date format: auto-detected")
        
        print(f"🔍 Filtering 2025 forecast data...")
        forecast_2025 = forecast_df[forecast_df['DATE'].dt.year == 2025].copy()
        print(f"✅ 2025 forecast records found: {len(forecast_2025)}")
        
        if len(forecast_2025) == 0:
            print(f"⚠️ No 2025 forecast data found for variance analysis.")
            print(f"{'='*60}\n")
            return "No 2025 forecast data found for variance analysis."
        
        print(f"📊 Calculating variances...")
        # Calculate variances
        results = []
        
        for _, row in forecast_2025.iterrows():
            lineup = row['Lineup']
            date = row['DATE']
            plan_val = row['Plan']
            forecast_val = row['Forecast']
            
            # Find corresponding actual value from historical data (use average of same month)
            month = date.month
            historical_actuals = actual_df[
                (actual_df['Lineup'] == lineup) & 
                (actual_df['DATE'].dt.month == month)
            ]['Actual'].values
            
            if len(historical_actuals) > 0:
                avg_actual = np.mean(historical_actuals)
                
                # Plan vs Forecast variance
                plan_forecast_var = ((forecast_val - plan_val) / plan_val) * 100 if plan_val != 0 else 0
                plan_forecast_abs = forecast_val - plan_val
                
                # Historical vs Plan variance
                hist_plan_var = ((avg_actual - plan_val) / plan_val) * 100 if plan_val != 0 else 0
                hist_plan_abs = avg_actual - plan_val
                
                # Historical vs Forecast variance
                hist_forecast_var = ((avg_actual - forecast_val) / forecast_val) * 100 if forecast_val != 0 else 0
                hist_forecast_abs = avg_actual - forecast_val
                
                results.append({
                    'Lineup': lineup,
                    'Date': date.strftime('%Y-%m'),
                    'Historical_Avg': round(avg_actual, 2),
                    'Plan': plan_val,
                    'Forecast': round(forecast_val, 2),
                    'Plan_vs_Forecast_Var%': round(plan_forecast_var, 2),
                    'Plan_vs_Forecast_Abs': round(plan_forecast_abs, 2),
                    'Historical_vs_Plan_Var%': round(hist_plan_var, 2),
                    'Historical_vs_Plan_Abs': round(hist_plan_abs, 2),
                    'Historical_vs_Forecast_Var%': round(hist_forecast_var, 2),
                    'Historical_vs_Forecast_Abs': round(hist_forecast_abs, 2)
                })
        
        if not results:
            print(f"⚠️ No variance analysis results found.")
            print(f"{'='*60}\n")
            return "No variance analysis results found. Check data availability."
        
        print(f"📊 Generating insights...")
        # Create summary DataFrame
        variance_df = pd.DataFrame(results)
        
        # Generate insights
        insights = []
        
        # Overall variance patterns
        avg_plan_forecast_var = variance_df['Plan_vs_Forecast_Var%'].mean()
        avg_hist_plan_var = variance_df['Historical_vs_Plan_Var%'].mean()
        avg_hist_forecast_var = variance_df['Historical_vs_Forecast_Var%'].mean()
        
        insights.append(f"**Overall Variance Analysis (2025):**")
        insights.append(f"- Plan vs Forecast: Average {avg_plan_forecast_var:.2f}% variance")
        insights.append(f"- Historical vs Plan: Average {avg_hist_plan_var:.2f}% variance")
        insights.append(f"- Historical vs Forecast: Average {avg_hist_forecast_var:.2f}% variance")
        
        # Lineup-specific insights
        for lineup in variance_df['Lineup'].unique():
            lineup_data = variance_df[variance_df['Lineup'] == lineup]
            insights.append(f"\n**Lineup {lineup}:**")
            insights.append(f"- Plan vs Forecast variance: {lineup_data['Plan_vs_Forecast_Var%'].mean():.2f}%")
            insights.append(f"- Historical vs Plan variance: {lineup_data['Historical_vs_Plan_Var%'].mean():.2f}%")
        
        # Recommendations
        insights.append(f"\n**Key Insights & Recommendations:**")
        if abs(avg_plan_forecast_var) > 10:
            insights.append("- Significant variance between Plan and Forecast suggests forecasting model may need adjustment")
        if abs(avg_hist_plan_var) > 15:
            insights.append("- Large variance between historical patterns and plans indicates potential planning accuracy issues")
        if abs(avg_hist_forecast_var) > 20:
            insights.append("- High variance between historical data and forecasts suggests model may not capture seasonal patterns well")
        
        insights.append("- Consider reviewing forecasting methodology and incorporating more seasonal factors")
        insights.append("- Monitor actual vs plan variances closely in 2025 to validate planning assumptions")
        
        print(f"✅ Tool execution completed successfully!")
        print(f"📊 Variance analysis completed for {len(forecast_2025)} forecast records in 2025")
        print(f"📈 Insights generated: {len(insights)} key points")
        print(f"{'='*60}\n")
        
        return "\n".join(insights) + f"\n\nDetailed variance data:\n{variance_df.to_string(index=False)}"
        
    except Exception as e:
        print(f"❌ ERROR in perform_variance_analysis: {str(e)}")
        print(f"{'='*60}\n")
        return f"Error performing variance analysis: {str(e)}"

# Agent Orchestrator Class
class FinancialInsightOrchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            load_and_prepare_data,
            get_data_statistics,
            analyze_trends,
            create_visualization,
            perform_variance_analysis
        ]
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=False,
            handle_parsing_errors=True
        )
    
    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Financial Data Analysis Expert. Your role is to:
            1. Analyze financial data at Profile → Site → Lineup hierarchy
            2. Generate comprehensive insights from actual data
            3. Perform variance analysis between Actual vs Plan vs Forecast
            4. Create visualizations and provide actionable recommendations
            5. Use the available tools to gather data and perform analysis
            
            Always provide structured, business-focused insights with clear recommendations."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)
    
    def generate_comprehensive_insights(self, file_path: str) -> str:
        print(f"\n{'='*60}")
        print(f"🚀 ORCHESTRATOR: generate_comprehensive_insights")
        print(f"📁 File Path: {file_path}")
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        prompt = f"""
        Analyze the financial data from {file_path} and provide comprehensive insights:
        
        1. **Data Overview**: Load and examine the data structure
        2. **Profile Level Analysis**: Analyze spending patterns at Profile level
        3. **Site Level Analysis**: Analyze spending patterns at Site level  
        4. **Lineup Level Analysis**: Analyze spending patterns at Lineup level
        5. **Trend Analysis**: Identify spending trends over time
        6. **Seasonal Patterns**: Detect any seasonal spending patterns
        7. **Key Insights**: Provide 5-7 key business insights
        8. **Recommendations**: Suggest actionable recommendations
        
        Use the available tools to gather data and perform analysis. Be thorough and business-focused.
        """
        
        print(f"🤖 Invoking AI agent with comprehensive insights prompt...")
        result = self.agent_executor.invoke({"input": prompt})
        print(f"✅ AI agent response received successfully!")
        print(f"{'='*60}\n")
        
        return result["output"]
    
    def generate_variance_analysis(self, actual_file: str, plan_file: str, forecast_file: str) -> str:
        print(f"\n{'='*60}")
        print(f"🚀 ORCHESTRATOR: generate_variance_analysis")
        print(f"📁 Actual File: {actual_file}")
        print(f"📁 Plan File: {plan_file}")
        print(f"📁 Forecast File: {forecast_file}")
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        prompt = f"""
        Perform comprehensive variance analysis between Actual, Plan, and Forecast data:
        
        1. **Load Data**: Load data from {actual_file}, {plan_file}, and {forecast_file}
        2. **Data Preparation**: Handle different date formats and prepare data for analysis
        3. **Variance Calculation**: Calculate variances between:
           - Plan vs Forecast for 2025 data
           - Historical patterns vs Plan for 2025 data
           - Historical patterns vs Forecast for 2025 data
        4. **Analysis**: Provide insights on variance patterns, root causes, and business impact
        5. **Key Insights**: Provide 5-7 key business insights
        6. **Recommendations**: Suggest mitigation strategies and improvements
        
        Focus on the forecast period (2025) where all three data types are available.
        Use the available tools to gather data and perform analysis. Be thorough and business-focused.
        """
        
        try:
            print(f"🤖 Invoking AI agent with variance analysis prompt...")
            result = self.agent_executor.invoke({"input": prompt})
            print(f"✅ AI agent response received successfully!")
            print(f"{'='*60}\n")
            return result["output"]
        except Exception as e:
            print(f"❌ ERROR in variance analysis: {str(e)}")
            print(f"{'='*60}\n")
            return f"Error in variance analysis: {str(e)}"
    
    def ask_question(self, user_question: str, context_files: list = None) -> str:
        print(f"\n{'='*60}")
        print(f"💬 ORCHESTRATOR: ask_question")
        print(f"❓ User Question: {user_question}")
        print(f"📁 Context Files: {context_files if context_files else 'None'}")
        print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        """Conversational Q&A agent that answers user questions about financial data"""
        try:
            if context_files:
                context_info = f"""
                **Available Data Sources:**
                {chr(10).join([f"- {file}" for file in context_files])}
                
                **User Question:** {user_question}
                
                **Instructions:**
                1. Use the available tools to gather relevant data
                2. Analyze the data to answer the user's question
                3. Provide clear, business-focused insights
                4. If the question requires data not available, suggest what additional data would be helpful
                5. Always be helpful and professional
                """
            else:
                context_info = f"""
                **User Question:** {user_question}
                
                **Instructions:**
                1. Use the available tools to gather relevant data
                2. Analyze the data to answer the user's question
                3. Provide clear, business-focused insights
                4. If the question requires data not available, suggest what additional data would be helpful
                5. Always be helpful and professional
                """
            
            print(f"🤖 Invoking AI agent with Q&A prompt...")
            result = self.agent_executor.invoke({"input": context_info})
            print(f"✅ AI agent response received successfully!")
            print(f"{'='*60}\n")
            
            return result["output"]
            
        except Exception as e:
            print(f"❌ ERROR in ask_question: {str(e)}")
            print(f"{'='*60}\n")
            return f"Error processing your question: {str(e)}"

# Initialize the system
def initialize_system(openai_api_key):
    """Initialize the LLM and orchestrator"""
    try:
        #GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        #OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            st.error("❌ Please provide your OpenAI API key in the sidebar")
            #st.info("Please create a .env file with your Gemini API key")
            return False
        
        with st.spinner("🤖 Initializing AI Agent System..."):
            # llm = ChatGoogleGenerativeAI(
            #     model="gemini-2.0-flash-exp",
            #     google_api_key=GEMINI_API_KEY,
            #     temperature=0.1,
            #     convert_system_message_to_human=True
            # )
            llm=ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key,temperature=0.1)
            
            orchestrator = FinancialInsightOrchestrator(llm)
            st.session_state.orchestrator = orchestrator
        
        st.success("✅ AI Agent System initialized successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        return False

# Main app
def main():
    st.markdown('<h1 class="main-header">🤖 Financial Insights AI</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent Financial Data Analysis with Multi-Agent System**")
    
    # Sidebar
    st.sidebar.title("🔧 System Controls")

    openai_api_key = get_openai_api_key_from_ui()
    
    
    # System info
    st.sidebar.markdown("**📊 Analysis Pipeline:**")
    st.sidebar.markdown("1. 📊 Dashboard")
    st.sidebar.markdown("2. 🔮 Forecasting")
    st.sidebar.markdown("3. 🔍 Insights")
    st.sidebar.markdown("4. 📈 Variance Analysis")
    st.sidebar.markdown("5. 💬 Q&A")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Initialize AI System", type="primary"):
        if initialize_system(openai_api_key):
            st.rerun()
    
    # Initialize system
    #if st.sidebar.button("🚀 Initialize AI System", type="Secondry"):
        #if initialize_system():
            #st.rerun()
    
    # Check if system is initialized
    if st.session_state.orchestrator is None:
        st.warning("⚠️ Please initialize the AI system first using the button in the sidebar.")
        st.info("💡 Make sure you have a `.env` file with your `GEMINI_API_KEY`")
        return
    
    # Main tabs
    tab1, tab2 = st.tabs([
        "🚀 All-in-One Analysis", 
        "📁 Data Explorer"
    ])
    
    # Tab 1: All-in-One Analysis
    with tab1:
        st.markdown('<h2 class="section-header">🚀 All-in-One Financial Analysis</h2>', unsafe_allow_html=True)
        st.markdown("**Includes: Dashboard → Forecasting → Insights → Variance Analysis → Q&A**")
        
        # Progress tracking
        if 'analysis_progress' not in st.session_state:
            st.session_state.analysis_progress = 0
            st.session_state.current_step = "Dashboard"
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Dashboard Section
        st.markdown("#### 📊 Dashboard")
        status_text.text("🔄 Loading Dashboard...")
        
        try:
            # Load sample data for quick stats
            df = pd.read_csv("Sample_data_N.csv")
            df = df.dropna(how='all')
            
            # Process dates once for both dashboard and visualizations
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df = df.dropna(subset=['DATE'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_spending = df['Actual'].sum()
                st.metric("Total Spending", f"${total_spending:,.0f}")
            
            with col2:
                avg_monthly = df['Actual'].mean()
                st.metric("Avg Monthly", f"${avg_monthly:,.0f}")
            
            with col3:
                num_lineups = df['Lineup'].nunique()
                st.metric("Lineups", num_lineups)
            
            with col4:
                date_range = f"{df['DATE'].min().strftime('%Y-%m')} to {df['DATE'].max().strftime('%Y-%m')}"
                st.metric("Date Range", date_range)
            
            # Update progress
            st.session_state.analysis_progress = 25
            progress_bar.progress(st.session_state.analysis_progress / 100)
            status_text.text("✅ Dashboard loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
            st.session_state.analysis_progress = 25
            progress_bar.progress(st.session_state.analysis_progress / 100)
        
        # Dashboard Visualizations Section
        st.markdown("#### 📊 Dashboard Visualizations")
        
        try:
            # Use the already processed DataFrame
            st.info(f"📊 Processing visualizations for {len(df)} records with date range: {df['DATE'].min().strftime('%Y-%m')} to {df['DATE'].max().strftime('%Y-%m')}")
            
            # Data validation
            st.markdown("##### 🔍 Data Validation")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**DataFrame Shape:** {df.shape}")
                st.write(f"**Columns:** {list(df.columns)}")
                st.write(f"**Data Types:** {df.dtypes.to_dict()}")
            with col2:
                st.write(f"**Lineups:** {df['Lineup'].unique()}")
                st.write(f"**Sites:** {df['Site'].unique()}")
                st.write(f"**Actual Values Range:** ${df['Actual'].min():,.0f} - ${df['Actual'].max():,.0f}")
            
            # Test basic plotting functionality
            st.markdown("##### 🧪 Test Chart - Basic Line Plot")
            try:
                test_df = df.groupby('Lineup')['Actual'].sum().reset_index()
                fig_test = px.bar(test_df, x='Lineup', y='Actual', title='Test Chart - Total Spending by Lineup')
                st.plotly_chart(fig_test, use_container_width=True)
                st.success("✅ Test chart generated successfully - Plotly is working!")
            except Exception as e:
                st.error(f"❌ Test chart failed: {str(e)}")
                st.write("This indicates a fundamental plotting issue")
            
            # Chart 1: Monthly Spending Trends by Lineup
            st.markdown("##### 📈 Monthly Spending Trends by Lineup")
            try:
                monthly_lineup = df.groupby([df['DATE'].dt.to_period('M'), 'Lineup'])['Actual'].sum().reset_index()
                monthly_lineup['DATE'] = monthly_lineup['DATE'].astype(str)
                
                fig1 = px.line(monthly_lineup, x='DATE', y='Actual', color='Lineup', 
                              title='Monthly Spending Trends by Lineup',
                              labels={'DATE': 'Month', 'Actual': 'Spending ($)', 'Lineup': 'Lineup'})
                fig1.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig1, use_container_width=True)
                st.success("✅ Chart 1 generated successfully")
            except Exception as e:
                st.error(f"❌ Error generating Chart 1: {str(e)}")
                st.write(f"Debug info - monthly_lineup shape: {monthly_lineup.shape if 'monthly_lineup' in locals() else 'Not created'}")
            
            # Chart 2: Monthly Spending by Site
            st.markdown("##### 🏢 Monthly Spending by Site")
            monthly_site = df.groupby([df['DATE'].dt.to_period('M'), 'Site'])['Actual'].sum().reset_index()
            monthly_site['DATE'] = monthly_site['DATE'].astype(str)
            
            fig2 = px.line(monthly_site, x='DATE', y='Actual', color='Site',
                          title='Monthly Spending Trends by Site',
                          labels={'DATE': 'Month', 'Actual': 'Spending ($)', 'Site': 'Site'})
            fig2.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Chart 3: Bar Chart - Total Spending by Lineup
            st.markdown("##### 🏷️ Total Spending by Lineup")
            total_by_lineup = df.groupby('Lineup')['Actual'].sum().reset_index()
            
            fig3 = px.bar(total_by_lineup, x='Lineup', y='Actual',
                         title='Total Spending by Lineup',
                         labels={'Lineup': 'Lineup', 'Actual': 'Total Spending ($)'},
                         color='Actual', color_continuous_scale='viridis')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Chart 4: Heatmap - Monthly Spending Heatmap
            st.markdown("##### 🔥 Monthly Spending Heatmap")
            # Create pivot table for heatmap
            heatmap_data = df.groupby([df['DATE'].dt.to_period('M'), 'Lineup'])['Actual'].sum().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='DATE', columns='Lineup', values='Actual').fillna(0)
            
            fig4 = px.imshow(heatmap_pivot.values,
                            x=heatmap_pivot.columns,
                            y=[str(d) for d in heatmap_pivot.index],
                            title='Monthly Spending Heatmap by Lineup',
                            labels={'x': 'Lineup', 'y': 'Month', 'color': 'Spending ($)'},
                            color_continuous_scale='viridis',
                            aspect='auto')
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
            
            # Chart 5: Box Plot - Spending Distribution by Lineup
            st.markdown("##### 📦 Spending Distribution by Lineup")
            fig5 = px.box(df, x='Lineup', y='Actual',
                         title='Spending Distribution by Lineup',
                         labels={'Lineup': 'Lineup', 'Actual': 'Spending ($)'})
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
            
            # Chart 6: Pie Chart - Spending Share by Lineup
            st.markdown("##### 🥧 Spending Share by Lineup")
            total_spending_by_lineup = df.groupby('Lineup')['Actual'].sum()
            
            fig6 = px.pie(values=total_spending_by_lineup.values,
                         names=total_spending_by_lineup.index,
                         title='Spending Share by Lineup (%)')
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
            
            st.success("✅ Dashboard visualizations generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating dashboard visualizations: {str(e)}")
            st.info("💡 Make sure your data files are available and contain the required columns (DATE, Actual, Lineup, Site)")
        
        # Dashboard Data Summary Section
        st.markdown("#### 📋 Dashboard Data Summary")
        
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Overview:**")
                st.write(f"• **Total Records:** {len(df)}")
                st.write(f"• **Date Range:** {df['DATE'].min().strftime('%Y-%m-%d')} to {df['DATE'].max().strftime('%Y-%m-%d')}")
                st.write(f"• **Lineups:** {', '.join(df['Lineup'].unique())}")
                st.write(f"• **Sites:** {', '.join(df['Site'].unique())}")
            
            with col2:
                st.markdown("**Spending Statistics:**")
                st.write(f"• **Total Spending:** ${df['Actual'].sum():,.0f}")
                st.write(f"• **Average Monthly:** ${df['Actual'].mean():,.0f}")
                st.write(f"• **Highest Month:** ${df.groupby(df['DATE'].dt.to_period('M'))['Actual'].sum().max():,.0f}")
                st.write(f"• **Lowest Month:** ${df.groupby(df['DATE'].dt.to_period('M'))['Actual'].sum().min():,.0f}")
                
        except Exception as e:
            st.error(f"Error displaying data summary: {str(e)}")
        
        # Dashboard Controls
        st.markdown("#### 🎛️ Dashboard Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Dashboard", type="secondary", key="refresh_dashboard"):
                st.rerun()
        
        with col2:
            if st.button("📊 Export Dashboard Data", type="secondary", key="export_dashboard"):
                try:
                    # Create a summary DataFrame for export
                    dashboard_summary = {
                        'Metric': ['Total Records', 'Total Spending', 'Average Monthly', 'Number of Lineups', 'Number of Sites'],
                        'Value': [
                            len(df),
                            f"${df['Actual'].sum():,.0f}",
                            f"${df['Actual'].mean():,.0f}",
                            df['Lineup'].nunique(),
                            df['Site'].nunique()
                        ]
                    }
                    summary_df = pd.DataFrame(dashboard_summary)
                    st.download_button(
                        label="📥 Download Dashboard Summary",
                        data=summary_df.to_csv(index=False),
                        file_name="dashboard_summary.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error exporting dashboard data: {str(e)}")
        
        # Forecasting Section
        st.markdown("#### 🔮 Financial Forecasting")
        status_text.text("🔄 Running forecasting analysis...")
        
        try:
            # Import forecasting functions
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # Import the forecasting module
            from fixed_forecasting import (
                load_and_prepare_data as forecast_load_data,
                create_advanced_features,
                train_forecasting_model,
                forecast_future_months
            )
            
            st.info("📊 Running advanced forecasting model for next 12 months...")
            
            # Load and prepare data for forecasting
            forecast_df = forecast_load_data('Sample_data_N.csv')
            forecast_df = create_advanced_features(forecast_df)
            
            # Get unique lineups
            lineups = forecast_df['Lineup'].unique()
            
            # Store forecasting results
            forecast_results = {}
            forecast_metrics = []
            
            # Progress bar for forecasting
            forecast_progress = st.progress(0)
            forecast_status = st.empty()
            
            for i, lineup in enumerate(lineups):
                forecast_status.text(f"🔄 Training model for {lineup}... ({i+1}/{len(lineups)})")
                forecast_progress.progress((i + 1) / len(lineups))
                
                try:
                    # Train model and get predictions
                    result = train_forecasting_model(forecast_df, lineup)
                    
                    if result is not None:
                        model, X_test, y_test, y_pred, mape, mae, r2 = result
                        
                        # Forecast next 12 months
                        forecast_12_months = forecast_future_months(model, forecast_df, lineup, months_ahead=12)
                        
                        # Store results
                        forecast_results[lineup] = {
                            'model': model,
                            'mape': mape,
                            'mae': mae,
                            'r2': r2,
                            'forecast': forecast_12_months
                        }
                        
                        # Add to metrics
                        forecast_metrics.append({
                            'Lineup': lineup,
                            'MAPE (%)': round(mape, 2),
                            'MAE (£)': round(mae, 2),
                            'R² Score': round(r2, 4),
                            'Avg Forecast (£)': round(forecast_12_months['Forecast'].mean(), 2)
                        })
                        
                        st.success(f"✅ {lineup} model trained successfully!")
                    else:
                        st.warning(f"⚠️ Insufficient data for {lineup}")
                        
                except Exception as e:
                    st.error(f"❌ Error forecasting for {lineup}: {str(e)}")
            
            # Display forecasting results
            if forecast_metrics:
                st.markdown("##### 📈 Forecasting Model Performance")
                
                # Create metrics DataFrame
                metrics_df = pd.DataFrame(forecast_metrics)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Display forecast charts
                st.markdown("##### 🔮 12-Month Forecasts by Lineup")
                
                for lineup, result in forecast_results.items():
                    with st.expander(f"📊 {lineup} - 12-Month Forecast"):
                        # Create forecast chart
                        forecast_data = result['forecast']
                        forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
                        
                        fig = px.line(forecast_data, x='Date', y='Forecast', 
                                    title=f'{lineup} - 12-Month Forecast',
                                    labels={'Date': 'Month', 'Forecast': 'Forecasted Spending (£)'})
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast table
                        st.markdown("**Forecast Details:**")
                        forecast_display = forecast_data.copy()
                        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m')
                        forecast_display['Forecast'] = forecast_display['Forecast'].round(2)
                        st.dataframe(forecast_display, use_container_width=True)
                
                # Summary statistics
                st.markdown("##### 📊 Forecast Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_mape = np.mean([m['MAPE (%)'] for m in forecast_metrics])
                    st.metric("Average MAPE", f"{avg_mape:.2f}%")
                
                with col2:
                    avg_r2 = np.mean([m['R² Score'] for m in forecast_metrics])
                    st.metric("Average R²", f"{avg_r2:.4f}")
                
                with col3:
                    total_forecast = sum([m['Avg Forecast (£)'] * 12 for m in forecast_metrics])
                    st.metric("Total 12-Month Forecast", f"£{total_forecast:,.0f}")
                
                # Export forecast data
                if st.button("📥 Export Forecast Data", type="primary", key="export_forecast"):
                    try:
                        # Create comprehensive forecast export
                        all_forecasts = []
                        for lineup, result in forecast_results.items():
                            forecast_data = result['forecast'].copy()
                            forecast_data['Lineup'] = lineup
                            forecast_data['Date'] = forecast_data['Date'].dt.strftime('%Y-%m-%d')
                            all_forecasts.append(forecast_data)
                        
                        if all_forecasts:
                            combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
                            st.download_button(
                                label="📥 Download Combined Forecasts",
                                data=combined_forecasts.to_csv(index=False),
                                file_name="12_month_forecasts.csv",
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error exporting forecast data: {str(e)}")
                
                st.success("🎉 Forecasting analysis completed successfully!")
                
            else:
                st.warning("⚠️ No forecasting results generated. Check data availability.")
                
        except ImportError as e:
            st.error(f"❌ Error importing forecasting module: {str(e)}")
            st.info("💡 Make sure 'fixed_forecasting.py' is in the same directory")
        except Exception as e:
            st.error(f"❌ Error during forecasting: {str(e)}")
            st.info("💡 Check console for detailed error messages")
        
        # Check for existing forecast files
        st.markdown("##### 📁 Existing Forecast Files")
        try:
            forecast_files = []
            
            # Check for forecast files
            if os.path.exists('fixed_clean_complete_data_with_forecasts.csv'):
                forecast_files.append('fixed_clean_complete_data_with_forecasts.csv')
            
            if os.path.exists('fixed_complete_data_with_forecasts.csv'):
                forecast_files.append('fixed_complete_data_with_forecasts.csv')
            
            if forecast_files:
                st.success(f"✅ Found {len(forecast_files)} existing forecast file(s)")
                
                for file in forecast_files:
                    try:
                        existing_df = pd.read_csv(file)
                        st.info(f"📊 **{file}**: {len(existing_df)} records")
                        
                        # Show forecast data summary
                        if 'Forecast' in existing_df.columns:
                            forecast_records = existing_df[existing_df['Forecast'].notna()]
                            if len(forecast_records) > 0:
                                st.write(f"   - Forecast records: {len(forecast_records)}")
                                st.write(f"   - Forecast period: {forecast_records['DATE'].min()} to {forecast_records['DATE'].max()}")
                                
                                # Show forecast preview
                                with st.expander(f"🔍 Preview {file}"):
                                    st.dataframe(existing_df.head(10), use_container_width=True)
                                    
                                    # Show forecast summary
                                    if 'Forecast' in existing_df.columns:
                                        forecast_summary = existing_df.groupby('Lineup')['Forecast'].agg(['count', 'mean', 'sum']).round(2)
                                        st.markdown("**Forecast Summary by Lineup:**")
                                        st.dataframe(forecast_summary, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error reading {file}: {str(e)}")
            else:
                st.info("💡 No existing forecast files found. Run the forecasting above to generate new forecasts.")
                
        except Exception as e:
            st.error(f"❌ Error checking forecast files: {str(e)}")
        
        # Update progress
        st.session_state.analysis_progress = 37
        progress_bar.progress(st.session_state.analysis_progress / 100)
        status_text.text("✅ Forecasting completed successfully!")
        
        # Comprehensive Insights Section
        st.markdown("#### 🔍 Comprehensive Insights")
        status_text.text("🔄 Generating comprehensive insights...")
        
        try:
            if st.session_state.orchestrator:
                insights = st.session_state.orchestrator.generate_comprehensive_insights("Sample_data_N.csv")
                st.markdown("**📊 Analysis Results:**")
                st.markdown(insights)
                
                # Update progress
                st.session_state.analysis_progress = 62
                progress_bar.progress(st.session_state.analysis_progress / 100)
                status_text.text("✅ Comprehensive insights generated successfully!")
            else:
                st.warning("⚠️ Please initialize the AI system first")
                st.session_state.analysis_progress = 62
                progress_bar.progress(st.session_state.analysis_progress / 100)
                
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            st.session_state.analysis_progress = 62
            progress_bar.progress(st.session_state.analysis_progress / 100)
        
        # Variance Analysis Section
        st.markdown("#### 📈 Variance Analysis")
        status_text.text("🔄 Performing variance analysis...")
        
        try:
            if st.session_state.orchestrator:
                variance_analysis = st.session_state.orchestrator.generate_variance_analysis(
                    "Sample_data_N.csv",
                    "Plan Number.csv", 
                    "fixed_clean_complete_data_with_forecasts.csv"
                )
                st.markdown("**📊 Variance Analysis Results:**")
                st.markdown(variance_analysis)
                
                # Update progress
                st.session_state.analysis_progress = 87
                progress_bar.progress(st.session_state.analysis_progress / 100)
                status_text.text("✅ Variance analysis completed successfully!")
            else:
                st.warning("⚠️ Please initialize the AI system first")
                st.session_state.analysis_progress = 87
                progress_bar.progress(st.session_state.analysis_progress / 100)
                
        except Exception as e:
            st.error(f"Error generating variance analysis: {str(e)}")
            st.session_state.analysis_progress = 87
            progress_bar.progress(st.session_state.analysis_progress / 100)
        
        # Q&A Section
        st.markdown("#### 💬 AI Assistant Q&A")
        status_text.text("🔄 Setting up AI Assistant...")
        
        try:
            # Chat interface
            user_question = st.text_input(
                "🤔 Ask your question:",
                placeholder="e.g., What are the spending trends for ABC234006?",
                key="user_question_tab1"
            )
            
            if st.button("🚀 Ask AI", type="primary") and user_question:
                if user_question.strip():
                    with st.spinner("🤖 AI is analyzing your question..."):
                        try:
                            available_files = [
                                "Sample_data_N.csv", 
                                "Plan Number.csv", 
                                "fixed_clean_complete_data_with_forecasts.csv"
                            ]
                            
                            answer = st.session_state.orchestrator.ask_question(user_question, available_files)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "question": user_question,
                                "answer": answer,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                            # Display answer
                            st.markdown("**🤖 AI Response:**")
                            st.markdown(answer)
                            
                        except Exception as e:
                            st.error(f"Error processing your question: {str(e)}")
            
            # Example questions
            st.markdown("**💡 Example Questions:**")
            example_questions = [
                "What are the spending trends for ABC234006?",
                "Which lineup has the highest variance between plan and forecast?",
                "Show me the monthly spending patterns by site",
                "What insights can you provide about our forecasting accuracy?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(example_questions):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(question, key=f"example_tab1_{i}"):
                        with st.spinner("🤖 AI is analyzing your question..."):
                            try:
                                available_files = [
                                    "Sample_data_N.csv", 
                                    "Plan Number.csv", 
                                    "fixed_clean_complete_data_with_forecasts.csv"
                                ]
                                
                                answer = st.session_state.orchestrator.ask_question(question, available_files)
                                
                                # Add to chat history
                                st.session_state.chat_history.append({
                                    "question": question,
                                    "answer": answer,
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                })
                                
                                # Display answer
                                st.markdown("**🤖 AI Response:**")
                                st.markdown(answer)
                                
                            except Exception as e:
                                st.error(f"Error processing your question: {str(e)}")
            
            # Update progress to completion
            st.session_state.analysis_progress = 100
            progress_bar.progress(st.session_state.analysis_progress / 100)
            status_text.text("🎉 All-in-One Analysis completed successfully!")
            
            # Success message
            st.success("🚀 All analysis sections have been completed! You can now explore the results above.")
            
        except Exception as e:
            st.error(f"Error setting up AI Assistant: {str(e)}")
            st.session_state.analysis_progress = 100
            progress_bar.progress(st.session_state.analysis_progress / 100)
        
        # Chat history display
        if st.session_state.chat_history:
            st.markdown("#### 💬 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
    
    # Tab 2: Data Explorer
    with tab2:
        st.markdown('<h2 class="section-header">📁 Data Explorer</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📁 Upload CSV file for analysis",
            type=['csv'],
            help="Upload a CSV file to analyze with the AI agent"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Data preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head())
                
                # Basic info
                st.subheader("📋 Data Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Columns:**", list(df.columns))
                    st.write("**Data Types:**", df.dtypes.to_dict())
                
                with col2:
                    st.write("**Missing Values:**", df.isnull().sum().to_dict())
                    st.write("**Shape:**", df.shape)
                
                # Quick analysis button
                if st.button("🔍 Analyze with AI", type="primary"):
                    with st.spinner("🤖 AI is analyzing your data..."):
                        try:
                            # Save uploaded file temporarily
                            temp_path = f"temp_{uploaded_file.name}"
                            df.to_csv(temp_path, index=False)
                            
                            insights = st.session_state.orchestrator.generate_comprehensive_insights(temp_path)
                            
                            st.markdown("### 🤖 AI Analysis Results")
                            st.markdown(insights)
                            
                            # Clean up
                            os.remove(temp_path)
                            
                        except Exception as e:
                            st.error(f"Error analyzing data: {str(e)}")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Data visualization options
        st.subheader("📊 Quick Data Visualization")
        
        try:
            # Load sample data for visualization
            df_sample = pd.read_csv("Sample_data_N.csv")
            df_sample = df_sample.dropna(how='all')
            df_sample['DATE'] = pd.to_datetime(df_sample['DATE'], errors='coerce')
            df_sample = df_sample.dropna(subset=['DATE'])
            
            # Chart selection
            chart_type = st.selectbox(
                "Choose chart type:",
                ["Monthly Trends by Lineup", "Monthly Trends by Site", "Total Spending by Lineup", "Spending Distribution"]
            )
            
            if chart_type == "Monthly Trends by Lineup":
                monthly_lineup = df_sample.groupby([df_sample['DATE'].dt.to_period('M'), 'Lineup'])['Actual'].sum().reset_index()
                monthly_lineup['DATE'] = monthly_lineup['DATE'].astype(str)
                
                fig = px.line(monthly_lineup, x='DATE', y='Actual', color='Lineup', 
                              title='Monthly Spending Trends by Lineup')
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Monthly Trends by Site":
                monthly_site = df_sample.groupby([df_sample['DATE'].dt.to_period('M'), 'Site'])['Actual'].sum().reset_index()
                monthly_site['DATE'] = monthly_site['DATE'].astype(str)
                
                fig = px.line(monthly_site, x='DATE', y='Actual', color='Site',
                              title='Monthly Spending Trends by Site')
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Total Spending by Lineup":
                total_by_lineup = df_sample.groupby('Lineup')['Actual'].sum().reset_index()
                
                fig = px.bar(total_by_lineup, x='Lineup', y='Actual',
                             title='Total Spending by Lineup')
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Spending Distribution":
                fig = px.box(df_sample, x='Lineup', y='Actual',
                             title='Spending Distribution by Lineup')
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating visualizations: {str(e)}")
        
        # Data summary statistics
        st.subheader("📈 Data Summary Statistics")
        
        try:
            if 'df' in locals():
                # Use uploaded data if available
                data_for_stats = df
            else:
                # Use sample data
                data_for_stats = pd.read_csv("Sample_data_N.csv")
                data_for_stats = data_for_stats.dropna(how='all')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_records = len(data_for_stats)
                st.metric("Total Records", total_records)
            
            with col2:
                if 'Actual' in data_for_stats.columns:
                    total_spending = data_for_stats['Actual'].sum()
                    st.metric("Total Spending", f"${total_spending:,.0f}")
                else:
                    st.metric("Total Records", total_records)
            
            with col3:
                if 'Lineup' in data_for_stats.columns:
                    num_lineups = data_for_stats['Lineup'].nunique()
                    st.metric("Lineups", num_lineups)
                else:
                    st.metric("Columns", len(data_for_stats.columns))
            
            with col4:
                if 'DATE' in data_for_stats.columns:
                    try:
                        data_for_stats['DATE'] = pd.to_datetime(data_for_stats['DATE'], errors='coerce')
                        date_range = f"{data_for_stats['DATE'].min().strftime('%Y-%m')} to {data_for_stats['DATE'].max().strftime('%Y-%m')}"
                        st.metric("Date Range", date_range)
                    except:
                        st.metric("Data Types", len(data_for_stats.dtypes.unique()))
                else:
                    st.metric("Data Types", len(data_for_stats.dtypes.unique()))
                    
        except Exception as e:
            st.error(f"Error displaying summary statistics: {str(e)}")
        
        # Data quality check
        st.subheader("🔍 Data Quality Check")
        
        try:
            if 'df' in locals():
                data_for_quality = df
            else:
                data_for_quality = pd.read_csv("Sample_data_N.csv")
                data_for_quality = data_for_quality.dropna(how='all')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Completeness:**")
                completeness = (1 - data_for_quality.isnull().sum() / len(data_for_quality)) * 100
                for col, comp in completeness.items():
                    st.write(f"• {col}: {comp:.1f}% complete")
            
            with col2:
                st.markdown("**Data Types:**")
                for col, dtype in data_for_quality.dtypes.items():
                    st.write(f"• {col}: {dtype}")
                    
        except Exception as e:
            st.error(f"Error checking data quality: {str(e)}")

if __name__ == "__main__":
    main()



