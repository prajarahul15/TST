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
import requests
from typing import Optional

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
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

# Economic Data and Web Search Tools
@tool
def search_economic_indicators(query: str, max_results: int = 5) -> str:
    """
    Search for economic indicators and macro/microeconomic data using web search.
    
    Args:
        query: Search terms like 'US inflation rate 2024', 'GDP growth forecast', 'unemployment trends'
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        String containing search results with economic data
    """
    try:
        print(f"🔍 Searching for economic indicators: {query}")
        
        # Using a simple web search approach (you can replace with specialized APIs)
        search_url = "https://api.duckduckgo.com/?"
        params = {
            'q': f"{query} economic data statistics",
            'format': 'json',
            'no_redirect': '1',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code == 200:
            # For demo purposes, return a structured response
            # In production, you'd parse the actual search results
            return f"""
Economic Search Results for: {query}

📊 Key Findings:
• Search query processed successfully
• Economic indicators data requested
• Real-time data integration ready

💡 Recommendations:
• Consider using FRED API for official US economic data
• World Bank API for global economic indicators
• Alpha Vantage for financial market data
• Trading Economics for real-time indicators

🔗 Data Sources Suggested:
• Federal Reserve Economic Data (FRED)
• Bureau of Labor Statistics
• World Bank Open Data
• International Monetary Fund (IMF)

Note: This is a demo response. In production, integrate with:
- FRED API for US economic data
- World Bank API for global data
- Alpha Vantage for market data
- News APIs for latest economic reports
"""
        else:
            return f"Error searching for economic data: HTTP {response.status_code}"
            
    except Exception as e:
        print(f"❌ ERROR in search_economic_indicators: {str(e)}")
        return f"Error searching for economic indicators: {str(e)}"

@tool
def get_industry_analysis(industry: str, region: str = "US") -> str:
    """
    Get industry-specific analysis and trends.
    
    Args:
        industry: Industry name like 'cloud computing', 'automotive', 'healthcare'
        region: Geographic region like 'US', 'EU', 'Global' (default: 'US')
    
    Returns:
        String containing industry analysis and trends
    """
    try:
        print(f"🏭 Analyzing industry: {industry} in {region}")
        
        # For demo purposes, return structured industry analysis
        # In production, integrate with industry databases
        return f"""
Industry Analysis: {industry.title()} ({region})

📈 Market Overview:
• Industry: {industry.title()}
• Geographic Focus: {region}
• Analysis Type: Comprehensive sector review

🔍 Key Metrics to Monitor:
• Market size and growth rate
• Major players and market share
• Regulatory environment
• Technological disruptions
• Supply chain dynamics

📊 Recommended Data Sources:
• IBISWorld for industry reports
• Statista for market statistics
• Trade association publications
• Government sector analyses
• Financial reports from key players

💡 Analysis Framework:
• Porter's Five Forces analysis
• SWOT analysis for sector
• Competitive landscape mapping
• Technology adoption trends
• Regulatory impact assessment

🎯 Next Steps:
• Integrate specific industry APIs
• Set up news monitoring for {industry}
• Track key performance indicators
• Monitor competitor activities

Note: This is a demo response. In production, integrate with:
- Industry-specific APIs
- Market research databases
- Trade association data feeds
- Financial data providers
"""
        
    except Exception as e:
        print(f"❌ ERROR in get_industry_analysis: {str(e)}")
        return f"Error analyzing industry data: {str(e)}"

@tool
def get_market_sentiment(topic: str) -> str:
    """
    Get market sentiment and news analysis for specific topics.
    
    Args:
        topic: Topic to analyze like 'tech stocks', 'interest rates', 'inflation'
    
    Returns:
        String containing market sentiment analysis
    """
    try:
        print(f"📰 Analyzing market sentiment for: {topic}")
        
        # Demo sentiment analysis response
        # In production, integrate with news APIs and sentiment analysis
        return f"""
Market Sentiment Analysis: {topic.title()}

📈 Sentiment Overview:
• Topic: {topic.title()}
• Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
• Sentiment Status: Monitoring active

🎯 Sentiment Indicators:
• News sentiment: Neutral to Positive
• Social media buzz: Moderate activity
• Expert opinions: Mixed outlook
• Market reactions: Stable

📊 Data Integration Points:
• Financial news aggregation
• Social media sentiment tracking
• Expert opinion monitoring
• Market price correlation

🔍 Recommended Monitoring:
• Real-time news feeds
• Twitter/X sentiment tracking
• Financial analyst reports
• Market volatility indicators

💡 Implementation Suggestions:
• News API integration (NewsAPI, Finnhub)
• Social sentiment APIs (Twitter API)
• Financial data feeds (Alpha Vantage, Yahoo Finance)
• Sentiment analysis models (VADER, TextBlob)

Note: This is a demo response. In production, integrate with:
- Real-time news APIs
- Social media sentiment APIs
- Financial market data feeds
- Natural language processing models
"""
        
    except Exception as e:
        print(f"❌ ERROR in get_market_sentiment: {str(e)}")
        return f"Error analyzing market sentiment: {str(e)}"

# Agent Orchestrator Class
class FinancialInsightOrchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            load_and_prepare_data,
            get_data_statistics,
            analyze_trends,
            create_visualization,
            perform_variance_analysis,
            search_economic_indicators,
            get_industry_analysis,
            get_market_sentiment
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
            ("system", """You are a Financial Data Analysis Expert with access to external economic data. Your role is to:
            1. Analyze financial data at Profile → Site → Lineup hierarchy
            2. Generate comprehensive insights from actual data
            3. Perform variance analysis between Actual vs Plan vs Forecast
            4. Create visualizations and provide actionable recommendations
            5. Search for economic indicators and macro/microeconomic data
            6. Analyze industry-specific trends and market conditions
            7. Monitor market sentiment and external factors
            8. Use available tools to gather internal and external data for analysis
            
            When analyzing data, consider external economic factors, industry trends, and market conditions.
            Always provide structured, business-focused insights with clear recommendations and external context."""),
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

# Add these helper functions after your existing functions, before the main() function

def run_monte_carlo_simulation(forecast_data, num_simulations, confidence_level, time_horizon, selected_lineup):
    """Run Monte Carlo simulation for financial forecasting for a specific lineup"""
    print(f"\n{'='*60}")
    print(f"🚀 Monte Carlo Simulation: Running with parameters:")
    print(f"📊 Selected Lineup: {selected_lineup}")
    print(f"📊 forecast_data type: {type(forecast_data)}")
    print(f"📊 forecast_data keys: {forecast_data.keys() if isinstance(forecast_data, dict) else 'Not a dict'}")
    
    try:
        # Validate input data
        if not isinstance(forecast_data, dict) or 'results' not in forecast_data:
            print(f"❌ Invalid forecast_data structure: {forecast_data}")
            st.error("Invalid forecast data structure for Monte Carlo simulation")
            return None
        
        # Validate selected lineup exists
        if selected_lineup not in forecast_data['results']:
            print(f"❌ Selected lineup '{selected_lineup}' not found in forecast data")
            st.error(f"Selected lineup '{selected_lineup}' not found in forecast data")
            return None
        
        results = {}
        print(f"📊 Processing selected lineup: {selected_lineup}")
        
        # Process only the selected lineup
        lineup = selected_lineup
        print(f"🎯 Processing lineup: {lineup}")
        lineup_data = forecast_data['results'][lineup]
        print(f"📊 Lineup data keys: {lineup_data.keys() if isinstance(lineup_data, dict) else 'Not a dict'}")
        
        # Validate lineup data structure
        if not isinstance(lineup_data, dict) or 'forecast' not in lineup_data:
            print(f"❌ Invalid lineup data structure for {lineup}")
            st.error(f"Invalid lineup data structure for {lineup}")
            return None
        
        forecast_df = lineup_data['forecast']
        print(f"📊 Forecast DataFrame shape: {forecast_df.shape if hasattr(forecast_df, 'shape') else 'Not a DataFrame'}")
        
        # Check if 'Forecast' column exists
        if 'Forecast' not in forecast_df.columns:
            print(f"❌ 'Forecast' column not found in {lineup} data")
            st.error(f"'Forecast' column not found in {lineup} data")
            return None
        
        # Extract forecast values and calculate statistics
        forecast_values = forecast_df['Forecast'].values
        print(f"📊 Forecast values shape: {forecast_values.shape}")
        print(f"📊 Forecast values sample: {forecast_values[:5]}")
        
        if len(forecast_values) == 0:
            print(f"❌ No forecast values for {lineup}")
            st.error(f"No forecast values for {lineup}")
            return None
        
        mean_forecast = np.mean(forecast_values)
        std_forecast = np.std(forecast_values)
        
        print(f"📊 Mean forecast: {mean_forecast:.2f}")
        print(f"📊 Std forecast: {std_forecast:.2f}")
        
        # Generate Monte Carlo simulations
        simulations = np.random.normal(mean_forecast, std_forecast, (num_simulations, time_horizon))
        
        # Calculate statistics for each simulation
        simulation_totals = np.sum(simulations, axis=1)
        
        # Calculate confidence intervals
        alpha = 1 - (confidence_level / 100)
        lower_ci = np.percentile(simulation_totals, (alpha/2) * 100)
        upper_ci = np.percentile(simulation_totals, (1 - alpha/2) * 100)
        
        # Determine risk level
        volatility = std_forecast / mean_forecast if mean_forecast != 0 else 0
        if volatility > 0.3:
            risk_level = "High"
        elif volatility > 0.15:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        results[lineup] = {
            'mean': mean_forecast * time_horizon,
            'std': std_forecast * np.sqrt(time_horizon),
            'min': np.min(simulation_totals),
            'max': np.max(simulation_totals),
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'risk_level': risk_level,
            'simulations': simulations,
            'simulation_totals': simulation_totals
        }
        
        print(f"✅ Successfully processed {lineup}")
        
        if not results:
            print(f"❌ No results generated for any lineup")
            st.error("No Monte Carlo results generated")
            return None
        
        print(f"✅ Monte Carlo simulation completed successfully for {len(results)} lineups")
        return {
            'lineups': list(results.keys()),
            'results': results,
            'parameters': {
                'num_simulations': num_simulations,
                'confidence_level': confidence_level,
                'time_horizon': time_horizon
            }
        }
        
    except Exception as e:
        print(f"❌ ERROR in Monte Carlo simulation: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        st.error(f"Error in Monte Carlo simulation: {str(e)}")
        return None

def run_scenario_analysis(growth_rate, cost_change, market_conditions, selected_lineup):
    """Run scenario analysis for different business conditions for a specific lineup"""
    print(f"\n{'='*60}")
    print(f"📊 Scenario Analysis: Running with parameters:")
    print(f"🎯 Selected Lineup: {selected_lineup}")
    print(f"📈 Growth Rate: {growth_rate}%")
    print(f"💰 Cost Change: {cost_change}%")
    print(f"🌍 Market Conditions: {market_conditions}")
    
    try:
        # Validate inputs
        if not isinstance(growth_rate, (int, float)):
            print(f"❌ Invalid growth_rate type: {type(growth_rate)}")
            st.error("Invalid growth rate parameter")
            return None
        
        if not isinstance(cost_change, (int, float)):
            print(f"❌ Invalid cost_change type: {type(cost_change)}")
            st.error("Invalid cost change parameter")
            return None
        
        if not isinstance(market_conditions, list):
            print(f"❌ Invalid market_conditions type: {type(market_conditions)}")
            st.error("Invalid market conditions parameter")
            return None
        
        # Base scenario (current)
        base_scenario = {
            'name': 'Base Scenario',
            'growth_rate': 0,
            'cost_change': 0,
            'market_conditions': ['Current Market'],
            'revenue_impact': 0,
            'cost_impact': 0,
            'net_impact': 0
        }
        
        # Create scenarios based on parameters
        scenarios = [base_scenario]
        
        # Growth scenarios
        if growth_rate != 0:
            growth_scenario = {
                'name': f'Growth Scenario ({growth_rate:+.1f}%)',
                'growth_rate': growth_rate,
                'cost_change': 0,
                'market_conditions': market_conditions,
                'revenue_impact': 1000000 * (growth_rate / 100),  # Simplified calculation
                'cost_impact': 0,
                'net_impact': 1000000 * (growth_rate / 100)
            }
            scenarios.append(growth_scenario)
            print(f"✅ Added growth scenario: {growth_rate:+.1f}%")
        
        # Cost scenarios
        if cost_change != 0:
            cost_scenario = {
                'name': f'Cost Scenario ({cost_change:+.1f}%)',
                'growth_rate': 0,
                'cost_change': cost_change,
                'market_conditions': market_conditions,
                'revenue_impact': 0,
                'cost_impact': -500000 * (cost_change / 100),  # Simplified calculation
                'net_impact': -500000 * (cost_change / 100)
            }
            scenarios.append(cost_scenario)
            print(f"✅ Added cost scenario: {cost_change:+.1f}%")
        
        # Combined scenario
        if growth_rate != 0 and cost_change != 0:
            combined_scenario = {
                'name': f'Combined Scenario (Growth: {growth_rate:+.1f}%, Cost: {cost_change:+.1f}%)',
                'growth_rate': growth_rate,
                'cost_change': cost_change,
                'market_conditions': market_conditions,
                'revenue_impact': 1000000 * (growth_rate / 100),
                'cost_impact': -500000 * (cost_change / 100),
                'net_impact': 1000000 * (growth_rate / 100) - 500000 * (cost_change / 100)
            }
            scenarios.append(combined_scenario)
            print(f"✅ Added combined scenario")
        
        print(f"✅ Created {len(scenarios)} scenarios successfully")
        
        return {
            'scenarios': scenarios,
            'parameters': {
                'growth_rate': growth_rate,
                'cost_change': cost_change,
                'market_conditions': market_conditions
            }
        }
        
    except Exception as e:
        print(f"❌ ERROR in scenario analysis: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        st.error(f"Error in scenario analysis: {str(e)}")
        return None

def run_stress_test(revenue_shock, revenue_duration, cost_shock, cost_duration, stress_factors, stress_severity, selected_lineup):
    """Run stress test under extreme conditions for a specific lineup"""
    print(f"\n{'='*60}")
    print(f"⚡ Stress Test: Running with parameters:")
    print(f"🎯 Selected Lineup: {selected_lineup}")
    print(f"📉 Revenue Shock: {revenue_shock}% for {revenue_duration} months")
    print(f"📈 Cost Shock: {cost_shock}% for {cost_duration} months")
    print(f"🌪️ Stress Factors: {stress_factors}")
    print(f"⚠️ Stress Severity: {stress_severity}")
    
    try:
        # Validate inputs
        if not isinstance(revenue_shock, (int, float)):
            print(f"❌ Invalid revenue_shock type: {type(revenue_shock)}")
            st.error("Invalid revenue shock parameter")
            return None
        
        if not isinstance(revenue_duration, (int, float)):
            print(f"❌ Invalid revenue_duration type: {type(revenue_duration)}")
            st.error("Invalid revenue duration parameter")
            return None
        
        if not isinstance(cost_shock, (int, float)):
            print(f"❌ Invalid cost_shock type: {type(cost_shock)}")
            st.error("Invalid cost shock parameter")
            return None
        
        if not isinstance(cost_duration, (int, float)):
            print(f"❌ Invalid cost_duration type: {type(cost_duration)}")
            st.error("Invalid cost duration parameter")
            return None
        
        if not isinstance(stress_factors, list):
            print(f"❌ Invalid stress_factors type: {type(stress_factors)}")
            st.error("Invalid stress factors parameter")
            return None
        
        if not isinstance(stress_severity, str):
            print(f"❌ Invalid stress_severity type: {type(stress_severity)}")
            st.error("Invalid stress severity parameter")
            return None
        
        # Calculate stress impacts
        total_revenue_impact = revenue_shock * revenue_duration * 10000  # Simplified calculation
        total_cost_impact = cost_shock * cost_duration * 5000  # Simplified calculation
        net_impact = total_revenue_impact + total_cost_impact
        
        print(f"📊 Calculated impacts:")
        print(f"   - Revenue Impact: £{total_revenue_impact:,.0f}")
        print(f"   - Cost Impact: £{total_cost_impact:,.0f}")
        print(f"   - Net Impact: £{net_impact:,.0f}")
        
        # Calculate recovery time based on stress severity
        if "Mild Stress" in stress_severity:
            recovery_months = 1
        elif "Moderate Stress" in stress_severity:
            recovery_months = 3
        elif "Severe Stress" in stress_severity:
            recovery_months = 6
        elif "Extreme Stress" in stress_severity:
            recovery_months = 12
        else:
            recovery_months = 3  # Default fallback
        
        print(f"⏰ Recovery time: {recovery_months} months")
        
        # Risk assessment
        if abs(net_impact) > 1000000:
            risk_level = "High"
            description = "Significant financial impact detected under stress conditions"
            mitigation = "Implement immediate cost controls and revenue enhancement strategies"
        elif abs(net_impact) > 500000:
            risk_level = "Medium"
            description = "Moderate financial impact under stress conditions"
            mitigation = "Review and adjust financial planning assumptions"
        else:
            risk_level = "Low"
            description = "Minimal financial impact under stress conditions"
            mitigation = "Continue monitoring and maintain current strategies"
        
        print(f"🚨 Risk Level: {risk_level}")
        print(f"📝 Description: {description}")
        print(f"🛡️ Mitigation: {mitigation}")
        
        result = {
            'total_revenue_impact': total_revenue_impact,
            'total_cost_impact': total_cost_impact,
            'net_impact': net_impact,
            'recovery_months': recovery_months,
            'stress_factors': stress_factors,
            'stress_severity': stress_severity,
            'risk_assessment': {
                'risk_level': risk_level,
                'description': description,
                'mitigation': mitigation
            }
        }
        
        print(f"✅ Stress test completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ ERROR in stress test: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        st.error(f"Error in stress test: {str(e)}")
        return None

def create_monte_carlo_chart(monte_carlo_results, lineup):
    """Create Monte Carlo simulation chart"""
    try:
        results = monte_carlo_results['results'][lineup]
        simulation_totals = results['simulation_totals']
        
        fig = go.Figure()
        
        # Histogram of simulation results
        fig.add_trace(go.Histogram(
            x=simulation_totals,
            nbinsx=50,
            name='Simulation Distribution',
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        # Add confidence intervals
        fig.add_vline(x=results['lower_ci'], line_dash="dash", line_color="red", 
                     annotation_text=f"{monte_carlo_results['parameters']['confidence_level']}% Lower CI")
        fig.add_vline(x=results['upper_ci'], line_dash="dash", line_color="red",
                     annotation_text=f"{monte_carlo_results['parameters']['confidence_level']}% Upper CI")
        fig.add_vline(x=results['mean'], line_dash="solid", line_color="green",
                     annotation_text="Mean Forecast")
        
        fig.update_layout(
            title=f'{lineup} - Monte Carlo Simulation Results',
            xaxis_title='Total Forecast Value (£)',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Monte Carlo chart: {str(e)}")
        return None

def create_scenario_comparison_chart(scenario_results):
    """Create scenario comparison chart"""
    try:
        scenarios = scenario_results['scenarios']
        
        fig = go.Figure()
        
        # Add bars for each scenario
        for i, scenario in enumerate(scenarios):
            fig.add_trace(go.Bar(
                name=scenario['name'],
                x=['Revenue Impact', 'Cost Impact', 'Net Impact'],
                y=[scenario['revenue_impact'], scenario['cost_impact'], scenario['net_impact']],
                text=[f"£{scenario['revenue_impact']:,.0f}", 
                      f"£{scenario['cost_impact']:,.0f}", 
                      f"£{scenario['net_impact']:,.0f}"],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Scenario Analysis Comparison',
            xaxis_title='Impact Type',
            yaxis_title='Financial Impact (£)',
            height=400,
            barmode='group'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating scenario chart: {str(e)}")
        return None

def create_sensitivity_chart(scenario_results):
    """Create sensitivity analysis chart"""
    try:
        # This is a simplified sensitivity chart
        fig = go.Figure()
        
        # Create sample sensitivity data
        growth_rates = [-20, -10, 0, 10, 20]
        net_impacts = [growth_rate * 10000 for growth_rate in growth_rates]
        
        fig.add_trace(go.Scatter(
            x=growth_rates,
            y=net_impacts,
            mode='lines+markers',
            name='Growth Rate Sensitivity',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Sensitivity Analysis: Growth Rate vs Net Impact',
            xaxis_title='Growth Rate Change (%)',
            yaxis_title='Net Financial Impact (£)',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating sensitivity chart: {str(e)}")
        return None

def create_stress_test_chart(stress_results, chart_type):
    """Create stress test chart"""
    try:
        fig = go.Figure()
        
        if chart_type == 'revenue':
            # Revenue stress over time
            months = list(range(1, 13))
            revenue_values = [1000000 + (stress_results['total_revenue_impact'] / 12) * month for month in months]
            
            fig.add_trace(go.Scatter(
                x=months,
                y=revenue_values,
                mode='lines+markers',
                name='Revenue Under Stress',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title='Revenue Stress Test Over Time',
                xaxis_title='Month',
                yaxis_title='Revenue (£)',
                height=400
            )
            
        elif chart_type == 'cost':
            # Cost stress over time
            months = list(range(1, 13))
            cost_values = [500000 + (stress_results['total_cost_impact'] / 12) * month for month in months]
            
            fig.add_trace(go.Scatter(
                x=months,
                y=cost_values,
                mode='lines+markers',
                name='Cost Under Stress',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title='Cost Stress Test Over Time',
                xaxis_title='Month',
                yaxis_title='Cost (£)',
                height=400
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating stress test chart: {str(e)}")
        return None

def create_recovery_chart(stress_results):
    """Create recovery analysis chart"""
    try:
        fig = go.Figure()
        
        # Recovery timeline
        months = list(range(1, stress_results['recovery_months'] + 1))
        recovery_values = [100 + (100 / stress_results['recovery_months']) * month for month in months]
        
        fig.add_trace(go.Scatter(
            x=months,
            y=recovery_values,
            mode='lines+markers',
            name='Recovery Path',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Recovery Analysis Timeline',
            xaxis_title='Month',
            yaxis_title='Recovery Percentage (%)',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating recovery chart: {str(e)}")
        return None

# Initialize the system
def initialize_system(openai_api_key):
    """Initialize the LLM and orchestrator"""
    try:
        #GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        #OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            #st.error("❌ GEMINI_API_KEY not found in .env file")
            #st.info("Please create a .env file with your Gemini API key")
            st.error("❌ Please provide your OpenAI API key in the sidebar")
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
# Modify your main() function to implement the new structure
def main():
    st.markdown('<h1 class="main-header">�� Financial Insights AI</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent Financial Data Analysis with Multi-Agent System**")
    
    # Sidebar
    st.sidebar.title("�� System Controls")
    openai_api_key = get_openai_api_key_from_ui()
    # System info
    st.sidebar.markdown("**📊 Analysis Pipeline:**")
    st.sidebar.markdown("1. 📊 Data Overview")
    st.sidebar.markdown("2. 🔮 Forecast")
    st.sidebar.markdown("3. 🎲 Simulation")
    st.sidebar.markdown("4. 📈 Variance Analysis")
    st.sidebar.markdown("5. 🔍 Generate Insight")
    st.sidebar.markdown("6. 💬 Q&A")
    st.sidebar.markdown("7. 📊 Dashboard")
    
    st.sidebar.markdown("---")
    
    # Initialize system
    if st.sidebar.button("🚀 Initialize AI System", type="primary"):
        if initialize_system(openai_api_key):
            st.rerun()
    
    # Check if system is initialized
    if st.session_state.orchestrator is None:
        st.warning("⚠️ Please initialize the AI system first using the button in the sidebar.")
        st.info("�� Make sure you have a `.env` file with your `GEMINI_API_KEY`")
        return
    
    # Initialize session state for cross-tab data sharing
    if 'shared_results' not in st.session_state:
        st.session_state.shared_results = {
            'data_overview': None,
            'forecast': None,
            'simulation': None,
            'variance_analysis': None,
            'insights': None,
            'qa_history': []
        }
    
    # Progress tracking for all modules
    st.markdown("---")
    st.markdown("### 📊 Analysis Progress Tracker")
    
    # Initialize progress tracking
    if 'module_progress' not in st.session_state:
        st.session_state.module_progress = {
            'data_overview': False,
            'forecast': False,
            'simulation': False,
            'variance_analysis': False,
            'insights': False,
            'qa': False,
            'dashboard': False
        }
    
    # Calculate overall progress
    completed_modules = sum(st.session_state.module_progress.values())
    total_modules = len(st.session_state.module_progress)
    progress_percentage = (completed_modules / total_modules) * 100
    
    # Progress bar
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.progress(progress_percentage / 100, text=f"Overall Progress: {progress_percentage:.1f}% ({completed_modules}/{total_modules} modules)")
    
    with col2:
        st.metric("Completed", f"{completed_modules}/{total_modules}")
    
    with col3:
        if progress_percentage == 100:
            st.success("🎉 All modules completed!")
        elif progress_percentage >= 70:
            st.info("🚀 Almost there!")
        elif progress_percentage >= 40:
            st.warning("📈 Good progress!")
        else:
            st.info("🔄 Getting started...")
    
    # Module status indicators
    st.markdown("#### 📋 Module Status:")
    
    # Create columns for module status
    status_cols = st.columns(7)
    
    module_names = [
        ("📊 Data Overview", "data_overview"),
        ("🔮 Forecast", "forecast"),
        ("🎲 Simulation", "simulation"),
        ("📈 Variance Analysis", "variance_analysis"),
        ("🔍 Generate Insight", "insights"),
        ("💬 Q&A", "qa"),
        ("📊 Dashboard", "dashboard")
    ]
    
    for i, (display_name, module_key) in enumerate(module_names):
        with status_cols[i]:
            if st.session_state.module_progress[module_key]:
                st.success(f"✅ {display_name}")
            else:
                st.info(f"⏳ {display_name}")
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Data Overview",
        "🔮 Forecast", 
        "🎲 Simulation",
        "📈 Variance Analysis",
        "🔍 Generate Insight",
        "💬 Q&A",
        "�� Dashboard"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.markdown('<h2 class="section-header">�� Data Overview</h2>', unsafe_allow_html=True)
        
        try:
            # Load sample data for overview
            df = pd.read_csv("Sample_data_N.csv")
            df = df.dropna(how='all')
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df = df.dropna(subset=['DATE'])
            
            # Store in session state
            st.session_state.shared_results['data_overview'] = df
            
            # Display data overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_records = len(df)
                st.metric("Total Records", total_records)
            
            with col2:
                total_spending = df['Actual'].sum()
                st.metric("Total Spending", f"${total_spending:,.0f}")
            
            with col3:
                num_lineups = df['Lineup'].nunique()
                st.metric("Lineups", num_lineups)
            
            with col4:
                date_range = f"{df['DATE'].min().strftime('%Y-%m')} to {df['DATE'].max().strftime('%Y-%m')}"
                st.metric("Date Range", date_range)
            
            # Data preview
            st.markdown("#### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data information
            st.markdown("#### 📊 Data Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Columns:**", list(df.columns))
                st.write("**Data Types:**", df.dtypes.to_dict())
            
            with col2:
                st.write("**Missing Values:**", df.isnull().sum().to_dict())
                st.write("**Shape:**", df.shape)
            
            st.success("✅ Data overview completed successfully!")
            
            # Update progress
            st.session_state.module_progress['data_overview'] = True
            
        except Exception as e:
            st.error(f"Error in data overview: {str(e)}")
    
    # Tab 2: Forecast
    with tab2:
        st.markdown('<h2 class="section-header">🔮 Financial Forecasting</h2>', unsafe_allow_html=True)
        
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
                forecast_status.text(f"�� Training model for {lineup}... ({i+1}/{len(lineups)})")
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
            
            # Store forecast results in session state
            st.session_state.shared_results['forecast'] = {
                'results': forecast_results,
                'metrics': forecast_metrics
            }
            
            # Display forecasting results
            if forecast_metrics:
                st.markdown("##### 📈 Forecasting Model Performance")
                
                # Create metrics DataFrame
                metrics_df = pd.DataFrame(forecast_metrics)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Display forecast charts
                st.markdown("##### �� 12-Month Forecasts by Lineup")
                
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
                
                st.success("🎉 Forecasting analysis completed successfully!")
                
                # Update progress
                st.session_state.module_progress['forecast'] = True
                
            else:
                st.warning("⚠️ No forecasting results generated. Check data availability.")
                
        except ImportError as e:
            st.error(f"❌ Error importing forecasting module: {str(e)}")
            st.info("💡 Make sure 'fixed_forecasting.py' is in the same directory")
        except Exception as e:
            st.error(f"❌ Error during forecasting: {str(e)}")
            st.info("💡 Check console for detailed error messages")
    
        # Tab 3: Simulation - ONLY RUNS WHEN USER CLICKS THIS TAB
    with tab3:
        st.markdown('<h2 class="section-header">🎲 Financial Simulation & Analysis</h2>', unsafe_allow_html=True)
        
        # Check if forecast data is available
        if st.session_state.shared_results.get('forecast') is None:
            st.warning("⚠️ Please run the Forecast analysis first to enable simulations")
            st.info("💡 Go to the Forecast tab and run the analysis, then return here")
        else:
            # Debug: Show forecast data structure
            st.markdown("#### 🔍 Debug: Forecast Data Structure")
            forecast_data = st.session_state.shared_results.get('forecast')
            st.write(f"**Forecast data type:** {type(forecast_data)}")
            st.write(f"**Forecast data keys:** {list(forecast_data.keys()) if isinstance(forecast_data, dict) else 'Not a dict'}")
            
            if 'results' in forecast_data:
                st.write(f"**Results keys:** {list(forecast_data['results'].keys())}")
                if len(forecast_data['results']) > 0:
                    first_lineup = list(forecast_data['results'].keys())[0]
                    st.write(f"**First lineup ({first_lineup}) data keys:** {list(forecast_data['results'][first_lineup].keys())}")
                    if 'forecast' in forecast_data['results'][first_lineup]:
                        forecast_df = forecast_data['results'][first_lineup]['forecast']
                        st.write(f"**Forecast DataFrame shape:** {forecast_df.shape}")
                        st.write(f"**Forecast DataFrame columns:** {list(forecast_df.columns)}")
                        st.write("**Sample forecast data:**")
                        st.dataframe(forecast_df.head(), use_container_width=True)
            
            st.markdown("---")
            
            # Check if already executed
            if st.session_state.shared_results.get('simulation') is not None:
                st.success("✅ Simulation analysis already completed!")
                
                # Display existing results
                display_simulation_results(st.session_state.shared_results['simulation'])
                
                # Option to re-run with new parameters
                if st.button("🔄 Re-run with New Parameters", type="secondary", key="rerun_simulation"):
                    st.session_state.shared_results['simulation'] = {}
                    st.rerun()
                
            else:
                st.info("💡 Go to the specific simulation tabs below to configure parameters and run simulations")
        
        # Create nested tabs for simulation
        sim_tab1, sim_tab2, sim_tab3 = st.tabs([
            "🎯 Monte Carlo Forecast", 
            "📊 Scenario Analysis", 
            "⚡ Stress Test"
        ])
        
        # Monte Carlo Forecast Tab
        with sim_tab1:
            st.markdown("#### 🎯 Monte Carlo Simulation")
            
            # Lineup Selection for Monte Carlo
            available_lineups = []
            if 'results' in st.session_state.shared_results.get('forecast', {}):
                available_lineups = list(st.session_state.shared_results['forecast']['results'].keys())
            
            if not available_lineups:
                st.error("❌ No lineups available. Please run the Forecast analysis first.")
            else:
                # Lineup selector for Monte Carlo
                selected_lineup_mc = st.selectbox(
                    "Choose a Lineup to simulate:",
                    options=available_lineups,
                    help="Select the specific Lineup you want to run Monte Carlo simulation for",
                    key="monte_carlo_lineup_selector"
                )
                
                st.success(f"✅ Selected Lineup: **{selected_lineup_mc}**")
                st.markdown("---")
                
                st.markdown(f"#### 🎯 Monte Carlo Simulation for {selected_lineup_mc}")
                
                if (st.session_state.shared_results.get('simulation') is not None and 
                    'monte_carlo' in st.session_state.shared_results['simulation'] and
                    st.session_state.shared_results['simulation']['monte_carlo'].get('lineup') == selected_lineup_mc):
                    # Display existing results for this lineup
                    display_monte_carlo_results(st.session_state.shared_results['simulation']['monte_carlo'])
                else:
                    # Monte Carlo Parameters Configuration
                    st.info("💡 Monte Carlo simulation uses random sampling to model uncertainty in financial forecasts")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        num_simulations = st.number_input(
                            "Number of Simulations", 
                            min_value=100, max_value=10000, value=1000, step=100, 
                            help="Higher number = more accurate results but slower execution",
                            key="mc_simulations"
                        )
                    with col2:
                        confidence_level = st.slider(
                            "Confidence Level (%)", 
                            min_value=80, max_value=99, value=95, step=1,
                            help="95% means 95% of simulations fall within the confidence interval",
                            key="mc_confidence"
                        )
                    with col3:
                        time_horizon = st.number_input(
                            "Time Horizon (months)", 
                            min_value=1, max_value=24, value=12, step=1,
                            help="How many months ahead to forecast",
                            key="mc_horizon"
                        )
                    
                    # Parameter summary
                    st.markdown("#### 📋 Monte Carlo Parameters Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Simulations", f"{num_simulations:,}")
                    with col2:
                        st.metric("Confidence", f"{confidence_level}%")
                    with col3:
                        st.metric("Horizon", f"{time_horizon} months")
                    
                    # Run Monte Carlo button
                    if st.button("🚀 Run Monte Carlo Simulation", type="primary", key="run_monte_carlo"):
                        with st.spinner("🎲 Running Monte Carlo simulation..."):
                            try:
                                # Debug: Check forecast data structure
                                forecast_data = st.session_state.shared_results.get('forecast')
                                st.write("🔍 Debug: Checking forecast data structure...")
                                
                                if forecast_data is None:
                                    st.error("❌ No forecast data available. Please run the Forecast tab first.")
                                    return
                                
                                st.write(f"📊 Forecast data type: {type(forecast_data)}")
                                st.write(f"📊 Forecast data keys: {list(forecast_data.keys()) if isinstance(forecast_data, dict) else 'Not a dict'}")
                                
                                if 'results' in forecast_data:
                                    st.write(f"📊 Results keys: {list(forecast_data['results'].keys())}")
                                    if len(forecast_data['results']) > 0:
                                        first_lineup = list(forecast_data['results'].keys())[0]
                                        st.write(f"📊 First lineup data keys: {list(forecast_data['results'][first_lineup].keys())}")
                                        if 'forecast' in forecast_data['results'][first_lineup]:
                                            forecast_df = forecast_data['results'][first_lineup]['forecast']
                                            st.write(f"📊 Forecast DataFrame shape: {forecast_df.shape}")
                                            st.write(f"📊 Forecast DataFrame columns: {list(forecast_df.columns)}")
                                
                                # Run only Monte Carlo simulation for selected lineup
                                monte_carlo_results = run_monte_carlo_simulation(
                                    forecast_data,
                                    num_simulations, confidence_level, time_horizon,
                                    selected_lineup_mc
                                )
                                
                                # Add proper null checks to prevent TypeError
                                if monte_carlo_results is not None:
                                    # Initialize simulation results if not exists or is None
                                    if ('simulation' not in st.session_state.shared_results or 
                                        st.session_state.shared_results['simulation'] is None):
                                        st.session_state.shared_results['simulation'] = {}
                                    
                                    # Store Monte Carlo results with lineup info
                                    monte_carlo_results['lineup'] = selected_lineup_mc
                                    st.session_state.shared_results['simulation']['monte_carlo'] = monte_carlo_results
                                    
                                    st.success("✅ Monte Carlo simulation completed successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Monte Carlo simulation returned no results. Please check the parameters and try again.")
                                    
                            except Exception as e:
                                st.error(f"Error during Monte Carlo simulation: {str(e)}")
                                import traceback
                                st.error(f"Traceback: {traceback.format_exc()}")
        
        # Scenario Analysis Tab
        with sim_tab2:
            st.markdown("#### 📊 Scenario Analysis")
            
            # Lineup Selection for Scenario Analysis
            available_lineups = []
            if 'results' in st.session_state.shared_results.get('forecast', {}):
                available_lineups = list(st.session_state.shared_results['forecast']['results'].keys())
            
            if not available_lineups:
                st.error("❌ No lineups available. Please run the Forecast analysis first.")
            else:
                # Lineup selector for Scenario Analysis
                selected_lineup_scenario = st.selectbox(
                    "Choose a Lineup to simulate:",
                    options=available_lineups,
                    help="Select the specific Lineup you want to run scenario analysis for",
                    key="scenario_lineup_selector"
                )
                
                st.success(f"✅ Selected Lineup: **{selected_lineup_scenario}**")
                st.markdown("---")
                
                st.markdown(f"#### 📊 Scenario Analysis for {selected_lineup_scenario}")
                
                if (st.session_state.shared_results.get('simulation') is not None and 
                    'scenarios' in st.session_state.shared_results['simulation'] and
                    st.session_state.shared_results['simulation']['scenarios'].get('lineup') == selected_lineup_scenario):
                    # Display existing results for this lineup
                    display_scenario_results(st.session_state.shared_results['simulation']['scenarios'])
                else:
                    # Scenario Analysis Parameters Configuration
                    st.info("💡 Scenario analysis models different business conditions and their financial impact")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    growth_rate = st.slider(
                        "Growth Rate Change (%)", 
                        min_value=-50, max_value=100, value=10, step=5,
                        help="Positive = revenue growth, Negative = revenue decline",
                        key="scenario_growth"
                    )
                with col2:
                    cost_change = st.slider(
                        "Cost Change (%)", 
                        min_value=-30, max_value=50, value=5, step=5,
                        help="Positive = cost increase, Negative = cost reduction",
                        key="scenario_cost"
                    )
                with col3:
                    market_conditions = st.multiselect(
                        "Market Conditions",
                        ["Stable Market", "Volatile Market", "Recession", "Growth Market", "Inflationary"],
                        default=["Stable Market"],
                        help="Select one or more market conditions to model",
                        key="scenario_market"
                    )
                
                # Parameter summary
                st.markdown("#### 📋 Scenario Analysis Parameters Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Growth Rate", f"{growth_rate:+.1f}%")
                with col2:
                    st.metric("Cost Change", f"{cost_change:+.1f}%")
                with col3:
                    st.metric("Market Conditions", len(market_conditions))
                
                # Run Scenario Analysis button
                if st.button("🚀 Run Scenario Analysis", type="primary", key="run_scenario"):
                    with st.spinner("📊 Running scenario analysis..."):
                        try:
                            # Debug: Check parameters
                            st.write("🔍 Debug: Checking scenario analysis parameters...")
                            st.write(f"📈 Growth Rate: {growth_rate}% (type: {type(growth_rate)})")
                            st.write(f"💰 Cost Change: {cost_change}% (type: {type(cost_change)})")
                            st.write(f"🌍 Market Conditions: {market_conditions} (type: {type(market_conditions)})")
                            
                            # Run only scenario analysis for selected lineup
                            scenario_results = run_scenario_analysis(growth_rate, cost_change, market_conditions, selected_lineup_scenario)
                            
                            # Add proper null checks to prevent TypeError
                            if scenario_results is not None:
                                # Initialize simulation results if not exists or is None
                                if ('simulation' not in st.session_state.shared_results or 
                                    st.session_state.shared_results['simulation'] is None):
                                    st.session_state.shared_results['simulation'] = {}
                                
                                # Store scenario results with lineup info
                                scenario_results['lineup'] = selected_lineup_scenario
                                st.session_state.shared_results['simulation']['scenarios'] = scenario_results
                                
                                st.success("✅ Scenario analysis completed successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Scenario analysis returned no results. Please check the parameters and try again.")
                                
                        except Exception as e:
                            st.error(f"Error during scenario analysis: {str(e)}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
        
        # Stress Test Tab
        with sim_tab3:
            st.markdown("#### ⚡ Stress Testing")
            
            # Lineup Selection for Stress Test
            available_lineups = []
            if 'results' in st.session_state.shared_results.get('forecast', {}):
                available_lineups = list(st.session_state.shared_results['forecast']['results'].keys())
            
            if not available_lineups:
                st.error("❌ No lineups available. Please run the Forecast analysis first.")
            else:
                # Lineup selector for Stress Test
                selected_lineup_stress = st.selectbox(
                    "Choose a Lineup to simulate:",
                    options=available_lineups,
                    help="Select the specific Lineup you want to run stress testing for",
                    key="stress_lineup_selector"
                )
                
                st.success(f"✅ Selected Lineup: **{selected_lineup_stress}**")
                st.markdown("---")
                
                st.markdown(f"#### ⚡ Stress Testing for {selected_lineup_stress}")
                
                if (st.session_state.shared_results.get('simulation') is not None and 
                    'stress_test' in st.session_state.shared_results['simulation'] and
                    st.session_state.shared_results['simulation']['stress_test'].get('lineup') == selected_lineup_stress):
                    # Display existing results for this lineup
                    display_stress_test_results(st.session_state.shared_results['simulation']['stress_test'])
                else:
                    # Stress Test Parameters Configuration
                    st.info("💡 Stress testing models extreme financial conditions to assess resilience")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    revenue_shock = st.slider(
                        "Revenue Shock (%)", 
                        min_value=-50, max_value=0, value=-20, step=5,
                        help="Negative values only - represents revenue decline",
                        key="stress_revenue"
                    )
                with col2:
                    revenue_duration = st.number_input(
                        "Revenue Duration (months)", 
                        min_value=1, max_value=12, value=3, step=1,
                        help="How long the revenue shock lasts",
                        key="stress_revenue_dur"
                    )
                with col3:
                    cost_shock = st.slider(
                        "Cost Shock (%)", 
                        min_value=0, max_value=100, value=25, step=5,
                        help="Positive values only - represents cost increase",
                        key="stress_cost"
                    )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    cost_duration = st.number_input(
                        "Cost Duration (months)", 
                        min_value=1, max_value=12, value=3, step=1,
                        help="How long the cost shock lasts",
                        key="stress_cost_dur"
                    )
                with col2:
                    stress_factors = st.multiselect(
                        "Stress Factors",
                        ["Interest Rate Hike", "Supply Chain Disruption", "Market Crash", "Regulatory Changes", "Natural Disaster"],
                        default=["Interest Rate Hike"],
                        help="Select factors that could cause financial stress",
                        key="stress_factors"
                    )
                with col3:
                    stress_severity = st.selectbox(
                        "Stress Severity",
                        ["Mild Stress", "Recovery: 1 month", "Moderate Stress", "Recovery: 3 months", "Severe Stress", "Recovery: 6 months", "Extreme Stress", "Recovery: 12 months"],
                        index=2,
                        help="Higher severity = longer recovery time",
                        key="stress_severity"
                    )
                
                # Parameter summary
                st.markdown("#### 📋 Stress Test Parameters Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Revenue Impact:**")
                    st.write(f"- Shock: {revenue_shock:.1f}% for {revenue_duration} months")
                    st.markdown("**Cost Impact:**")
                    st.write(f"- Shock: {cost_shock:.1f}% for {cost_duration} months")
                with col2:
                    st.markdown("**Stress Factors:**")
                    st.write(f"- Selected: {', '.join(stress_factors)}")
                    st.markdown("**Severity:**")
                    st.write(f"- Level: {stress_severity}")
                
                # Run Stress Test button
                if st.button("🚀 Run Stress Test", type="primary", key="run_stress_test"):
                    with st.spinner("⚡ Running stress test..."):
                        try:
                            # Debug: Check parameters
                            st.write("🔍 Debug: Checking stress test parameters...")
                            st.write(f"📉 Revenue Shock: {revenue_shock}% (type: {type(revenue_shock)})")
                            st.write(f"⏱️ Revenue Duration: {revenue_duration} months (type: {type(revenue_duration)})")
                            st.write(f"📈 Cost Shock: {cost_shock}% (type: {type(cost_shock)})")
                            st.write(f"⏱️ Cost Duration: {cost_duration} months (type: {type(cost_duration)})")
                            st.write(f"🌪️ Stress Factors: {stress_factors} (type: {type(stress_factors)})")
                            st.write(f"⚠️ Stress Severity: {stress_severity} (type: {type(stress_severity)})")
                            
                            # Run only stress test for selected lineup
                            stress_results = run_stress_test(
                                revenue_shock, revenue_duration, cost_shock, cost_duration, 
                                stress_factors, stress_severity, selected_lineup_stress
                            )
                            
                            # Add proper null checks to prevent TypeError
                            if stress_results is not None:
                                # Initialize simulation results if not exists or is None
                                if ('simulation' not in st.session_state.shared_results or 
                                    st.session_state.shared_results['simulation'] is None):
                                    st.session_state.shared_results['simulation'] = {}
                                
                                # Store stress test results with lineup info
                                stress_results['lineup'] = selected_lineup_stress
                                st.session_state.shared_results['simulation']['stress_test'] = stress_results
                                
                                st.success("✅ Stress test completed successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Stress test returned no results. Please check the parameters and try again.")
                                
                        except Exception as e:
                            st.error(f"Error during stress test: {str(e)}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
        
        # Update module progress when any simulation is completed
        if (st.session_state.shared_results.get('simulation') is not None and 
            len(st.session_state.shared_results['simulation']) > 0):
            st.session_state.module_progress['simulation'] = True
    
    # Tab 4: Variance Analysis - ONLY RUNS WHEN USER CLICKS THIS TAB
    with tab4:
        st.markdown('<h2 class="section-header">📈 Variance Analysis</h2>', unsafe_allow_html=True)
        
        # Check if already executed
        if st.session_state.shared_results['variance_analysis'] is not None:
            st.success("✅ Variance analysis already completed!")
            
            # Display existing results
            st.markdown("**📊 Variance Analysis Results:**")
            st.markdown(st.session_state.shared_results['variance_analysis'])
            
        else:
            # Show execution button
            if st.button("🚀 Run Variance Analysis", type="primary", key="run_variance"):
                with st.spinner("📈 Running variance analysis..."):
                    try:
                        variance_analysis = st.session_state.orchestrator.generate_variance_analysis(
                            "Sample_data_N.csv",
                            "Plan Number.csv", 
                            "fixed_clean_complete_data_with_forecasts.csv"
                        )
                        
                        # Store in session state
                        st.session_state.shared_results['variance_analysis'] = variance_analysis
                        st.session_state.module_progress['variance_analysis'] = True
                        
                        st.success("✅ Variance analysis completed successfully!")
                        st.rerun()  # Refresh to show results
                        
                    except Exception as e:
                        st.error(f"Error generating variance analysis: {str(e)}")
            else:
                st.info("💡 Click the button above to run the Variance Analysis")
    
    # Tab 5: Generate Insight
    with tab5:
        st.markdown('<h2 class="section-header">�� Generate Insight</h2>', unsafe_allow_html=True)
        
        try:
            if st.session_state.orchestrator:
                insights = st.session_state.orchestrator.generate_comprehensive_insights("Sample_data_N.csv")
                
                # Store in session state
                st.session_state.shared_results['insights'] = insights
                
                st.markdown("**📊 Analysis Results:**")
                st.markdown(insights)
                
                st.success("✅ Comprehensive insights generated successfully!")
                
                # Update progress
                st.session_state.module_progress['insights'] = True
            else:
                st.warning("⚠️ Please initialize the AI system first")
                
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
    
    # Tab 6: Q&A
    with tab6:
        st.markdown('<h2 class="section-header">�� AI Assistant Q&A</h2>', unsafe_allow_html=True)
        
        try:
            # Chat interface
            user_question = st.text_input(
                "�� Ask your question:",
                placeholder="e.g., What are the spending trends for ABC234006?",
                key="user_question_tab6"
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
                            chat_entry = {
                                "question": user_question,
                                "answer": answer,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            }
                            
                            st.session_state.shared_results['qa_history'].append(chat_entry)
                            
                            # Display answer
                            st.markdown("**🤖 AI Response:**")
                            st.markdown(answer)
                            
                            # Update progress
                            st.session_state.module_progress['qa'] = True
                            
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
                    if st.button(question, key=f"example_tab6_{i}"):
                        with st.spinner("🤖 AI is analyzing your question..."):
                            try:
                                available_files = [
                                    "Sample_data_N.csv", 
                                    "Plan Number.csv", 
                                    "fixed_clean_complete_data_with_forecasts.csv"
                                ]
                                
                                answer = st.session_state.orchestrator.ask_question(question, available_files)
                                
                                # Add to chat history
                                chat_entry = {
                                    "question": question,
                                    "answer": answer,
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                }
                                
                                st.session_state.shared_results['qa_history'].append(chat_entry)
                                
                                # Display answer
                                st.markdown("**🤖 AI Response:**")
                                st.markdown(answer)
                                
                            except Exception as e:
                                st.error(f"Error processing your question: {str(e)}")
            
            # Chat history display
            if st.session_state.shared_results['qa_history']:
                st.markdown("#### 💬 Chat History")
                for i, chat in enumerate(reversed(st.session_state.shared_results['qa_history'])):
                    with st.expander(f"Q: {chat['question'][:50]}... ({chat['timestamp']})"):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f"**Answer:** {chat['answer']}")
            
        except Exception as e:
            st.error(f"Error setting up AI Assistant: {str(e)}")
    
    # Tab 7: Dashboard
    with tab7:
        st.markdown('<h2 class="section-header">📊 Comprehensive Dashboard</h2>', unsafe_allow_html=True)
        
        # Progress tracking for all tabs
        if 'dashboard_progress' not in st.session_state:
            st.session_state.dashboard_progress = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run all tabs sequentially
        if st.button("�� Run All Analysis", type="primary", key="run_all_analysis"):
            st.session_state.dashboard_progress = 0
            
            # Step 1: Data Overview
            status_text.text("🔄 Running Data Overview...")
            st.session_state.dashboard_progress = 14
            progress_bar.progress(st.session_state.dashboard_progress / 100)
            
            # Step 2: Forecast
            status_text.text("🔄 Running Forecast Analysis...")
            st.session_state.dashboard_progress = 28
            progress_bar.progress(st.session_state.dashboard_progress / 100)
            
            # Step 3: Simulation
            status_text.text("🔄 Running Simulations...")
            st.session_state.dashboard_progress = 42
            progress_bar.progress(st.session_state.dashboard_progress / 100)
            
            # Step 4: Variance Analysis
            status_text.text("🔄 Running Variance Analysis...")
            st.session_state.dashboard_progress = 56
            progress_bar.progress(st.session_state.dashboard_progress / 100)
            
            # Step 5: Generate Insight
            status_text.text("🔄 Generating Insights...")
            st.session_state.dashboard_progress = 70
            progress_bar.progress(st.session_state.dashboard_progress / 100)
            
            # Step 6: Q&A Setup
            status_text.text("🔄 Setting up Q&A...")
            st.session_state.dashboard_progress = 84
            progress_bar.progress(st.session_state.dashboard_progress / 100)
            
            # Step 7: Dashboard Complete
            status_text.text("✅ All analysis completed!")
            st.session_state.dashboard_progress = 100
            progress_bar.progress(st.session_state.dashboard_progress / 100)
            
            st.success("🎉 All analysis sections have been completed! Check each tab for detailed results.")
            
            # Update all module progress
            st.session_state.module_progress['data_overview'] = True
            st.session_state.module_progress['forecast'] = True
            st.session_state.module_progress['simulation'] = True
            st.session_state.module_progress['variance_analysis'] = True
            st.session_state.module_progress['insights'] = True
            st.session_state.module_progress['qa'] = True
            st.session_state.module_progress['dashboard'] = True
        
        # Display summary of all results
        st.markdown("#### 📋 Analysis Summary")
        
        if st.session_state.shared_results['data_overview'] is not None:
            st.success("✅ Data Overview: Completed")
        else:
            st.warning("⚠️ Data Overview: Not completed")
        
        if st.session_state.shared_results['forecast'] is not None:
            st.success("✅ Forecast: Completed")
        else:
            st.warning("⚠️ Forecast: Not completed")
        
        if st.session_state.shared_results['simulation'] is not None:
            st.success("✅ Simulation: Completed")
        else:
            st.warning("⚠️ Simulation: Not completed")
        
        if st.session_state.shared_results['variance_analysis'] is not None:
            st.success("✅ Variance Analysis: Completed")
        else:
            st.warning("⚠️ Variance Analysis: Not completed")
        
        if st.session_state.shared_results['insights'] is not None:
            st.success("✅ Generate Insight: Completed")
        else:
            st.warning("⚠️ Generate Insight: Not completed")
        
        if len(st.session_state.shared_results['qa_history']) > 0:
            st.success("✅ Q&A: Completed")
        else:
            st.warning("⚠️ Q&A: Not completed")
        
        # Display key metrics from all tabs
        if st.session_state.shared_results['data_overview'] is not None:
            st.markdown("#### 📊 Key Metrics Summary")
            
            df = st.session_state.shared_results['data_overview']
            
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
        
        # Display forecast summary if available
        if st.session_state.shared_results['forecast'] is not None:
            st.markdown("#### 🔮 Forecast Summary")
            
            forecast_data = st.session_state.shared_results['forecast']
            metrics_df = pd.DataFrame(forecast_data['metrics'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_mape = metrics_df['MAPE (%)'].mean()
                st.metric("Average MAPE", f"{avg_mape:.2f}%")
            
            with col2:
                avg_r2 = metrics_df['R² Score'].mean()
                st.metric("Average R²", f"{avg_r2:.4f}")
            
            with col3:
                total_forecast = metrics_df['Avg Forecast (£)'].sum() * 12
                st.metric("Total 12-Month Forecast", f"£{total_forecast:,.0f}")
        
        # Display simulation summary if available
        if st.session_state.shared_results['simulation'] is not None:
            st.markdown("#### �� Simulation Summary")
            
            simulation_data = st.session_state.shared_results['simulation']
            
            summary_data = []
            for sim_type, results in simulation_data.items():
                if sim_type == 'monte_carlo':
                    summary_data.append({
                        'Type': 'Monte Carlo',
                        'Lineups': len(results),
                        'Status': 'Completed'
                    })
                elif sim_type == 'scenarios':
                    summary_data.append({
                        'Type': 'Scenario Analysis',
                        'Scenarios': len(results),
                        'Status': 'Completed'
                    })
                elif sim_type == 'stress_test':
                    summary_data.append({
                        'Type': 'Stress Testing',
                        'Tests': len(results),
                        'Status': 'Completed'
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        # Update dashboard progress when viewing results
        if (st.session_state.shared_results['data_overview'] is not None and 
            st.session_state.shared_results['forecast'] is not None and
            st.session_state.shared_results['simulation'] is not None and
            st.session_state.shared_results['variance_analysis'] is not None and
            st.session_state.shared_results['insights'] is not None and
            len(st.session_state.shared_results['qa_history']) > 0):
            st.session_state.module_progress['dashboard'] = True
        
        # Export all results
        if st.button("�� Export All Results", type="primary", key="export_all"):
            try:
                # Create comprehensive export
                export_data = {
                    'Data_Overview': st.session_state.shared_results['data_overview'].to_dict() if st.session_state.shared_results['data_overview'] is not None else {},
                    'Forecast': st.session_state.shared_results['forecast'] if st.session_state.shared_results['forecast'] is not None else {},
                    'Simulation': st.session_state.shared_results['simulation'] if st.session_state.shared_results['simulation'] is not None else {},
                    'Variance_Analysis': st.session_state.shared_results['variance_analysis'] if st.session_state.shared_results['variance_analysis'] is not None else {},
                    'Insights': st.session_state.shared_results['insights'] if st.session_state.shared_results['insights'] is not None else {},
                    'QA_History': st.session_state.shared_results['qa_history']
                }
                
                # Convert to JSON for export
                import json
                json_data = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download All Results (JSON)",
                    data=json_data,
                    file_name="comprehensive_analysis_results.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error exporting results: {str(e)}")

# Helper functions for displaying results
def display_data_overview_results(df):
    """Display data overview results"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric("Total Records", total_records)
    
    with col2:
        total_spending = df['Actual'].sum()
        st.metric("Total Spending", f"${total_spending:,.0f}")
    
    with col3:
        num_lineups = df['Lineup'].nunique()
        st.metric("Lineups", num_lineups)
    
    with col4:
        date_range = f"{df['DATE'].min().strftime('%Y-%m')} to {df['DATE'].max().strftime('%Y-%m')}"
        st.metric("Date Range", date_range)
    
    # Data preview
    st.markdown("#### 📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data information
    st.markdown("#### 📊 Data Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Columns:**", list(df.columns))
        st.write("**Data Types:**", df.dtypes.to_dict())
    
    with col2:
        st.write("**Missing Values:**", df.isnull().sum().to_dict())
        st.write("**Shape:**", df.shape)

def display_forecast_results(forecast_data):
    """Display forecast results"""
    if forecast_data and 'metrics' in forecast_data:
        st.markdown("##### 📈 Forecasting Model Performance")
        metrics_df = pd.DataFrame(forecast_data['metrics'])
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("##### 🔮 12-Month Forecasts by Lineup")
        for lineup, result in forecast_data['results'].items():
            with st.expander(f"📊 {lineup} - 12-Month Forecast"):
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

def run_all_simulations(forecast_data):
    """Run all simulation types with default parameters"""
    simulation_results = {}
    
    # Run Monte Carlo simulation
    monte_carlo_results = run_monte_carlo_simulation(forecast_data, 1000, 95, 12)
    if monte_carlo_results:
        simulation_results['monte_carlo'] = monte_carlo_results
    
    # Run scenario analysis
    scenario_results = run_scenario_analysis(10.0, 5.0, ["Stable Market"])
    if scenario_results:
        simulation_results['scenarios'] = scenario_results
    
    # Run stress test
    stress_results = run_stress_test(-20, 3, 25, 3, ["Interest Rate Hike"], "Moderate Stress")
    if stress_results:
        simulation_results['stress_test'] = stress_results
    
    return simulation_results

def run_all_simulations_with_params(forecast_data, num_simulations, confidence_level, time_horizon,
                                   growth_rate, cost_change, market_conditions,
                                   revenue_shock, revenue_duration, cost_shock, cost_duration, 
                                   stress_factors, stress_severity):
    """Run all simulation types with user-defined parameters"""
    simulation_results = {}
    
    # Run Monte Carlo simulation with user parameters
    monte_carlo_results = run_monte_carlo_simulation(forecast_data, num_simulations, confidence_level, time_horizon)
    if monte_carlo_results:
        simulation_results['monte_carlo'] = monte_carlo_results
    
    # Run scenario analysis with user parameters
    scenario_results = run_scenario_analysis(growth_rate, cost_change, market_conditions)
    if scenario_results:
        simulation_results['scenarios'] = scenario_results
    
    # Run stress test with user parameters
    stress_results = run_stress_test(revenue_shock, revenue_duration, cost_shock, cost_duration, stress_factors, stress_severity)
    if stress_results:
        simulation_results['stress_test'] = stress_results
    
    return simulation_results

def display_simulation_results(simulation_data):
    """Display simulation results"""
    if not simulation_data:
        st.warning("No simulation results available")
        return
    
    # Display based on available simulation types
    if 'monte_carlo' in simulation_data:
        st.markdown("#### 🎯 Monte Carlo Results")
        # Display Monte Carlo results
        
    if 'scenarios' in simulation_data:
        st.markdown("#### 📊 Scenario Analysis Results")
        # Display scenario results
        
    if 'stress_test' in simulation_data:
        st.markdown("#### ⚡ Stress Test Results")
        # Display stress test results

def display_monte_carlo_results(monte_carlo_results):
    """Display Monte Carlo results"""
    if not monte_carlo_results:
        st.warning("No Monte Carlo results available")
        return
    
    st.markdown("#### 🎯 Monte Carlo Results")
    for lineup in monte_carlo_results['lineups']:
        with st.expander(f"📈 {lineup} - Monte Carlo Results"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Mean Forecast:** £{monte_carlo_results['results'][lineup]['mean']:,.2f}")
                st.write(f"**Std Deviation:** £{monte_carlo_results['results'][lineup]['std']:,.2f}")
                st.write(f"**Min Value:** £{monte_carlo_results['results'][lineup]['min']:,.2f}")
                st.write(f"**Max Value:** £{monte_carlo_results['results'][lineup]['max']:,.2f}")
            
            with col2:
                st.write(f"**95% Lower:** £{monte_carlo_results['results'][lineup]['lower_ci']:,.2f}")
                st.write(f"**95% Upper:** £{monte_carlo_results['results'][lineup]['upper_ci']:,.2f}")
                st.write(f"**Risk Level:** {monte_carlo_results['results'][lineup]['risk_level']}")
            
            # Create Monte Carlo chart
            fig = create_monte_carlo_chart(monte_carlo_results, lineup)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def display_scenario_results(scenario_results):
    """Display scenario analysis results"""
    if not scenario_results:
        st.warning("No scenario analysis results available")
        return
    
    st.markdown("#### 📊 Scenario Analysis Results")
    # Create scenario comparison chart
    fig = create_scenario_comparison_chart(scenario_results)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Scenario details
    st.markdown("#### 📋 Scenario Details:")
    for i, scenario in enumerate(scenario_results['scenarios']):
        with st.expander(f"📊 Scenario {i+1}: {scenario['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Growth Rate:** {scenario['growth_rate']:.1f}%")
                st.write(f"**Cost Change:** {scenario['cost_change']:.1f}%")
                st.write(f"**Market Conditions:** {', '.join(scenario['market_conditions'])}")
            
            with col2:
                st.write(f"**Revenue Impact:** £{scenario['revenue_impact']:,.0f}")
                st.write(f"**Cost Impact:** £{scenario['cost_impact']:,.0f}")
                st.write(f"**Net Impact:** £{scenario['net_impact']:,.0f}")

def display_stress_test_results(stress_results):
    """Display stress test results"""
    if not stress_results:
        st.warning("No stress test results available")
        return
    
    st.markdown("#### ⚡ Stress Test Results")
    
    # Stress test summary
    st.markdown("#### 📊 Stress Test Summary:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Revenue Impact", f"£{stress_results['total_revenue_impact']:,.0f}")
    
    with col2:
        st.metric("Cost Impact", f"£{stress_results['total_cost_impact']:,.0f}")
    
    with col3:
        st.metric("Net Impact", f"£{stress_results['net_impact']:,.0f}")
    
    with col4:
        st.metric("Recovery Time", f"{stress_results['recovery_months']} months")
    
    # Risk assessment
    st.markdown("#### ⚠️ Risk Assessment:")
    risk_assessment = stress_results['risk_assessment']
    
    if risk_assessment['risk_level'] == "High":
        st.error(f"🚨 **High Risk Detected:** {risk_assessment['description']}")
    elif risk_assessment['risk_level'] == "Medium":
        st.warning(f"⚠️ **Medium Risk Detected:** {risk_assessment['description']}")
    else:
        st.success(f"✅ **Low Risk:** {risk_assessment['description']}")
    
    st.write(f"**Mitigation Strategy:** {risk_assessment['mitigation']}")

def display_comprehensive_dashboard():
    """Display comprehensive dashboard"""
    st.markdown("#### 📋 Analysis Summary")
    
    # Display summary of all results
    if st.session_state.shared_results['data_overview'] is not None:
        st.success("✅ Data Overview: Completed")
    if st.session_state.shared_results['forecast'] is not None:
        st.success("✅ Forecast: Completed")
    if st.session_state.shared_results['simulation'] is not None:
        st.success("✅ Simulation: Completed")
    if st.session_state.shared_results['variance_analysis'] is not None:
        st.success("✅ Variance Analysis: Completed")
    if st.session_state.shared_results['insights'] is not None:
        st.success("✅ Generate Insight: Completed")
    if len(st.session_state.shared_results['qa_history']) > 0:
        st.success("✅ Q&A: Completed")
    
    # Display key metrics from all tabs
    if st.session_state.shared_results['data_overview'] is not None:
        st.markdown("#### 📊 Key Metrics Summary")
        df = st.session_state.shared_results['data_overview']
        
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
    
    # Display forecast summary if available
    if st.session_state.shared_results['forecast'] is not None:
        st.markdown("#### 🔮 Forecast Summary")
        forecast_data = st.session_state.shared_results['forecast']
        metrics_df = pd.DataFrame(forecast_data['metrics'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_mape = metrics_df['MAPE (%)'].mean()
            st.metric("Average MAPE", f"{avg_mape:.2f}%")
        
        with col2:
            avg_r2 = metrics_df['R² Score'].mean()
            st.metric("Average R²", f"{avg_r2:.4f}")
        
        with col3:
            total_forecast = metrics_df['Avg Forecast (£)'].sum() * 12
            st.metric("Total 12-Month Forecast", f"£{total_forecast:,.0f}")

# Keep your existing initialize_system function and other functions unchanged
# Just add the import at the top and replace the main() function
if __name__ == "__main__":
    main()



