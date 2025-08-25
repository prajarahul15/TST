# Financial Data Insight Generation System
# Agent Orchestration with LangChain + Jupyter

"""
This script implements a multi-agent system for generating comprehensive financial insights using:
- Data Analysis Agent: Processes CSV data and performs statistical analysis
- LLM Insight Agent: Generates natural language insights using Gemini
- Visualization Agent: Creates charts and plots
- Variance Analysis Agent: Compares Actual vs Plan vs Forecast

Architecture: Profile → Site → Lineup hierarchy analysis with agent collaboration
"""

# =============================================================================
# CELL 1: Install required packages
# =============================================================================
# !pip install -r requirements_dev.txt

# =============================================================================
# CELL 2: Import required libraries
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Environment setup
from dotenv import load_dotenv
load_dotenv()

print("✅ All libraries imported successfully!")

# =============================================================================
# CELL 3: Load environment variables and initialize LLM
# =============================================================================
# Load environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in .env file")
    print("Please create a .env file with your Gemini API key")
else:
    print("✅ Gemini API key loaded successfully")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1,
    convert_system_message_to_human=True
)

print(f"✅ LLM initialized: {llm}")

# =============================================================================
# CELL 4: Define data models for structured output
# =============================================================================
class FinancialInsight(BaseModel):
    """Structured financial insight output"""
    insight_type: str = Field(description="Type of insight (trend, pattern, variance, recommendation)")
    hierarchy_level: str = Field(description="Hierarchy level (Profile, Site, Lineup)")
    description: str = Field(description="Detailed description of the insight")
    significance: str = Field(description="Business significance of the insight")
    data_support: str = Field(description="Data points supporting the insight")
    recommendations: str = Field(description="Actionable recommendations based on the insight")

class VarianceAnalysis(BaseModel):
    """Structured variance analysis output"""
    comparison_type: str = Field(description="Type of comparison (Actual vs Plan, Actual vs Forecast)")
    hierarchy_level: str = Field(description="Hierarchy level being analyzed")
    variance_percentage: float = Field(description="Percentage variance")
    variance_amount: float = Field(description="Absolute variance amount")
    explanation: str = Field(description="Explanation of the variance")
    impact_assessment: str = Field(description="Business impact assessment")
    mitigation_strategies: str = Field(description="Strategies to address the variance")

print("✅ Data models defined successfully!")

# =============================================================================
# CELL 5: Define agent tools
# =============================================================================
@tool
def load_and_prepare_data(file_path: str) -> str:
    """Load and prepare CSV data for analysis"""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(how='all')
        
        # Handle different date formats automatically
        try:
            # Try the original format first
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        except ValueError:
            try:
                # Try ISO format
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
            except ValueError:
                # Try to infer format
                df['DATE'] = pd.to_datetime(df['DATE'])
        
        df = df.sort_values(['Lineup', 'DATE'])
        
        # Basic data info
        info = f"""
        Data loaded successfully from {file_path}
        - Total records: {len(df)}
        - Date range: {df['DATE'].min().strftime('%Y-%m')} to {df['DATE'].max().strftime('%Y-%m')}
        - Lineups: {', '.join(df['Lineup'].unique())}
        - Columns: {', '.join(df.columns)}
        """
        return info
    except Exception as e:
        return f"Error loading data: {str(e)}"

@tool
def get_data_statistics(file_path: str, hierarchy_level: str = "Lineup") -> str:
    """Get statistical summary for specified hierarchy level"""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(how='all')
        
        # Handle different date formats automatically
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        except ValueError:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
            except ValueError:
                df['DATE'] = pd.to_datetime(df['DATE'])
        
        if hierarchy_level == "Lineup":
            stats = df.groupby('Lineup')['Actual'].agg(['sum', 'mean', 'std', 'min', 'max']).round(2)
        elif hierarchy_level == "Site":
            stats = df.groupby('Site')['Actual'].agg(['sum', 'mean', 'std', 'min', 'max']).round(2)
        elif hierarchy_level == "Profile":
            stats = df.groupby('Profile')['Actual'].agg(['sum', 'mean', 'std', 'min', 'max']).round(2)
        
        return f"Statistics for {hierarchy_level} level:\n{stats.to_string()}"
    except Exception as e:
        return f"Error getting statistics: {str(e)}"

@tool
def analyze_trends(file_path: str, hierarchy_level: str = "Lineup") -> str:
    """Analyze spending trends over time for specified hierarchy level"""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(how='all')
        
        # Handle different date formats automatically
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        except ValueError:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
            except ValueError:
                df['DATE'] = pd.to_datetime(df['DATE'])
        
        df['Month'] = df['DATE'].dt.to_period('M')
        
        if hierarchy_level == "Lineup":
            trends = df.groupby(['Lineup', 'Month'])['Actual'].sum().reset_index()
        elif hierarchy_level == "Site":
            trends = df.groupby(['Site', 'Month'])['Actual'].sum().reset_index()
        elif hierarchy_level == "Profile":
            trends = df.groupby(['Profile', 'Month'])['Actual'].sum().reset_index()
        
        # Calculate growth rates
        trends['Growth_Rate'] = trends.groupby(hierarchy_level)['Actual'].pct_change() * 100
        
        return f"Trend analysis for {hierarchy_level} level:\n{trends.tail(10).to_string()}"
    except Exception as e:
        return f"Error analyzing trends: {str(e)}"

@tool
def create_visualization(file_path: str, chart_type: str = "line", hierarchy_level: str = "Lineup") -> str:
    """Create visualization charts for the data"""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(how='all')
        
        # Handle different date formats automatically
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        except ValueError:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
            except ValueError:
                df['DATE'] = pd.to_datetime(df['DATE'])
        
        if chart_type == "line":
            if hierarchy_level == "Lineup":
                fig = px.line(df, x='DATE', y='Actual', color='Lineup', title='Spending Trends by Lineup')
            elif hierarchy_level == "Site":
                fig = px.line(df, x='DATE', y='Actual', color='Site', title='Spending Trends by Site')
            
            fig.write_html(f"spending_trends_{hierarchy_level.lower()}.html")
            return f"Line chart created and saved as spending_trends_{hierarchy_level.lower()}.html"
        
        elif chart_type == "bar":
            monthly_totals = df.groupby([df['DATE'].dt.to_period('M'), hierarchy_level])['Actual'].sum().reset_index()
            monthly_totals['DATE'] = monthly_totals['DATE'].astype(str)
            
            fig = px.bar(monthly_totals, x='DATE', y='Actual', color=hierarchy_level, 
                        title=f'Monthly Spending by {hierarchy_level}')
            fig.write_html(f"monthly_spending_{hierarchy_level.lower()}.html")
            return f"Bar chart created and saved as monthly_spending_{hierarchy_level.lower()}.html"
        
        return "Visualization created successfully"
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

@tool
def perform_variance_analysis(actual_file: str, plan_file: str, forecast_file: str) -> str:
    """Perform comprehensive variance analysis between Actual, Plan, and Forecast data"""
    try:
        # Load all three datasets
        actual_df = pd.read_csv(actual_file)
        plan_df = pd.read_csv(plan_file)
        forecast_df = pd.read_csv(forecast_file)
        
        # Handle different date formats for each file
        try:
            actual_df['DATE'] = pd.to_datetime(actual_df['DATE'], format='%d-%m-%Y')
        except ValueError:
            actual_df['DATE'] = pd.to_datetime(actual_df['DATE'])
            
        try:
            plan_df['DATE'] = pd.to_datetime(plan_df['DATE'], format='%d-%m-%Y')
        except ValueError:
            plan_df['DATE'] = pd.to_datetime(plan_df['DATE'])
            
        try:
            forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE'], format='%Y-%m-%d')
        except ValueError:
            forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE'])
        
        # Filter to 2025 data only (where we have Plan and Forecast)
        forecast_2025 = forecast_df[forecast_df['DATE'].dt.year == 2025].copy()
        
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
            return "No variance analysis results found. Check data availability."
        
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
        
        return "\n".join(insights) + f"\n\nDetailed variance data:\n{variance_df.to_string(index=False)}"
        
    except Exception as e:
        return f"Error performing variance analysis: {str(e)}"

# Create a standalone variance analysis function for direct calls
def perform_variance_analysis_direct(actual_file: str, plan_file: str, forecast_file: str) -> str:
    """Perform comprehensive variance analysis between Actual, Plan, and Forecast data (direct call version)"""
    try:
        # Load all three datasets
        actual_df = pd.read_csv(actual_file)
        plan_df = pd.read_csv(plan_file)
        forecast_df = pd.read_csv(forecast_file)
        
        # Handle different date formats for each file
        try:
            actual_df['DATE'] = pd.to_datetime(actual_df['DATE'], format='%d-%m-%Y')
        except ValueError:
            actual_df['DATE'] = pd.to_datetime(actual_df['DATE'])
            
        try:
            plan_df['DATE'] = pd.to_datetime(plan_df['DATE'], format='%d-%m-%Y')
        except ValueError:
            plan_df['DATE'] = pd.to_datetime(plan_df['DATE'])
            
        try:
            forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE'], format='%Y-%m-%d')
        except ValueError:
            forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE'])
        
        # Filter to 2025 data only (where we have Plan and Forecast)
        forecast_2025 = forecast_df[forecast_df['DATE'].dt.year == 2025].copy()
        
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
            return "No variance analysis results found. Check data availability."
        
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
        
        return "\n".join(insights) + f"\n\nDetailed variance data:\n{variance_df.to_string(index=False)}"
        
    except Exception as e:
        return f"Error performing variance analysis: {str(e)}"

# Create a standalone visualization function for direct calls
def create_visualization_direct(file_path: str, chart_type: str = "line", hierarchy_level: str = "Lineup") -> str:
    """Create visualization charts for the data (direct call version)"""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(how='all')
        
        # Handle different date formats automatically
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
        except ValueError:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
            except ValueError:
                df['DATE'] = pd.to_datetime(df['DATE'])
        
        if chart_type == "line":
            if hierarchy_level == "Lineup":
                fig = px.line(df, x='DATE', y='Actual', color='Lineup', title='Spending Trends by Lineup')
            elif hierarchy_level == "Site":
                fig = px.line(df, x='DATE', y='Actual', color='Site', title='Spending Trends by Site')
            
            fig.write_html(f"spending_trends_{hierarchy_level.lower()}.html")
            return f"Line chart created and saved as spending_trends_{hierarchy_level.lower()}.html"
        
        elif chart_type == "bar":
            monthly_totals = df.groupby([df['DATE'].dt.to_period('M'), hierarchy_level])['Actual'].sum().reset_index()
            monthly_totals['DATE'] = monthly_totals['DATE'].astype(str)
            
            fig = px.bar(monthly_totals, x='DATE', y='Actual', color=hierarchy_level, 
                        title=f'Monthly Spending by {hierarchy_level}')
            fig.write_html(f"monthly_spending_{hierarchy_level.lower()}.html")
            return f"Bar chart created and saved as monthly_spending_{hierarchy_level.lower()}.html"
        
        return "Visualization created successfully"
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

print("✅ Agent tools defined successfully!")

# =============================================================================
# CELL 6: Create agent orchestration system
# =============================================================================
class FinancialInsightOrchestrator:
    """Main orchestrator for financial insight generation"""
    
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
        
        # Create the agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _create_agent(self):
        """Create the LangChain agent with tools"""
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
        """Generate comprehensive insights for the financial data"""
        print("\n ******Inside the summary generation function****\n",file_path)
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
        
        result = self.agent_executor.invoke({"input": prompt})
        return result["output"]
    
    def generate_variance_analysis(self, actual_file: str, plan_file: str, forecast_file: str) -> str:
        """Generate variance analysis between Actual, Plan, and Forecast using the agent"""
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
            result = self.agent_executor.invoke({"input": prompt})
            return result["output"]
        except Exception as e:
            return f"Error in variance analysis: {str(e)}"
    
    def ask_question(self, user_question: str, context_files: list = None) -> str:
        """Conversational Q&A agent that answers user questions about financial data"""
        try:
            # Build context-aware prompt
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
            
            # Get response from agent
            result = self.agent_executor.invoke({"input": context_info})
            return result["output"]
            
        except Exception as e:
            return f"Error processing your question: {str(e)}"

print("✅ Agent orchestrator created successfully!")

# =============================================================================
# CELL 7: Initialize and test the system
# =============================================================================
# Initialize the orchestrator
orchestrator = FinancialInsightOrchestrator(llm)

print("✅ Financial Insight Orchestrator initialized successfully!")
print("Ready to generate insights!")

# =============================================================================
# CELL 8: Generate comprehensive insights (run this cell to start analysis)
# =============================================================================
# Generate comprehensive insights
print("🔍 Generating comprehensive insights...")
comprehensive_insights = orchestrator.generate_comprehensive_insights("Sample_data_N.csv")
print("\n" + "="*80)
print("COMPREHENSIVE INSIGHTS")
print("="*80)
print(comprehensive_insights)

# =============================================================================
# CELL 9: Generate variance analysis (run this cell for variance analysis)
# =============================================================================
# Generate variance analysis
print("📊 Generating variance analysis...")
variance_analysis = orchestrator.generate_variance_analysis(
    "Sample_data_N.csv",
    "Plan Number.csv", 
    "fixed_clean_complete_data_with_forecasts.csv"
)
print("\n" + "="*80)
print("VARIANCE ANALYSIS")
print("="*80)
print(variance_analysis)

# =============================================================================
# CELL 10: Create visualizations (run this cell to generate charts)
# =============================================================================
# Create various visualizations
print("📈 Creating visualizations...")

# Line charts for trends
create_visualization_direct("Sample_data_N.csv", "line", "Lineup")
create_visualization_direct("Sample_data_N.csv", "line", "Site")

# Bar charts for monthly spending
create_visualization_direct("Sample_data_N.csv", "bar", "Lineup")
create_visualization_direct("Sample_data_N.csv", "bar", "Site")

print("✅ All visualizations created successfully!")
print("Check the generated HTML files for interactive charts.")
