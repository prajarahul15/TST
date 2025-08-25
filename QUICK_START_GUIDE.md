# Quick Start Guide: Financial Insights Agents

## 🚀 Getting Started

This guide will help you set up and run the Financial Data Insight Generation System using Jupyter + LangChain with agent orchestration.

## 📁 File Structure

```
E:\TS\
├── .env                                    # Environment variables (GEMINI_API_KEY)
├── Sample_data_N.csv                       # Historical actual data (2022-2024)
├── Plan Number.csv                         # Plan data for 2025
├── fixed_clean_complete_data_with_forecasts.csv  # Complete data with forecasts
├── financial_insights_agents.py            # Core agent system code
├── financial_insights_agents.ipynb         # Jupyter notebook (auto-generated)
├── create_notebook.py                      # Script to generate notebook
├── test_agents.py                          # Test basic functionality
├── test_visualization.py                   # Test visualization functions
├── test_variance_analysis.py               # Test variance analysis
├── requirements_dev_simple.txt              # Development dependencies
└── QUICK_START_GUIDE.md                    # This file
```

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_dev_simple.txt
```

### 2. Set Up Environment
Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

### 3. Generate Jupyter Notebook
```bash
python create_notebook.py
```

### 4. Launch Jupyter Lab
```bash
jupyter lab financial_insights_agents.ipynb
```

## 🎯 How to Use

### **Cell 1: Imports & Setup**
- Runs automatically to import all required libraries
- Loads environment variables and initializes Gemini LLM

### **Cell 2: Environment Setup**
- Loads your Gemini API key
- Initializes the LLM connection

### **Cell 3: Data Models**
- Defines structured output models for insights and variance analysis

### **Cell 4: Agent Tools**
- **`load_and_prepare_data`**: Handles multiple date formats automatically
- **`get_data_statistics`**: Generates statistical summaries
- **`analyze_trends`**: Analyzes spending patterns over time
- **`create_visualization`**: Creates interactive charts
- **`perform_variance_analysis`**: NEW! Comprehensive variance analysis tool

### **Cell 5: Agent Orchestrator**
- Creates the multi-agent system
- Manages tool coordination and memory

### **Cell 6: System Initialization**
- Initializes the orchestrator
- Ready for analysis

### **Cell 7: Comprehensive Insights**
- Generates insights for Profile → Site → Lineup hierarchy
- Uses actual data only (2022-2024)

### **Cell 8: Variance Analysis** ⭐ **NEW!**
- **Fixed date format handling** for all three data sources
- Compares Plan vs Forecast for 2025
- Compares Historical patterns vs Plan for 2025
- Provides actionable recommendations

### **Cell 9: Visualizations**
- Creates interactive HTML charts
- Uses `create_visualization_direct()` for notebook compatibility

## 🔧 Key Features

### **Smart Date Format Handling**
- Automatically detects and handles different date formats
- Supports both `dd-mm-yyyy` and `yyyy-mm-dd` formats
- No more date parsing errors!

### **Comprehensive Variance Analysis**
- **Plan vs Forecast**: How well does the forecast align with plans?
- **Historical vs Plan**: How realistic are the plans based on historical patterns?
- **Lineup-specific insights**: Detailed analysis for each business unit
- **Actionable recommendations**: Business-focused insights and mitigation strategies

### **Agent Orchestration**
- Multi-agent collaboration for complex analysis
- Memory and conversation management
- Extensible tool system for future enhancements

## 📊 Expected Outputs

### **Comprehensive Insights**
- Profile, Site, and Lineup level analysis
- Trend identification and seasonal patterns
- Key business insights and recommendations

### **Variance Analysis**
- Detailed variance calculations for 2025
- Percentage and absolute variances
- Root cause analysis and impact assessment
- Mitigation strategies

### **Visualizations**
- Interactive HTML charts (open in browser)
- Line charts for trends
- Bar charts for monthly spending
- Hierarchical analysis support

## 🧪 Testing

### **Test Basic Functionality**
```bash
python test_agents.py
```

### **Test Visualization**
```bash
python test_visualization.py
```

### **Test Variance Analysis** ⭐ **NEW!**
```bash
python test_variance_analysis.py
```

## 🚨 Troubleshooting

### **Date Format Issues** ✅ **FIXED!**
- The system now automatically handles different date formats
- No more manual format specification needed

### **Visualization Errors** ✅ **FIXED!**
- Use `create_visualization_direct()` for direct notebook calls
- `@tool` decorated functions are for agent use only

### **Variance Analysis Failures** ✅ **FIXED!**
- New dedicated tool handles all three data sources
- Automatic date format detection
- Comprehensive error handling

## 🔮 Future Enhancements

The system is designed for extensibility:
- **Additional Agents**: Specialized analysis agents
- **More Data Sources**: Database connections, APIs
- **Advanced Analytics**: Machine learning models, anomaly detection
- **Web Interface**: Streamlit or FastAPI deployment

## 📞 Support

If you encounter issues:
1. Check the test scripts first
2. Verify your `.env` file has the correct API key
3. Ensure all dependencies are installed
4. Check file paths and data availability

---

**🎉 You're all set! The system now handles date formats automatically and provides comprehensive variance analysis.**
