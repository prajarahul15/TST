# Financial Data Insight Generation System

## 📋 Project Overview

This project aims to create an intelligent insight generation system for financial data analysis, specifically designed to analyze AWS cloud spending data with hierarchical structure (Profile → Site → Lineup). The system will generate comprehensive insights using LLM (Gemini) and agentic AI approaches.

## 🎯 Core Requirements

### 1. Data Structure Analysis
- **Actual Data**: Complete 3-year period (2022-2024) for all hierarchies
- **Plan Data**: 12-month forecast period (2025) only
- **Forecast Data**: 12-month forecast period (2025) only
- **Hierarchy**: Profile → Site → Lineup

### 2. Insight Generation Requirements

#### A. Comprehensive Analysis (Actual Data Only)
- Profile-level analysis
- Site-level analysis  
- Lineup-level analysis
- Statistical insights and trends
- Seasonal pattern identification
- Growth rate analysis

#### B. Variance Analysis (Actual vs Plan vs Forecast)
- Actual vs Plan comparisons for forecast period
- Actual vs Forecast comparisons for forecast period
- Statistical significance testing
- Variance explanations and insights

### 3. Technical Requirements
- **LLM Integration**: Google Gemini model
- **Agent-Based Solution**: Autonomous insight generation
- **Agent Orchestration**: Multi-agent collaboration
- **Visualization**: Automated plot generation
- **Scalability**: Future module integration capability
- **Environment**: .env file for API key management

## 🏗️ Framework Analysis & Recommendations

### Initial Recommendation: LangChain + Streamlit
- **Pros**: Quick development, good LLM integration
- **Cons**: Limited scalability, basic agent orchestration

### Revised Recommendation: FastAPI + LangChain + Agent Orchestration
- **Pros**: High performance, async support, enterprise-grade
- **Cons**: Higher complexity, steeper learning curve

### Alternative Frameworks Analyzed

#### 1. CrewAI - Multi-Agent Collaboration
```python
# Example Structure
from crewai import Agent, Task, Crew

class InsightCrew:
    def __init__(self):
        self.data_analyst = Agent(
            role="Data Analyst",
            goal="Process and analyze financial data",
            backstory="Expert in financial data analysis"
        )
        
        self.llm_specialist = Agent(
            role="LLM Specialist", 
            goal="Generate insights using Gemini",
            backstory="AI expert in natural language generation"
        )
        
        self.visualization_expert = Agent(
            role="Visualization Expert",
            goal="Create compelling charts and plots",
            backstory="Data visualization specialist"
        )
```

**✅ Pros:**
- Natural Agent Roles
- Built-in Collaboration
- Task Chaining
- Human-in-the-Loop
- Memory Management

**❌ Cons:**
- Learning Curve
- Limited Async
- Complex Deployment

#### 2. AutoGen (Microsoft) - Conversational AI
```python
# Example Structure
import autogen

class InsightAutoGen:
    def __init__(self):
        self.data_analyst = autogen.AssistantAgent(
            name="data_analyst",
            system_message="Expert in financial data analysis",
            llm_config={"config_list": self.config_list}
        )
        
        self.llm_specialist = autogen.AssistantAgent(
            name="llm_specialist", 
            system_message="AI expert in generating insights",
            llm_config={"config_list": self.config_list}
        )
```

**✅ Pros:**
- Microsoft Backing
- Conversational Flow
- Multi-Modal Support
- Human Integration
- Tool Integration

**❌ Cons:**
- Complexity
- Performance
- Resource Usage

#### 3. LlamaIndex - Data-Centric AI
```python
# Example Structure
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.agent import ReActAgent

class InsightLlamaIndex:
    def __init__(self):
        self.documents = SimpleDirectoryReader('data/').load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        
        self.agent = ReActAgent.from_tools(
            tools=[data_tool, analysis_tool, plot_tool],
            llm=gemini_llm,
            verbose=True
        )
```

**✅ Pros:**
- Data-First Approach
- Vector Search
- Document Processing
- Tool Integration
- Performance

**❌ Cons:**
- Agent Complexity
- Learning Curve
- Overhead

#### 4. AGNO - Lightweight Agent Framework
```python
# Example Structure
from agno import Agent, Workflow

class InsightAGNO:
    def __init__(self):
        self.data_agent = Agent(
            name="data_processor",
            function=self.process_data
        )
        
        self.analysis_agent = Agent(
            name="insight_generator", 
            function=self.generate_insights
        )
        
        self.workflow = Workflow([
            self.data_agent,
            self.analysis_agent
        ])
```

**✅ Pros:**
- Lightweight
- Fast
- Simple
- Flexible

**❌ Cons:**
- Limited Features
- Small Community
- Less Mature

## 🎯 Final Recommendation: CrewAI + FastAPI Hybrid

### Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI LAYER (Orchestration)                │
│  • API Endpoints                                               │
│  • Request/Response Handling                                   │
│  • Authentication & Rate Limiting                              │
│  • Background Task Management                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CREWAI LAYER (Agent Management)              │
│  • Data Analysis Agent                                         │
│  • LLM Insight Agent                                           │
│  • Visualization Agent                                          │
│  • Variance Analysis Agent                                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA & LLM LAYER                             │
│  • Gemini API Integration                                      │
│  • CSV Processing                                              │
│  • Statistical Analysis                                        │
│  • Plot Generation                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Hybrid Approach?

#### 1. FastAPI Benefits
- **Async Performance**: Handle multiple concurrent analysis requests
- **Scalability**: Easy horizontal scaling and load balancing
- **Production Ready**: Enterprise-grade deployment and monitoring
- **API Management**: Professional API documentation and versioning

#### 2. CrewAI Benefits
- **Agent Collaboration**: Natural multi-agent workflows
- **Role-Based Design**: Clear agent responsibilities and goals
- **Task Orchestration**: Built-in task chaining and dependencies
- **Human Oversight**: Easy to add human-in-the-loop validation

#### 3. Combined Advantages
- **Best of Both**: Performance + Agent Intelligence
- **Future-Proof**: Easy to add new agent types and workflows
- **Scalable**: Can handle complex multi-step analyses
- **Maintainable**: Clear separation of concerns

## 🚀 Implementation Strategy

### Phase 1: CrewAI Agent Setup
```python
class InsightCrew:
    def __init__(self):
        self.data_agent = Agent(
            role="Financial Data Analyst",
            goal="Process CSV data and perform statistical analysis",
            backstory="Expert in financial data with 10+ years experience"
        )
        
        self.llm_agent = Agent(
            role="AI Insight Specialist",
            goal="Generate natural language insights using Gemini",
            backstory="AI expert specializing in financial insights"
        )
        
        self.plot_agent = Agent(
            role="Data Visualization Expert",
            goal="Create compelling charts and interactive plots",
            backstory="Expert in financial data visualization"
        )
```

### Phase 2: FastAPI Integration
```python
@app.post("/generate-insights")
async def generate_insights(request: InsightRequest):
    crew = InsightCrew()
    result = await crew.run_analysis(request)
    return result

@app.post("/generate-insights-async")
async def generate_insights_async(request: InsightRequest):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(crew.run_analysis, task_id, request)
    return {"task_id": task_id, "status": "queued"}
```

### Phase 3: Advanced Orchestration
```python
class AdvancedInsightCrew:
    def __init__(self):
        self.agents = self._create_agent_hierarchy()
        self.workflow_manager = WorkflowManager()
    
    async def run_complex_analysis(self, request):
        # Multi-step analysis with agent collaboration
        # Parallel processing where possible
        # Error handling and recovery
```

## 📊 Data Analysis Requirements

### Sample Data Structure
- **File**: `Sample_data_N.csv`
- **Columns**: Profile, Line_Item, Budget Unit, Token, Body, Site, Lineup, Institutions, DATE, Actual
- **Time Period**: 2022-2024 (36 months per Lineup)
- **Total Records**: 72 (36 × 2 Lineups)

### Plan Data Structure
- **File**: `Plan Number.csv`
- **Time Period**: 2025 (12 months)
- **Total Records**: 24 (12 × 2 Lineups)

### Forecast Data Structure
- **Generated by**: `fixed_forecasting.py`
- **Time Period**: 2025 (12 months)
- **Total Records**: 24 (12 × 2 Lineups)

## 🔧 Technical Stack

### Backend
- **Python**: Core programming language
- **FastAPI**: High-performance web framework
- **CrewAI**: Multi-agent orchestration
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### LLM Integration
- **Google Gemini**: Primary LLM for insight generation
- **LangChain**: LLM framework integration
- **Environment**: .env file for API key management

### Data Processing
- **CSV Processing**: Pandas for data loading and manipulation
- **Statistical Analysis**: SciPy for advanced statistical functions
- **Data Validation**: Pydantic for data models and validation

### Visualization
- **Plotly**: Interactive charts and dashboards
- **Matplotlib**: Static chart generation
- **Seaborn**: Statistical visualization
- **Export Formats**: PNG, PDF, HTML

## 📈 Expected Outputs

### 1. Comprehensive Analysis Report
- Profile-level spending trends
- Site-level performance metrics
- Lineup-level detailed analysis
- Statistical summaries and insights
- Growth rate calculations
- Seasonal pattern identification

### 2. Variance Analysis Report
- Actual vs Plan comparisons
- Actual vs Forecast comparisons
- Statistical significance testing
- Variance explanations
- Recommendations and insights

### 3. Visualizations
- Time series trend charts
- Hierarchical spending treemaps
- Variance comparison bar charts
- Seasonal pattern heatmaps
- Distribution histograms
- Interactive dashboards

## 🚀 Future Scalability

### Module Integration
- **Plugin Architecture**: Easy addition of new analysis modules
- **Agent Discovery**: Dynamic agent registration and management
- **Workflow Templates**: Reusable analysis patterns
- **API Extensions**: RESTful endpoints for new functionalities

### Performance Optimization
- **Caching**: Redis integration for result caching
- **Async Processing**: Background task execution
- **Load Balancing**: Multiple instance deployment
- **Monitoring**: Performance metrics and health checks

## 📝 Next Steps

1. **Framework Selection**: Choose between recommended approaches
2. **Architecture Design**: Detailed system design and API specifications
3. **Implementation**: Phase-wise development and testing
4. **Integration**: LLM and agent system integration
5. **Testing**: Comprehensive testing and validation
6. **Deployment**: Production deployment and monitoring

---

## 🔗 Related Files

- `Sample_data_N.csv`: Historical financial data
- `Plan Number.csv`: Planned spending data
- `fixed_forecasting.py`: Forecast generation script
- `sample_data_insights.py`: Basic data analysis script

## 📞 Contact

For questions or further discussions about this insight generation system, please refer to the conversation history and framework analysis provided above.
