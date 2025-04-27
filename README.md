# AI-Powered Smart Cost Estimation for Construction Planning

## Project Overview
This project builds an AI + Machine Learning-driven web application that assists construction managers in accurately estimating project costs and planning more effectively. It combines a Random Forest Regressor model with a GPT-4 Large Language Model (LLM) to deliver real-time cost predictions, conversational insights, and dynamic scenario simulations through an easy-to-use Streamlit interface.

## Features
- **Cost Estimation**: Predict total construction costs based on project parameters.
- **Conversational AI**: GPT-4 integration for expert-level guidance and real-time project consultations.
- **What-If Simulations**: Analyze cost impacts by dynamically adjusting project parameters.
- **EDA Dashboard**: Explore project data patterns like location-based cost variations.

## System Architecture
- **Frontend**: Streamlit Web Application
- **Machine Learning Model**: Random Forest Regressor for cost prediction
- **AI Model**: GPT-4 via LangChain for natural language interaction
- **Dataset**: Historical construction project data
- **Integration**: Real-time user inputs combined with machine learning and AI for smart estimations

## Methodology
- **Data Preprocessing**: Handling missing values, outlier removal, and label encoding.
- **Model Training**: Training a Random Forest Regressor with key project attributes.
- **LLM Integration**: Using LangChain to connect GPT-4 for contextual and conversational responses.
- **Web App Development**: Building interactive forms, simulators, and dashboards using Streamlit.

## Installation and Setup

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```
### 2. Install the required packages
```bash
pip install -r requirements.txt
```
### 3. Set up OpenAI API Key
- Create a .env file in the project root directory.
- Add your OpenAI API Key inside the file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
### 4. Run the Streamlit app
```bash
streamlit run app.py
```
