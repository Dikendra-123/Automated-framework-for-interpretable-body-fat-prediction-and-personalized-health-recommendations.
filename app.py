import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
import openai
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
import sklearn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Load environment variables from .env file (if you're using one)
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure that the API key is set
if not openai.api_key:
    st.error("OpenAI API key is not set. Please set the 'OPENAI_API_KEY' environment variable.")
    st.stop()

# Utility Functions
def generate_code(prompt):
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',  # Use 'gpt-4' if you have access
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that generates Python code based on user prompts. "
                        "Provide only the code, without additional explanations or comments."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=3000,
            temperature=0,
            n=1,
        )
        code = response['choices'][0]['message']['content']

        # Extract code between ```python and ```
        if '```' in code:
            code_blocks = code.split('```')
            for block in code_blocks:
                if block.strip().startswith('python'):
                    code = block.strip()[len('python'):].strip()
                    break
                elif block.strip().startswith('Python'):
                    code = block.strip()[len('Python'):].strip()
                    break
            else:
                # If no language specified, take the first code block
                code = code_blocks[1].strip() if len(code_blocks) > 1 else code.strip()
        else:
            # If no code blocks, use the whole response
            code = code.strip()

        return code
    except Exception as e:
        st.error(f"Error generating code: {e}")
        return ""

def generate_explanation(prompt):
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that provides detailed explanations. "
                        "Provide only the explanation, without additional comments."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0,
            n=1,
        )
        explanation = response['choices'][0]['message']['content'].strip()
        return explanation
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
        return ""

# [All previous imports and code remain the same up to the EDAAgent class]

class EDAAgent:
    def generate_code(self):
        prompt = """
Generate Python code that performs exploratory data analysis on a pandas DataFrame 'data', including:
- Displaying statistical summaries with data.describe() using st.write().
- Storing data.describe() in 'data_summary'.
- Creating and displaying a correlation heatmap using seaborn and st.pyplot():
    - Use plt.figure(figsize=(12, 10)) to set the figure size.
    - Use the 'coolwarm' colormap.
    - Annotate the heatmap with correlation coefficients.
    - Adjust the font sizes and label rotations for readability.
- Storing the correlation matrix in 'correlation_matrix'.
- Creating and displaying a pair plot using seaborn and st.pyplot():
    - Use a sample size up to 1000 or the full dataset if it has fewer rows.
    - Store the pair plot figure in 'pairplot_fig'.
Ensure that plots are displayed in Streamlit using st.pyplot(fig).
Provide only the code, without additional explanations or comments.
"""
        code = generate_code(prompt)
        return code
 
    
    def generate_explanation(self, data_summary, correlation_matrix):
        prompt = f"""
Provide a detailed explanation of the EDA results, highlighting key insights, patterns, and correlations.

Statistical Summary:
{data_summary}

Correlation Matrix:
{correlation_matrix}

Provide only the explanation, focusing on the data insights.
"""
        explanation = generate_explanation(prompt)
        return explanation

class ModelingAgent:
    def generate_code(self):
        prompt = """
Generate Python code that:
- Assumes 'data' is a pandas DataFrame and 'BodyFat' is the target variable.
- Splits 'data' into features 'X' and target 'y' (drop 'BodyFat' from 'X').
- Splits 'X' and 'y' into training and testing sets using train_test_split.
- Standardizes the features using StandardScaler.
- Defines regression models: MLPRegressor, SVR, RandomForestRegressor, XGBRegressor.
- Performs GridSearchCV for hyperparameter tuning on each model.
- Stores the best models in a dictionary 'best_models'.
- Uses appropriate hyperparameters for each model.
Provide only the code, without additional explanations or comments.
"""
        code = generate_code(prompt)
        return code

class EvaluationAgent:
    def generate_code(self):
        prompt = """
Generate Python code that:
- Evaluates each model in 'best_models' using mean_squared_error and r2_score.
- Computes training metrics (on X_train, y_train) and testing metrics (on X_test, y_test).
- Identifies the best model based on the lowest Test MSE.
- Uses SHAP to explain the best model's predictions.
- Stores performance metrics in a dictionary 'performance'.
- Stores the best model in 'best_model'.
- Stores the SHAP values in 'shap_values' and the sample data used in 'X_sample'.
- Displays performance metrics using st.write().
- Displays SHAP summary plot using st.pyplot().
Provide only the code, without additional explanations or comments.
"""
        code = generate_code(prompt)
        return code

    def generate_explanation(self, performance_metrics, shap_summary):
        prompt = f"""
Provide a detailed explanation of the model evaluation results, including:
- Why the selected model outperforms the others based on the performance metrics.
- Insights from the SHAP analysis, explaining how features contribute to the predictions.

Performance Metrics:
{performance_metrics}

SHAP Summary:
{shap_summary}

Provide only the explanation, focusing on data-driven insights.
"""
        explanation = generate_explanation(prompt)
        return explanation

class RecommendationAgent:
    def generate_recommendations(self, selected_samples, predictions):
        # Combine selected samples and predictions into a DataFrame
        results_df = selected_samples.copy()
        results_df['Predicted BodyFat'] = predictions.values

        # Create a prompt for OpenAI
        prompt = f"""
Provide personalized health recommendations for each individual based on their data and predicted body fat percentage.

Data:
{results_df.to_string(index=False)}

For each individual, include advice on:
- Diet and eating habits
- Exercise routines (include YouTube links to relevant exercise videos)
- Lifestyle changes

Provide the recommendations in a clear and concise manner, addressing each individual separately.
"""

        recommendations = generate_explanation(prompt)
        return recommendations

# AgentController Class
class AgentController:
    def __init__(self):
        self.eda_agent = EDAAgent()
        self.modeling_agent = ModelingAgent()
        self.evaluation_agent = EvaluationAgent()
        self.recommendation_agent = RecommendationAgent()

    def run(self):
        st.sidebar.title("Navigation")
        steps = [
            "Upload Data",
            "Perform EDA",
            "Train Models",
            "Evaluate Models",
            "Perform Inference and Get Recommendations"
        ]
        choice = st.sidebar.radio("Go to", steps)

        if choice == "Upload Data":
            self.upload_data()
        elif choice == "Perform EDA":
            self.perform_eda()
        elif choice == "Train Models":
            self.train_models()
        elif choice == "Evaluate Models":
            self.evaluate_models()
        elif choice == "Perform Inference and Get Recommendations":
            self.perform_inference()

    def upload_data(self):
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("First five rows:")
                st.write(data.head())
                st.write("Last five rows:")
                st.write(data.tail())
                st.session_state['data'] = data  # Store data in session_state
                st.success("Data loaded successfully.")
            except Exception as e:
                st.error(f"Error reading the CSV file: {e}")
        else:
            st.info("Please upload a CSV file to proceed.")

    def perform_eda(self):
        st.header("Perform Exploratory Data Analysis")
        data = st.session_state.get('data')
        if data is not None:
            eda_code = self.eda_agent.generate_code()
            exec_locals = {'data': data, 'st': st, 'plt': plt, 'sns': sns, 'pd': pd, 'np': np}
            try:
                with st.spinner("Performing EDA..."):
                    exec(eda_code, {}, exec_locals)
                # Retrieve data summaries for explanation
                data_summary = exec_locals.get('data_summary')
                correlation_matrix = exec_locals.get('correlation_matrix')
                correlation_fig = exec_locals.get('correlation_fig')
                pairplot_fig = exec_locals.get('pairplot_fig')
                if data_summary is not None and correlation_matrix is not None:
                    # Organize EDA outputs
                    st.subheader("Statistical Summary")
                    st.write(data_summary)
                    with st.expander("Correlation Heatmap"):
                        st.pyplot(correlation_fig)
                    with st.expander("Pair Plot"):
                        st.pyplot(pairplot_fig)

                    # Generate and display explanation
                    eda_explanation = self.eda_agent.generate_explanation(data_summary.to_string(), correlation_matrix.to_string())
                    st.write(eda_explanation)
                else:
                    st.error("Failed to retrieve data summaries for explanation.")
            except Exception as e:
                st.error(f"Error executing generated code: {e}")
                st.code(eda_code)
        else:
            st.error("Please upload data first.")

    def train_models(self):
        st.header("Train Models")
        data = st.session_state.get('data')
        if data is not None:
            modeling_code = self.modeling_agent.generate_code()
            exec_locals = {
                'data': data,
                'st': st,
                'np': np,
                'pd': pd,
                'MLPRegressor': MLPRegressor,
                'SVR': SVR,
                'RandomForestRegressor': RandomForestRegressor,
                'xgb': xgb,
                'train_test_split': train_test_split,
                'GridSearchCV': GridSearchCV,
                'StandardScaler': StandardScaler,
                'sklearn': sklearn
            }
            try:
                with st.spinner("Training models... This may take a few minutes."):
                    exec(modeling_code, {}, exec_locals)
                best_models = exec_locals.get('best_models')
                if best_models:
                    # Store variables in session_state
                    st.session_state.update(exec_locals)
                    st.success("Model training completed.")
                    # Display trained models and their parameters
                    st.subheader("Trained Models and their Best Parameters:")
                    for model_name, model in best_models.items():
                        with st.expander(f"{model_name}"):
                            st.write(model.get_params())
                else:
                    st.error("Model training failed.")
            except Exception as e:
                st.error(f"Error executing generated code: {e}")
                st.code(modeling_code)
        else:
            st.error("Please upload data first.")

    def evaluate_models(self):
        st.header("Evaluate Models")
        best_models = st.session_state.get('best_models')
        X_train = st.session_state.get('X_train')
        y_train = st.session_state.get('y_train')
        X_test = st.session_state.get('X_test')
        y_test = st.session_state.get('y_test')
        if best_models and X_test is not None and y_test is not None:
            performance = {}
            best_model = None
            best_mse = float('inf')
            best_model_name = None
            for model_name, model in best_models.items():
                y_train_pred = model.predict(X_train)
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)

                y_test_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                performance[model_name] = {
                    'Train MSE': train_mse,
                    'Train R2': train_r2,
                    'Test MSE': test_mse,
                    'Test R2': test_r2
                }

                if test_mse < best_mse:
                    best_model = model
                    best_mse = test_mse
                    best_model_name = model_name

            st.subheader("Performance Metrics for Each Model:")
            performance_df = pd.DataFrame(performance).T
            st.write(performance_df)

            st.success(f"Model evaluation completed. Best model: {best_model_name}")

            # SHAP Analysis
            st.write("Performing SHAP analysis on the best model...")
            sample_size = min(100, X_train.shape[0])
            X_sample = X_train.iloc[:sample_size]
            try:
                if isinstance(best_model, (RandomForestRegressor, xgb.XGBRegressor)):
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_sample)
                    st.write("SHAP Summary Plot:")
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot()
                elif isinstance(best_model, (SVR, MLPRegressor)):
                    st.write("Computing SHAP values using KernelExplainer (this may take some time)...")
                    def model_predict(X):
                        return best_model.predict(X)
                    explainer = shap.KernelExplainer(model_predict, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                    st.write("SHAP Summary Plot:")
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot()
                else:
                    st.write("SHAP analysis is not available for this model.")
            except Exception as e:
                st.write(f"SHAP analysis failed: {e}")

            # Store best model and performance
            st.session_state['best_model'] = best_model
            st.session_state['performance'] = performance
            st.session_state['best_model_name'] = best_model_name

            # Generate and display explanation
            evaluation_explanation = self.evaluation_agent.generate_explanation(
                performance_df.to_string(), "SHAP analysis performed."
            )
            st.write(evaluation_explanation)
        else:
            st.error("Please train models first.")

    def perform_inference(self):
        st.header("Perform Inference and Get Recommendations")
        best_model = st.session_state.get('best_model')
        X_test = st.session_state.get('X_test')
        y_test = st.session_state.get('y_test')
        if best_model is not None and X_test is not None:
            st.write("First 15 test samples from the input CSV data:")
            test_samples = X_test.iloc[:15].reset_index(drop=True)
            st.write(test_samples)

            st.write("Select samples to perform prediction:")
            sample_indices = st.multiselect(
                "Select sample indices:",
                options=test_samples.index.tolist(),
                default=[0]
            )
            if sample_indices:
                selected_samples = test_samples.loc[sample_indices]
                true_values = y_test.iloc[sample_indices].reset_index(drop=True)

                predictions = best_model.predict(selected_samples)
                predictions_series = pd.Series(predictions, name='Predicted BodyFat')

                results_df = selected_samples.copy()
                results_df['Actual BodyFat'] = true_values
                results_df['Predicted BodyFat'] = predictions_series.values

                st.subheader("Inference Results:")
                st.write(results_df)

                st.session_state['predictions'] = predictions_series
                st.session_state['inference_results'] = results_df
                st.session_state['selected_samples'] = selected_samples.reset_index(drop=True)
                st.session_state['true_values'] = true_values

                # Generate recommendations using OpenAI
                with st.spinner("Generating personalized recommendations..."):
                    recommendations = self.recommendation_agent.generate_recommendations(selected_samples, predictions_series)
                if recommendations:
                    st.success("Recommendations:")
                    st.write(recommendations)
                else:
                    st.error("Failed to generate recommendations.")

                st.success("Inference and recommendation completed.")
            else:
                st.info("Please select at least one sample to perform prediction.")
        else:
            st.error("Please train and evaluate models first.")

def main():
    st.set_page_config(
        page_title="Autogen Body Fat Prediction",
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Autogen (LLM) Based Body Fat Prediction & Fitness Recommendation")
    controller = AgentController()
    controller.run()

if __name__ == "__main__":
    main()
