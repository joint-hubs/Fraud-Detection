# Fraud Prediction R&D
The aim of this Proof of Concept (PoC) project is to build a model capable of accurately identifying fraudulent or suspicious transactions within financial datasets. This project serves as research proof of work, with models documented in corresponding Python notebooks.

## Results

The provided dataset was a valuable starting point but insufficient for training a production-ready model. It offered insights into the data structure and facilitated the development of workflows running multiple models: Random Forest, XGBoost, and a custom Dictionary-Based Model, as well as an ensemble approach combining Dictionary-Based and XGBoost models.

For industrialization, processing a significant volume of training data is crucial. The established workflows are designed for scalability and adaptability to new data structures. For practical application, models should be serialized and deployed as private API endpoints to integrate predictions into existing databases.

Explore the model scripts and workflows in the [src](src) directory containing Python Notebooks.

## Summary

The project adopts a comprehensive analytical approach, leveraging various Data Science and Machine Learning techniques:

- Exploratory Data Analysis (EDA): An in-depth examination of transactional data to uncover trends and patterns, laying the groundwork for advanced model development.

- Tree-Based Models: An exploration of Random Forest and Gradient Boosting models for their efficacy in fraud detection.

- Dictionary-Based Approach: A custom Dictionary-Based Model using transactional metadata and statistical metrics for dynamic fraud detection.

- Combined Model: A sophisticated ensemble model that merges the strengths of tree-based and dictionary-based methods to enhance overall detection accuracy.

### Key Insights and Findings

- Tree-Based Models: Strong in feature importance analysis but less consistent in True Positive identification. Expanding the training dataset could improve accuracy.

- Dictionary-Based Model: Low accuracy, it could be used as feature engineering algorithm to enhance the data. 

- Optimization and Robustness: The project opens avenues for further optimization, including threshold adjustments and the potential adoption of adaptive, Reinforcement Learning techniques.

- Ensemble Benefits: The combined model strategy indicates a promising path, potentially offering a more holistic solution by harnessing the strengths of different models.

### Recommendations and Future Work

- Training with Big Data: Utilize larger datasets to refine model parameters, enhancing precision and recall balance.

- API Deployment: Industrialize models as private API endpoints for seamless integration with transaction databases.

- Reinforcement Learning Integration: To improve adaptability and responsiveness to new fraud patterns, thereby bolstering long-term model effectiveness.

- Ensemble Model Exploration: Further development of the combined model strategy is recommended, including the integration of diverse Machine Learning techniques for a more comprehensive fraud detection system.

- Hierarchical Modeling: Investigate training specialized ML models for distinct customer clusters (identified via Unsupervised Learning), potentially increasing model specificity and effectiveness.

### Conclusion

The Fraud Prediction PoC project marks a significant step towards advanced, data-driven solutions in fraud detection.

## Machine Setup

This project requires `Python 3.10.5` or later. You can download Python from the official website: https://www.python.org/downloads/

### Installing Dependencies

All required dependencies for this project are listed in the `requirements.txt` file. To install these dependencies, run the following command in your terminal:

> pip install -r requirements.txt

### Virtual Environment

It is recommended to use a virtual environment to manage dependencies for your project. By using a virtual environment, you can ensure that the dependencies for this project do not interfere with other Python projects on your system. To create and activate a virtual environment in Python, follow these steps:

1. Open a terminal and navigate to the root directory of the project

2. Run the following command to create a virtual environment named env:

    for Unix/macOS
    > python3 -m venv env

    or for Windows
    > py -m venv env


3. Activate the virtual environment with the following command:

    for Unix/macOS
    > source env/bin/activate

    or for Windows
    > .\env\Scripts\activate

    or
    > env\Scripts\activate.bat

4. Install the dependencies

    > pip install -r requirements.txt

5. Run Jupyter notebook with command:
    > jupyter notebook

6. Inside Jupyter go to `src` directory and use the notebooks.

<br>





