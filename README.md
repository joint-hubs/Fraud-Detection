# Fraud Prediction
The aim of this project is to build a model that can accurately identify fraudulent or suspicious transactions in financial datasets, which is crucial for ensuring financial security and reducing fraudulent activities.

## Results

Go to [src](src) directory.

## Summary

The project employs a multi-faceted analytical approach, utilizing various data science and machine learning techniques:

- Data Analysis and Currency Overview: Initial analysis includes a thorough examination of transactional data, with its trends and patterns. This foundational analysis sets the stage for more complex model building.

- Tree-Based Models: We explored tree-based models such as Random Forest and Gradient Boosting for their effectiveness in detecting fraud. These models are known for their ability to handle large, complex datasets and provide interpretable results.

- Dictionary-Based Approach: A novel dictionary-based model was developed, which utilizes transactional metadata and statistical metrics to identify suspicious activities. This approach allows for a dynamic and adaptive response to evolving patterns of fraud.

- Combined Model: The project culminates in a combined model that integrates the strengths of both tree-based and dictionary-based methodologies. This ensemble approach aims to leverage the unique advantages of each model to improve overall fraud detection accuracy.

### Key Insights and Findings

- Tree-Based Models' Strengths and Limitations: While tree-based models demonstrated strong performance in certain aspects, particularly in feature importance analysis, they faced challenges in consistently identifying true positives.

- Dictionary-Based Model's Unique Contributions: The dictionary-based model showed a higher rate of true positive detections, an essential factor in fraud detection. However, it exhibited a trade-off with a slight decrease in overall accuracy compared to tree-based models.

- Optimization and Robustness: The project highlights opportunities for further optimization, such as adjusting thresholds in the dictionary-based model and exploring reinforcement learning to update model parameters adaptively.

- Ensemble Benefits: The combined model approach suggests a promising avenue, potentially offering a more balanced and robust solution by integrating various models' strengths.

### Recommendations and Future Work

- Continued Model Tuning: We recommend further fine-tuning of the model parameters, especially the thresholds used in the dictionary-based model, to enhance precision and recall balance.

- Integration of Reinforcement Learning: Incorporating Reinforcement Learning could make the models more adaptive and responsive to new patterns of fraudulent activity, thereby increasing their long-term effectiveness.

- Ensemble Model Exploration: Further exploration and development of the combined model approach are advised, potentially incorporating additional machine learning techniques to create a more comprehensive fraud detection system.

### Conclusion
The Fraud Prediction project demonstrates significant potential in using advanced data analytics and machine learning to combat financial fraud. The insights and methodologies developed here lay a strong foundation for building more sophisticated and effective fraud detection systems in the future.

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





