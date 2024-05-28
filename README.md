### Obesity Level Prediction Project
- Overview
This project aims to predict obesity levels based on various features such as gender, age, height, weight, and lifestyle factors.

Several machine learning models were trained and evaluated to achieve accurate predictions.

### Table of Contents
Project Description
Data Visualization Insights
Density Plot Observations
Age Distribution Plot Observations
Box Plot Observations
Model Training Results
Test Set Results
Conclusion
Deployment
### Project Description
The project workflow includes the following steps:

1. Loading the data: Importing the dataset ObesityDataSet_raw_and_data_sinthetic.csv.
2. Feature Engineering: Processing and transforming raw data into meaningful features for model training.
3. Training the data: Applying various machine learning and deep learning algorithms.
4. Evaluating the models' performance: Using cross-validation and other metrics to assess the models.
5. Saving the model: Persisting the best models for future use.
6. Selection of the best models for deployment: Choosing the top-performing models for real-world application.

## Test Set Results
| Model                       | Precision | Recall | F1-Score | Support | Accuracy |
|-----------------------------|-----------|--------|----------|---------|----------|
| GradientBoostingClassifier  | 0.967     | 0.966  | 0.966    | 739     | 0.966    |
| SVM                         | 0.942     | 0.940  | 0.940    | 739     | 0.940    |
| RandomForestClassifier      | 0.930     | 0.927  | 0.928    | 739     | 0.927    |
| DecisionTreeClassifier      | 0.913     | 0.912  | 0.912    | 739     | 0.912    |
| KNN                         | 0.886     | 0.878  | 0.873    | 739     | 0.878    |
| MLPClassifier               | 0.855     | 0.854  | 0.852    | 739     | 0.854    |

Stacked Classifier Results
Accuracy: 0.951
Precision: 0.951
Recall: 0.951
F1-score: 0.951
Conclusion
Based on the trained models and their performance metrics, the GradientBoostingClassifier achieved the highest accuracy of 96%. This model can be utilized for predicting obesity levels based on the provided features.

Deployment
The model has been deployed using Streamlit. The app can be accessed here. Make sure to install the necessary dependencies from requirements.txt and follow the instructions below to run the app locally.

Running the App Locally
### Clone the Repository

```sh
git clone https://github.com/Wayneotc/Obesity-Levels.git
cd Obesity-Levels
sh
Copy code
python -m venv myenv
Activate the virtual environment:

Windows:
sh
Copy code
myenv\Scripts\activate
macOS/Linux:
sh
Copy code
source myenv/bin/activate
Install the dependencies:

sh
Copy code
pip install -r requirements.txt
Run the Streamlit app:

sh
Copy code
streamlit run app.py
