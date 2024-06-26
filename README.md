# Obesity Level Prediction Project
### Overview
This project aims to predict obesity levels based on various features such as gender, age, height, weight, and lifestyle factors.

Several machine learning models were trained and evaluated to achieve accurate predictions.

# Project Description
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

- Stacked Classifier Results : F1-score: 0.951
# Conclusion
Based on the trained models and their performance metrics, the GradientBoostingClassifier achieved the highest accuracy of 96%. This model can be utilized for predicting obesity levels based on the provided features.

# Deployment
The model has been deployed using Streamlit.

You can access the deployed Streamlit application [here](https://obesity-levelsgit-3mnn7sqzxuxrjjh8dtd5zo.streamlit.app/).

# Usage
1. Open the app in your web browser.
2. Enter the required information in the input fields.
3. Click the 'Predict' button to generate the prediction.

# Inputs
- Gender
- Do you frequently consume high caloric food?
- Do you smoke?
- Do you monitor your calorie consumption?
- How often do you drink alcohol?
- CAEC
- MTRANS
- Age
- Height (in cm)
- Weight (in kg)
- Frequency of consumption of vegetables
- Number of main meals
- Daily water consumption (CH2O)
- Time using technology devices (hours per day)
- Physical activity frequency (times per week)

# Outputs
The app will display following messages:
- Overweight_Level_I
- Overweight_Level_II
- Insufficient_weight
- Obesity_Type_III
- Obesity_Type_II
- Obesity_Type_I
- Normal_weight

# Contact
kiplagatwayne@gmail.com
# Installation
Make sure to install the necessary dependencies from requirements.txt and follow the instructions below to run the app locally.
```sh
git clone https://github.com/Wayneotc/Obesity-Levels.git

cd Obesity-Levels

python -m venv myenv

myenv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
