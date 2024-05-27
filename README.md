### Obesity Level Prediction Project

Overview

This project aims to predict obesity levels based on various features such as gender, age, height, weight, and lifestyle factors. Several machine learning models were trained and evaluated to achieve accurate predictions

- Loading the data(ObesityDataSet_raw_and_data_sinthetic).

- Feature Engineering.

- Training the data with machine and deep learning algorithms.

- Evaluating the models' performance on the training and testing data.

- Saving the model.

- Selection of the best models for deployment.

### Data Visualization Insights
Density Plot Observations

Weight, Age, and Height:

Obesity type III is prevalent for people with a weight of 100 and above.
Individuals between ages 20 and 30 are mostly affected by Obesity type III.
Obesity type II is prevalent for people with a height of 1.8 meters.
Age Distribution Plot Observations
Age Distribution:

The majority of individuals in the dataset are between the ages of 20 and 30.
Older age groups (40 and above) are underrepresented in the dataset.
Box Plot Observations
Age and Gender:
Younger individuals are more represented in the Normal Weight and Insufficient Weight categories.
Older individuals tend to be more represented in higher obesity levels, especially Obesity Types I, II, and III.
Noticeable differences in age distribution between genders across obesity levels, with males generally showing higher median ages in more severe obesity categories.
Model Training Results
Cross-Validation Accuracy Scores

Average Cross-Validation Accuracy: 0.819
Best parameters from Grid Search: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Random Forest:

Average Cross-Validation Accuracy: 0.931
Best parameters from Grid Search: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
Support Vector Classifier (SVC):

Average Cross-Validation Accuracy: 0.781
Best parameters from Grid Search: {'svc__C': 10.0, 'svc__class_weight': None, 'svc__degree': 2, 'svc__gamma': 'scale', 'svc__kernel': 'linear'}
KNeighborsClassifier:

Average Cross-Validation Accuracy: 0.953
Best parameters from Grid Search: {'knn__p': 1, 'knn__weights': 'distance'}
GradientBoostingClassifier:

Average Cross-Validation Accuracy: 0.883
Best parameters from Grid Search: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300}
MLPClassifier:

Average Cross-Validation Accuracy: N/A
Best parameters from Grid Search: {'mlp__activation': 'tanh', 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant'}
## Test Set Results

| Model                      | Precision | Recall  | F1-Score | Support | Accuracy |
|----------------------------|-----------|---------|----------|---------|----------|
| GradientBoostingClassifier | 0.967     | 0.966   | 0.966    | 739     | 0.966    |
| SVM                        | 0.942     | 0.940   | 0.940    | 739     | 0.940    |
| RandomForestClassifier     | 0.930     | 0.927   | 0.928    | 739     | 0.927    |
| DecisionTreeClassifier     | 0.913     | 0.912   | 0.912    | 739     | 0.912    |
| KNN                        | 0.886     | 0.878   | 0.873    | 739     | 0.878    |
| MLPClassifier              | 0.855     | 0.854   | 0.852    | 739     | 0.854    |

### Stacked Classifier Results
- **Accuracy**: 0.951
- **Precision**: 0.951
- **Recall**: 0.951
- **F1-score**: 0.951
- 
###Conclusion
Based on the trained models and their performance metrics, the Gradient boosting model achieved the highest accuracy of 96%. This model can be utilized for predicting obesity levels based on the provided features.
