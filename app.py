import streamlit as st
import pickle
import numpy as np

# Load the model
model_path = 'best_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title of the Streamlit app
st.title('Obesity Level Prediction')

# Input features
st.write("Please input the features for prediction:")

# Categorical features
gender = st.selectbox('Gender', ['Male', 'Female'])
favc = st.selectbox('Do you frequently consume high caloric food?', ['Yes', 'No'])
smoke = st.selectbox('Do you smoke?', ['Yes', 'No'])
scc = st.selectbox('Do you monitor your calorie consumption?', ['Yes', 'No'])
calc = st.selectbox('How often do you drink alcohol?', ['No', 'Sometimes', 'Frequently'])
caec = st.selectbox('CAEC', ['Always', 'Frequently', 'Sometimes', 'No'])
mtrans = st.selectbox('MTRANS', ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])

# Numeric features
age = st.number_input('Age', min_value=0)
height = st.number_input('Height (in cm)', min_value=0.0)
weight = st.number_input('Weight (in kg)', min_value=0.0)
# Frequency of consumption of vegetables options
fcvc_options = ['Sometimes', 'Always', 'Never']

# Frequency of consumption of vegetables input
fcvc_input = st.selectbox('Frequency of consumption of vegetables', fcvc_options)

# Map input to numerical value
fcvc_map = {'Sometimes': 1, 'Always': 3, 'Never': 0}
fcvc = fcvc_map[fcvc_input]

# Number of main meals options
ncp_options = ['More than three', 'Between 1 and 2', 'Three']

# Number of main meals input
ncp_input = st.selectbox('Number of main meals', ncp_options)

# Map input to numerical value
ncp_map = {'More than three': 4, 'Between 1 and 2': 2.5, 'Three': 3}
ncp = ncp_map[ncp_input]

ch2o_options = ['Between 1 and 2 L', 'More than 2 L', 'Less than a liter']
ch2o_input = st.selectbox('Daily water consumption (CH2O)', ch2o_options)
ch2o_map = {'Less than a liter': 0, 'Between 1 and 2 L': 1, 'More than 2 L': 2}
ch2o = ch2o_map[ch2o_input]
# Time using technology devices options
tue_options = ['I do not have', '0–2 hours', '3–5 hours']

# Time using technology devices input
tue_input = st.selectbox('Time using technology devices (hours per day)', tue_options)

# Map input to numerical value
tue_map = {'I do not have': 0, '0–2 hours': 1, '3–5 hours': 2}
tue = tue_map[tue_input]

# Physical activity frequency options
faf_options = ['I do not have', '1 or 2 days', '2 or 4 days', '4 or 5 days']

# Physical activity frequency input
faf_input = st.selectbox('Physical activity frequency (times per week)', faf_options)

# Map input to numerical value
faf_map = {'I do not have': 0, '1 or 2 days': 1, '2 or 4 days': 2, '4 or 5 days': 3}
faf = faf_map[faf_input]


# Encode categorical features
gender = 1 if gender == 'Male' else 0
favc = 1 if favc == 'Yes' else 0
smoke = 1 if smoke == 'Yes' else 0
scc = 1 if scc == 'Yes' else 0
calc_map = {'No': 0, 'Sometimes': 1, 'Frequently': 2}
calc = calc_map[calc]
caec_map = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'No': 3}
caec = caec_map[caec]
mtrans_map = {'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4}
mtrans = mtrans_map[mtrans]

# Combine all features into a single array
features = np.array([gender, age, height, weight, favc, fcvc, ncp, smoke, ch2o, scc, faf, tue, calc, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]).reshape(1, -1)

# Predict button
if st.button('Predict'):
    prediction = model.predict(features)
    st.write(f'The predicted obesity level is: {prediction[0]}')
