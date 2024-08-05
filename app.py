import streamlit as st
import pandas as pd
import joblib
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('extra_trees_model.pkl')
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None

model = load_model()

# Preprocessing function
def preprocess_data(input_df):
    # Apply Box-Cox transformation to 'Ripeness (1-5)'
    if 'Ripeness (1-5)' in input_df.columns:
        input_df['Ripeness (1-5)'] = input_df['Ripeness (1-5)'] + 1e-6
        input_df['Ripeness (1-5)'] = boxcox(input_df['Ripeness (1-5)'], lmbda=1.534807950678563)

    # Columns to exclude from Min-Max scaling
    exclude_columns = ['Color', 'Variety', 'Blemishes (Y/N)', 'Quality (1-5)']
    scale_columns = [col for col in input_df.columns if col not in exclude_columns]

    # Apply Min-Max scaling to selected columns
    scaler = MinMaxScaler()
    input_df[scale_columns] = scaler.fit_transform(input_df[scale_columns])
    
    return input_df

# Encoding function for categorical variables
def encode_categorical_data(input_df):
    color_mapping = {
        'Deep Orange': 0,
        'Light Orange': 0.25,
        'Orange-Red': 0.5,
        'Orange': 0.75,
        'Yellow-Orange': 1
    }
    
    variety_mapping = {
        'Valencia': 0.95652174,
        'Navel': 0.56521739,
        'Cara Cara': 0.13043478,
        'Blood Orange': 0.04347826,
        'Hamlin': 0.26086957,
        'Tangelo (Hybrid)': 0.82608696,
        'Murcott (Hybrid)': 0.52173913,
        'Moro (Blood)': 0.47826087,
        'Jaffa': 0.34782609,
        'Clementine': 0.17391304,
        'Washington Navel': 1,
        'Star Ruby': 0.7826087,
        'Tangerine': 0.86956522,
        'Ambiance': 0,
        'California Valencia': 0.08695652,
        'Honey Tangerine': 0.30434783,
        'Navel (Late Season)': 0.65217391,
        'Clementine (Seedless)': 0.2173913,
        'Temple': 0.91304348,
        'Minneola (Hybrid)': 0.43478261,
        'Satsuma Mandarin': 0.73913043,
        'Midsweet (Hybrid)': 0.39130435,
        'Navel (Early Season)': 0.60869565,
        'Ortanique (Hybrid)': 0.69565217
    }
    
    blemishes_mapping = {
        'N': 0,
        'Y (Minor)': 0.98356539,
        'Y (Sunburn)': 1,
        'Y (Mold Spot)': 0.98783356,
        'Y (Bruise)': 0.96359596,
        'Y (Split Skin)': 0.99466169,
        'Y (Sunburn Patch)': 0.99747699,
        'Y (Scars)': 0.9914815,
        'Y (Minor Insect Damage)': 0.97843731,
        'Y (Bruising)': 0.97204112,
        'N (Minor)': 0.92917576,
        'N (Split Skin)': 0.95130148
    }

    input_df['Color'] = input_df['Color'].map(color_mapping)
    input_df['Variety'] = input_df['Variety'].map(variety_mapping)
    input_df['Blemishes (Y/N)'] = input_df['Blemishes (Y/N)'].map(blemishes_mapping)
    
    return input_df

# Streamlit interface
st.title('Orange Quality Prediction')

st.write("Please input the features for prediction:")

# Input form
with st.form(key='predict_form'):
    size = st.number_input('Size (cm)', min_value=6.0, max_value=10.0, step=0.1)
    weight = st.number_input('Weight (g)', min_value=100.0, max_value=300.0, step=1.0)
    brix = st.number_input('Brix (Sweetness)', min_value=5.5, max_value=16.0, step=0.1)
    ph = st.number_input('pH (Acidity)', min_value=2.8, max_value=4.4, step=0.1)
    softness = st.number_input('Softness (1-5)', min_value=1.0, max_value=5.0, step=0.5)
    harvest_time = st.number_input('Harvest Time (days)', min_value=6.0, max_value=25.0, step=1.0)
    ripeness = st.number_input('Ripeness (1-5)', min_value=1.0, max_value=5.0, step=0.5)
    
    color = st.selectbox('Color', ['Deep Orange', 'Light Orange', 'Orange-Red', 'Orange', 'Yellow-Orange'])
    variety = st.selectbox('Variety', [
        'Valencia', 'Navel', 'Cara Cara', 'Blood Orange', 'Hamlin', 
        'Tangelo (Hybrid)', 'Murcott (Hybrid)', 'Moro (Blood)', 'Jaffa', 
        'Clementine', 'Washington Navel', 'Star Ruby', 'Tangerine', 'Ambiance', 
        'California Valencia', 'Honey Tangerine', 'Navel (Late Season)', 
        'Clementine (Seedless)', 'Temple', 'Minneola (Hybrid)', 'Satsuma Mandarin', 
        'Midsweet (Hybrid)', 'Navel (Early Season)', 'Ortanique (Hybrid)'
    ])
    blemishes = st.selectbox('Blemishes (Y/N)', [
        'N', 'Y (Minor)', 'Y (Sunburn)', 'Y (Mold Spot)', 'Y (Bruise)', 
        'Y (Split Skin)', 'Y (Sunburn Patch)', 'Y (Scars)', 'Y (Minor Insect Damage)', 
        'Y (Bruising)', 'N (Minor)', 'N (Split Skin)'
    ])
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    try:
        # Prepare input data
        input_data = {
            'Size (cm)': [size],
            'Weight (g)': [weight],
            'Brix (Sweetness)': [brix],
            'pH (Acidity)': [ph],
            'Softness (1-5)': [softness],
            'HarvestTime (days)': [harvest_time],
            'Ripeness (1-5)': [ripeness],
            'Color': [color],
            'Variety': [variety],
            'Blemishes (Y/N)': [blemishes]
        }
        
        input_df = pd.DataFrame.from_dict(input_data)

        # Encode the categorical variables
        input_df_encoded = encode_categorical_data(input_df)

        # Preprocess the input data
        input_df_preprocessed = preprocess_data(input_df_encoded)

        # Make the prediction
        prediction = model.predict(input_df_preprocessed)
        predicted_kualitas = prediction[0]
        st.success(f'Predicted Quality (1-5): {predicted_kualitas}')
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")