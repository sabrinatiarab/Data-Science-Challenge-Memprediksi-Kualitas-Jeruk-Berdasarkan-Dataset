import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('gradient_boost_model.pkl')  # Ensure the filename matches
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None

model = load_model()

# Preprocessing function
def preprocess_data(input_df):
    try:
        # Columns to include from Min-Max scaling
        include_columns = ['Weight (g)']
        scale_columns = [col for col in input_df.columns if col in include_columns]

        # Apply Min-Max scaling to selected columns
        scaler = MinMaxScaler()
        input_df[scale_columns] = scaler.fit_transform(input_df[scale_columns])
        
        return input_df

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Updated encoding function for categorical variables
def encode_categorical_data(input_df):
    color_mapping = {
        'Deep Orange': 0,
        'Light Orange': 1,
        'Orange-Red': 2,
        'Orange': 3,
        'Yellow-Orange': 4
    }
    
    variety_mapping = {
        'Valencia': 22,
        'Navel': 13,
        'Cara Cara': 3,
        'Blood Orange': 1,
        'Hamlin': 6,
        'Tangelo (Hybrid)': 19,
        'Murcott (Hybrid)': 12,
        'Moro (Blood)': 11,
        'Jaffa': 8,
        'Clementine': 4,
        'Washington Navel': 23,
        'Star Ruby': 18,
        'Tangerine': 20,
        'Ambiance': 0,
        'California Valencia': 2,
        'Honey Tangerine': 7,
        'Navel (Late Season)': 15,
        'Clementine (Seedless)': 5,
        'Temple': 21,
        'Minneola (Hybrid)': 10,
        'Satsuma Mandarin': 17,
        'Midsweet (Hybrid)': 9,
        'Navel (Early Season)': 14,
        'Ortanique (Hybrid)': 16
    }
    
    blemishes_mapping = {
        'N': 0,
        'Y (Minor)': 1,
        'Y (Sunburn)': 2,
        'Y (Mold Spot)': 3,
        'Y (Bruise)': 4,
        'Y (Split Skin)': 5,
        'Y (Sunburn Patch)': 6,
        'Y (Scars)': 7,
        'Y (Minor Insect Damage)': 8,
        'Y (Bruising)': 9,
        'N (Minor)': 10,
        'N (Split Skin)': 11
    }

    input_df['Color'] = input_df['Color'].map(color_mapping)
    input_df['Variety'] = input_df['Variety'].map(variety_mapping)
    input_df['Blemishes (Y/N)'] = input_df['Blemishes (Y/N)'].map(blemishes_mapping)
    
    return input_df

# Mapping of encoded quality values to bins
def bin_quality(encoded_quality):
    # Define the bin edges
    bins = [0, 1, 2, 3, 4, 5, 6, 7]
    # Define the labels
    labels = [2, 2.5, 3, 3.5, 4, 4.5, 5]  # Updated to match the unique quality values
    
    # Use pd.cut to bin the encoded quality
    binned_quality = pd.cut([encoded_quality], bins=bins, labels=labels, include_lowest=True)[0]
    
    return binned_quality

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
        'Y (Split Skin)', 'Y (Sunburn Patch)', 'Y (Scars)', 
        'Y (Minor Insect Damage)', 'Y (Bruising)', 'N (Minor)', 
        'N (Split Skin)'
    ])
    
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    # Collect input data
    input_data = {
        'Size (cm)': [size],
        'Weight (g)': [weight],
        'Brix (Sweetness)': [brix],
        'pH (Acidity)': [ph],
        'Softness (1-5)': [softness],
        'Harvest Time (days)': [harvest_time],  # Ensure column names match
        'Ripeness (1-5)': [ripeness],
        'Color': [color],
        'Variety': [variety],
        'Blemishes (Y/N)': [blemishes]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Preprocess and encode the data
    preprocessed_data = preprocess_data(input_df)
    if preprocessed_data is not None:
        encoded_data = encode_categorical_data(preprocessed_data)
        if encoded_data is not None:
            try:
                # Predict
                prediction = model.predict(encoded_data)
                predicted_quality = bin_quality(prediction[0])
                st.write(f"The predicted orange quality is: {predicted_quality}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")