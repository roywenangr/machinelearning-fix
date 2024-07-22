# diabetes_app.py

# imports
import pandas as pd
import joblib
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

# Header
st.write("""
# Prediksi Risiko Diabetes
Jawablah 6 pertanyaan untuk mengetahui apakah Anda berisiko terkena Diabetes Tipe II.
""")

with st.expander("Klik untuk Menampilkan FAQ"):
    st.write("""
    * **Pertanyaan**:
        1. Berat
        2. Tinggi
        3. Umur
        4. Tingkat Kolesterol
        5. Tekanan Darah
        6. Kesehatan Secara Menyeluruh
        
            * Lebih dari 8 dari 10 orang dewasa yang mengidap prediabetes tidak mengetahui bahwa mereka mengidapnya.
    """)
with st.expander("Klik untuk melihat Decision Tree:"):
    st.write("""Beginilah cara prediksi risiko Diabetes dibuat oleh aplikasi ini.""")
    st.image('decisiontree.png')

st.write("### Jawablah 6 Pertanyaan berikut ini:")

# create the colums to hold user inputs
col1, col2, col3 = st.columns(3)

# function to convert lbs to kg
def lbs_to_kg(lbs):
    return lbs * 0.45359237

# function to convert cm to inch
def cm_to_inch(cm):
    return cm / 2.54

# gather user inputs

# 1. Weight
weight_unit = col1.selectbox('Pilih satuan berat', ('lbs', 'kg'), index=0)
weight = col1.number_input(
    '1. Berat Badan', min_value=10, max_value=999, value=190)

# Convert weight to lbs if the user selected kg
if weight_unit == 'kg':
    weight = weight / 0.45359237

# 2. Height
height_unit = col2.selectbox('Pilih satuan tinggi badan', ('inch', 'cm'), index=0)
height = col2.number_input(
    '2. Tinggi Badan', min_value=10, max_value=250, value=170)

# Convert height to inches if the user selected cm
if height_unit == 'cm':
    height = cm_to_inch(height)

# 3. Age
age = col3.selectbox(
    '3. Umur', ('18 - 24',
                '25 - 29',
                '30 - 34',
                '35 - 39',
                '40 - 44',
                '45 - 49',
                '50 - 54',
                '55 - 59',
                '60 - 64',
                '65 - 69',
                '70 - 74',
                '75 - 79',
                '80 - Lebih Tua'), index=4)

# 4. HighChol
highchol = col1.selectbox(
    "4. Kolesterol Tinggi: Pernahkah Anda diberitahu oleh dokter, perawat, atau profesional kesehatan lainnya bahwa Kolesterol Darah Anda tinggi?",
    ('Ya', 'Tidak'), index=1)

# 5. HighBP
highbp = col2.selectbox(
    "5. Tekanan Darah Tinggi: Pernahkah Anda diberitahu oleh dokter, perawat, atau profesional kesehatan lainnya bahwa Anda memiliki Tekanan Darah Tinggi?",
    ('Ya', 'Tidak'), index=0)

# 6. GenHlth
genhlth = col3.selectbox("6. Kesehatan Umum: Bagaimana Anda memberi peringkat Kesehatan Umum Anda dalam skala dari 1 = Sempurna hingga 5 = Buruk? Pertimbangkan kesehatan fisik dan mental.",
                         ('Sempurna', 'Sangat Baik', 'Baik', 'Cukup', 'Kurang'), index=3)

# Create dataframe:
df1 = pd.DataFrame([[round(weight), round(height), age, highchol, highbp, genhlth]], columns=[
                   'Weight', 'Height', 'Age', 'HighChol', 'HighBP', 'GenHlth'])


def calculate_bmi(weight, height):
    """
    Calculate BMI from weight in lbs and height in inches.
    Args:
        weight: the weight in lbs
        height: the height in inches

    Returns:
        bmi - the body mass index

    """
    bmi = round((703 * weight) / (height ** 2))

    return bmi


def prep_df(df):
    """Prepare user .

    Args:
        df: the dataframe containing the 6 user inputs.

    Returns:
        the dataframe with 5 outputs. BMI, Age, HighChol, HighBP, and GenHlth

    """
    # BMI
    df['BMI'] = df.apply(lambda row: calculate_bmi(
        row['Weight'], row['Height']), axis=1)

    # Drop Weight and Height
    df = df.drop(columns=['Weight', 'Height'])

    # Re-Order columns
    df = df[['BMI', 'Age', 'HighChol', 'HighBP', 'GenHlth']]

    # Age
    df['Age'] = df['Age'].replace({'18 - 24': 1, '25 - 29': 2, '30 - 34': 3, '35 - 39': 4, '40 - 44': 5, '45 - 49': 6,
                                   '50 - 54': 7, '55 - 59': 8, '60 - 64': 9, '65 - 69': 10, '70 - 74': 11, '75 - 79': 12, '80 - Lebih Tua': 13})
    # HighChol
    df['HighChol'] = df['HighChol'].replace({'Ya': 1, 'Tidak': 0})
    # HighBP
    df['HighBP'] = df['HighBP'].replace({'Ya': 1, 'Tidak': 0})
    # GenHlth
    df['GenHlth'] = df['GenHlth'].replace(
        {'Sempurna': 1, 'Sangat Baik': 2, 'Baik': 3, 'Cukup': 4, 'Kurang': 5})

    return df


# prepare the user inputs for the model to accept
df = prep_df(df1)

with st.expander("Lihat data anda"):
    st.write("**User Inputs** ", df1)
with st.expander("Klik untuk melihat apa yang masuk ke dalam Decision Tree untuk prediksi"):
    st.write("**Input Pengguna Disiapkan untuk Decision Tree** ", df,
             "** Perhatikan bahwa BMI dihitung dari Berat dan Tinggi Badan yang Anda masukkan. Usia memiliki 14 kategori dari 1 hingga 13 dengan langkah 5 tahun. HighChol dan HighBP adalah 0 untuk Tidak dan 1 untuk Ya. GenHlth memiliki skala dari 1 = Sangat Baik hingga 5 = Buruk. Ini berasal langsung dari pertanyaan BRFSS yang dipelajari oleh model.")

# load in the model
model = joblib.load('./dt_model.pkl')

# Make the prediction:
if st.button('Klik di sini untuk memprediksi Risiko Diabetes Tipe II Anda'):

    # make the predictions
    prediction = model.predict(df)
    prediction_probability = model.predict_proba(df)
    low_risk_proba = round(prediction_probability[0][0] * 100)
    high_risk_proba = round(prediction_probability[0][1] * 100)

    if(prediction[0] == 0):
        st.write("Anda berada pada **Resiko Rendah** untuk Diabetes Tipe II atau pradiabetes")
        st.write("Prediksi probabilitas risiko rendah",
                 low_risk_proba, "%")
        st.write("Prediksi probabilitas risiko tinggi",
                 high_risk_proba, "%")
    else:
        st.write("Anda berada pada **Resiko Tinggi** untuk Diabetes Tipe II atau pradiabetes")
        st.write("Prediksi probabilitas risiko rendah",
                 low_risk_proba, "%")
        st.write("Prediksi probabilitas risiko tinggi",
                 high_risk_proba, "%")
