import streamlit as st
import pandas as pd
import joblib 
import os
import xgboost 
import nltk
import re


nltk.download('punkt')
stemmer=nltk.stem.SnowballStemmer('english')
nltk.download('stopwords')
stop_words=set(nltk.corpus.stopwords.words('english'))


def main():

    st.title("Financial Complaints Product type prediction app")
    st.markdown("""
    	#### Description
    	+ This is a simple app for generating classification prediction of product-type using Streamlit.

    	#### Purpose
    	+ To show a simple MLapps using Streamlit framework. 
    	""")
    my_dataset = "complaints.csv"

    @st.cache(persist=True)
    def explore_data(dataset):
    	df = pd.read_csv(os.path.join(dataset))
    	return df

    def tokenize(text):
        tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('Xx/')) > 2 and len(re.sub('\d+', '', word.strip('Xx/'))) > 3) ] 
        tokens = map(str.lower, tokens)
        stems = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
        return stems

    @st.cache(persist=True)
    def load_model():
        xgb_model = joblib.load('NLP_complaint_classification.pkl')
        #xgb_model=xgb.Booster({'nthread':4})
        #xgb_model.load_model(os.path.join('NLP_complaint'))

        vectorizer_feat_transf = joblib.load('feature_transfomer.pkl')

        return xgb_model,vectorizer_feat_transf

    #data = explore_data(my_dataset)
    data = pd.read_csv('complaints.csv')


        # Show Dataset
    if st.checkbox("Preview DataFrame"):
    	if st.button("Head"):
    		st.write(data.head())
    	if st.button("Tail"):
    		st.write(data.tail())
    	else:
    		st.write(data.head(2))

    # Show Entire Dataframe
    if st.checkbox("Show All DataFrame"):
    	st.dataframe(data)

    # Show All Column Names
    if st.checkbox("Show All Column Name"):
    	st.text("Columns:")
    	st.write(data.columns)

    # Show Dimensions and Shape of Dataset
    data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
    if data_dim == 'Rows':
    	st.text("Showing Length of Rows")
    	st.write(len(data))
    if data_dim == 'Columns':
    	st.text("Showing Length of Columns")
    	st.write(data.shape[1])

    # Show Summary of Dataset
    if st.checkbox("Show Summary of Dataset"):
    	st.write(data.describe())

    samples_option = st.selectbox('Select Random Sample',(5,10,15,20,25))

    if st.button('Generate Scores'):
        xgb_model,vectorizer_feat_transf=load_model()

        val_complaints=data.sample(samples_option).complaints
        
        val_vectors=vectorizer_feat_transf.transform(val_complaints)

        predictions_val=xgb_model.predict(val_vectors)
        prediction_proba = xgb_model.predict_proba(val_vectors)

        target_encoding_label={'Checking or savings account': 3,
        'Credit card or prepaid card': 1,
        'Debt collection': 0,
        'Mortgage': 2,
        'Student loan': 4,
        'Vehicle loan or lease': 5}

        pred_label=pd.Series(predictions_val).map(dict(map(reversed,target_encoding_label.items())))
        pred_df=pd.DataFrame.from_dict({'complaints':val_complaints.tolist(),'predictions':pred_label.tolist(),'pred_prob':prediction_proba.tolist()})
        
        st.subheader("Prediction")
        st.write(pred_df)

    
    if st.checkbox("By"):
        st.text("Shankar Chavan")
        st.text("shanky12@gmail.com")



if __name__ == "__main__":
    main()

