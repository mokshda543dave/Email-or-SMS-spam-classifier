import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title='Spam Classifier')

st.title("Email/SMS Spam Classifier")

ps = PorterStemmer()


# function to perform data processing
def transform_text(text):
    # lower case
    text = text.lower()

    # tokenize into words
    text = nltk.word_tokenize(text)

    # removing special chars
    remspec = []
    for x in text:
        if x.isalnum():
            remspec.append(x)
    text = remspec[:]
    remspec.clear()

    # removing stopwords and punctuations
    for x in text:
        if x not in stopwords.words('english') and x not in string.punctuation:
            remspec.append(x)
    text = remspec[:]
    remspec.clear()

    # stemming
    ps = PorterStemmer()
    for x in text:
        remspec.append(ps.stem(x))

    return " ".join(remspec)

input_sms = st.text_area("Enter the message : ")

if st.button("Predict"):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")