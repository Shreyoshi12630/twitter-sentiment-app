
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.title("Twitter Sentiment Analysis with ANN")
uploaded_file = st.file_uploader("Upload the Tweets CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.dropna(subset=['text', 'sentiment'], inplace=True)

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_text'] = df['text'].apply(clean_text)

    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['sentiment'])

    st.subheader("Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='sentiment', data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("WordClouds for Each Sentiment")
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        text = " ".join(subset['clean_text'].tolist())
        wordcloud = WordCloud(stopwords=stop_words).generate(text)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis("off")
        st.pyplot(fig2)

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = to_categorical(df['label'], num_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, input_dim=2000, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=[early_stop], verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test)
    st.success(f"Model Test Accuracy: {accuracy:.4f}")

    st.subheader("Model Accuracy & Loss")
    fig3, ax3 = plt.subplots()
    ax3.plot(history.history['accuracy'], label='Train Accuracy')
    ax3.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax3.set_title("Accuracy")
    ax3.legend()
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    ax4.plot(history.history['loss'], label='Train Loss')
    ax4.plot(history.history['val_loss'], label='Validation Loss')
    ax4.set_title("Loss")
    ax4.legend()
    st.pyplot(fig4)

    def predict_sentiment_nn(tweet):
        cleaned = clean_text(tweet)
        vect = vectorizer.transform([cleaned]).toarray()
        pred = model.predict(vect)[0]
        return encoder.inverse_transform([np.argmax(pred)])[0]

    st.subheader("Try Predicting a Tweet Sentiment")
    user_input = st.text_area("Enter a tweet:")
    if user_input:
        prediction = predict_sentiment_nn(user_input)
        st.info(f"Predicted Sentiment: {prediction}")
