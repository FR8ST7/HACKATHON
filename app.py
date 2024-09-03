from flask import Flask, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(text)

@app.route('/', methods=['GET', 'POST'])
def home():
    background_image = 'https://wallpapercave.com/wp/wp2691486.jpg'
    if request.method == 'POST':
        input_sms = request.form['message']
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        prediction = "Spam" if result == 1 else "Not Spam"
        return render_template('result.html', prediction=prediction, message=input_sms, background_image=background_image)
    
    return render_template('index.html', background_image=background_image)

if __name__ == '__main__':
    app.run(debug=True)
