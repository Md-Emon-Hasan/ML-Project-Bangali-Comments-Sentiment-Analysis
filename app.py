from flask import Flask
from flask import render_template
from flask import request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from indicnlp.tokenize import indic_tokenize

app = Flask(__name__)

# Necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# load the pre-trained model and vectorizer
voc = pickle.load(open('./models/vectorizer.pkl','rb'))
model = pickle.load(open('./models/model.pkl','rb'))

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text) 
    # Remove hashtags 
    text = re.sub(r'#\w+', '', text)  
    # Clean specific HTML entities and single quotes
    text = re.sub(r'&amp;|<br />|\'', ' ', text)
    # Keep only Bengali characters and whitespace  
    text = re.sub(r'[^\u0980-\u09FF\s]', ' ', text)  
    # Remove numbers (both English and Bengali)
    text = re.sub(r'[0-9০-৯]', '', text) 
    # Remove extra whitespace and trim 
    text = re.sub(r'\s+', ' ', text).strip()  

    # Tokenize the text using the Indic NLP library
    words = indic_tokenize.trivial_tokenize(text, lang='bn')

    # Get the set of Bengali stopwords
    stop_words = set(stopwords.words('bengali'))

    # Filter out stopwords and remove any empty strings
    result = [word for word in words if word and word not in stop_words]

    # Join the filtered words back into a single string
    return ' '.join(result)

@app.route('/',methods=['GET','POST'])
def index():
    result = None
    input_review = ''
    if request.method == 'POST':
        input_review = request.form['review']
        # 1. Preprocess the text
        transformed_sms = preprocess_text(input_review)
        # 2. Vectorize the text
        vector_input = voc.transform([transformed_sms])
        # 3. Predict the result
        result = model.predict(vector_input)[0]
    
    return render_template('index.html',result=result,input_review=input_review) 

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)