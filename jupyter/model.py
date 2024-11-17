# import pandas
import pandas as pd

# import datasets
df = pd.read_excel('../data/Social Media Data.xlsx')

# view first five row
df.head()

# check null values
df.isnull().sum()

# drop null values
df.dropna(inplace=True)
df.isnull().sum()

# check shape of the total data
df.shape

# check unique values
df.Tag.unique()

# label encoding
df['Label'] = df['Tag'].map({'Positive':1, 'Negative':0}).astype(int)
df.head()

# check data types
df.info()

# values countes
df['Label'].value_counts()

# create number of characters
df['num_characters'] = df['Text'].apply(len)
df.head(3)

import nltk

# create number of words
df['num_words'] = df['Text'].apply(lambda x:len(nltk.word_tokenize(x)))
df.head(3)

# create number of sentences
df['num_sentences'] = df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head(3)

df[['num_characters','num_words','num_sentences']].describe()

df['Label'].value_counts()

# import data visualization library
import seaborn as sns
import matplotlib.pyplot as plt

# count plot
sns.countplot(x='Label', data=df)

label_counts = df['Label'].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#fe0000','#007e02'])
plt.show()

sns.pairplot(df,hue='Label')

df[df['Label'] == 0][['num_characters','num_words','num_sentences']].describe()

df[df['Label'] == 1][['num_characters','num_words','num_sentences']].describe()

plt.figure(figsize=(8,8))
sns.histplot(df[df['Label'] == 0]['num_characters'],color='green')
sns.histplot(df[df['Label'] == 1]['num_characters'],color='blue')

plt.figure(figsize=(8,8))
sns.histplot(df[df['Label'] == 0]['num_words'],color='orange')
sns.histplot(df[df['Label'] == 1]['num_words'],color='blue')

plt.figure(figsize=(8,8))
sns.histplot(df[df['Label'] == 0]['num_sentences'],color='orange')
sns.histplot(df[df['Label'] == 1]['num_sentences'],color='blue')

import re
import nltk
from nltk.corpus import stopwords
from indicnlp.tokenize import indic_tokenize
nltk.download('stopwords')

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

# Example usage
result = preprocess_text("@RainnyMemories: অনেক দেখলাম, বুঝলাম এই পৃথিবীতে আমাকে বোঝার মতো কেউ নেই। আফসোস :(you don't know yourself.)")
print(result)

df['Text'][2]

preprocess_text('নোয়াখালীতে একজন মারা গেছে পিকেটারদের ইটের আঘাতে :(((')

df['Cleaned'] = df['Text'].apply(preprocess_text)

# find unique words
neg_corpus = []
for msg in df[df['Label'] == 0]['Cleaned'].tolist():
    for word in msg.split():
        neg_corpus.append(word)

neg_corpus

# count all unique words
len(neg_corpus)

# view all unique words
from collections import Counter
Counter(neg_corpus)

# view most common words
Counter(neg_corpus).most_common(10)

# convert to dataframe
pd.DataFrame(Counter(neg_corpus).most_common(10))

# find all words
pos_corpus = []
for msg in df[df['Label'] == 1]['Cleaned'].tolist():
    for word in msg.split():
        pos_corpus.append(word)

pos_corpus

# count all unique words
len(pos_corpus)

# view all uniques words
Counter(pos_corpus)

# view most common words
Counter(pos_corpus).most_common(10)

# convert to dataframe
pd.DataFrame(Counter(pos_corpus).most_common(10))

# import countvectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

# fit and transform
x = cv.fit_transform(df['Cleaned']).toarray()
y = df['Label']

# import train test split
from sklearn.model_selection import train_test_split

# train and test the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# import all necessary metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# import all necessary algortihms
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# initial all algorithm and classifier
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

# conver into dictionary
clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}

# train and test all models
def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    percision = precision_score(y_test,y_pred)
    return accuracy,percision
train_classifier(knc,x_train,y_train,x_test,y_test)

# comparison all models
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf, x_train,y_train,x_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

# conver into models
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Accuracy',ascending=False)

performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")

# plot all models accuracy and percision
sns.catplot(x = 'Algorithm', y='value', hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft':accuracy_scores,'Precision_max_ft':precision_scores}).sort_values('Precision_max_ft',ascending=False)

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)

# merge the dataframe
new_df = performance_df.merge(temp_df,on='Algorithm')

# merge the datagrame
new_df_scaled = new_df.merge(temp_df,on='Algorithm')

# sort the values
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)
new_df_scaled.merge(temp_df,on='Algorithm')

# best model selection
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)

# import voting classifer
from sklearn.ensemble import VotingClassifier

# combine in voting classifer
voting = VotingClassifier(estimators=[('et',etc),('lrc',lrc),('rfc',rfc),('xgb',xgb),('gbdt',gbdt)],voting='soft')

# fit the training model
voting.fit(x_train,y_train)

model = voting
voting = model

# predict x_test data
y_pred = voting.predict(x_test)
print('Accuracy:...',accuracy_score(y_test,y_pred))
print('Precision:...',precision_score(y_test,y_pred))

# import stacking classifer
from sklearn.ensemble import StackingClassifier

# applying best model for stacking
estimators = [('lrc',lrc),('rfc',rfc),('xgb',xgb),('gbdt',gbdt)]
# define final estimators
final_estimator = ExtraTreesClassifier(n_estimators=50, random_state=2)

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

# fit into stacking classsifer
clf.fit(x_train,y_train)

# predict x_test using stacking classifer
y_pred = clf.predict(x_test)
print('Accuracy',accuracy_score(y_test,y_pred))
print('Precision',precision_score(y_test,y_pred))

import pickle

# convert into pickel file
pickle.dump(cv,open('../models/vectorizer.pkl','wb'))
pickle.dump(model,open('../models/model.pkl','wb'))