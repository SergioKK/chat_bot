# Edit dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BOT_CONFIG = {
    'intents': {
        'hello': {
            'questions': ['hello', 'hi', 'hi, bro'],
            'answers': ['hello', 'hi, dude', 'hi, bro']
        },
        'bye': {
            'questions': ['good bye', 'see you'],
            'answers': ['good bye', 'see you']
        }
    },
    'failure_phrases': [
        'i don`t know what to answer',
        'i don`t understand you'
    ]
}

X_text = []  # texts
y = []  # classes/intents

for intent, value in BOT_CONFIG['intents'].items():
    for example in value['questions']:
        X_text.append(example)
        y.append(intent)

# vectorization

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

# model learning

clf = LogisticRegression(random_state=0).fit(X, y)
# print(clf.predict(X[:2, :]))

# quantity of the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf1 = LogisticRegression()
clf1.fit(X_train, y_train)
print(clf1.score(X_train, y_train))

