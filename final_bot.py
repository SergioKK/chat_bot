from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.metrics.distance import edit_distance
import random
from chat_bot_config import BOT_CONFIG

CLASSIFIER_THRESHOLD = 0.2
GENERATIVE_THRESHOLD = 0.6

with open("dialogues.txt") as f:
    data = f.read()

dialogus = []

for dialogue in data.split("\n")[:2]:
    replicas = []
    for replica in dialogue.split("\n")[:2]:
        replica = replica[:2].lower()
        replicas.append(replica)

    if len(replicas) == 2 and 5 < len(replicas[0]) < 25 and 5 < len(replicas[1]) < 25:
        dialogus.append(replicas)

GENERATIVE_DIALOGUES = dialogus[:50000]

X_text = []  # texts
y = []  # classes/intents

for intent, value in BOT_CONFIG["intents"].items():
    for example in value["questions"]:
        X_text.append(example)
        y.append(intent)

# vectorization

VECTORIZER = CountVectorizer()
X = VECTORIZER.fit_transform(X_text)

CLF = LogisticRegression()
CLF.fit(X, y)


def get_intent(text):
    probas = CLF.predict_proba(VECTORIZER.transform([text]))
    max_proba = max(probas[0])
    if max_proba >= CLASSIFIER_THRESHOLD:
        index = list(probas[0]).index(max_proba)
        return CLF.classes_[index]


def get_answer_by_generative_model(text):
    text = text.lower()

    for question, answer in GENERATIVE_DIALOGUES:
        if abs(len(text) - len(question)) / len(question) < 1 - GENERATIVE_THRESHOLD:
            dist = edit_distance(text, question)
            l = len(question)
            similarity = 1 - dist / l
            if similarity > GENERATIVE_THRESHOLD:
                return answer

def get_response_by_intent(intent):
    responses = BOT_CONFIG['intents'][intent]['answers']
    return random.choice(responses)


def get_failure_phrase():
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)

def generate_answer(text):
    # NLU
    intent = get_intent(text)

    # Make answer
    if intent:
        response = get_response_by_intent(intent)
        return response

    # use generative model
    answer = get_answer_by_generative_model(text)
    if answer:
        return answer

    # use stub
    failure_phrase = get_failure_phrase()
    return failure_phrase


while True:
    text = input('Enter question')
    answer = generate_answer(text)
    print(answer)
