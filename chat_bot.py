import random
from nltk.metrics.distance import edit_distance

BOT_CONFIG = {
    'intents': {
        'hello': {
            'questions': ['hello', 'hi'],
            'answers': ['hello', 'hi, dude']
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


def get_intent(text):
    # TODO
    intents = BOT_CONFIG['intents']

    for intent, value in intents.items():
        for example in value['questions']:
            dist = edit_distance(text.lower(), example.lower())
            l = len(example)
            similarity = 1 - dist / l
            if similarity > 0.6:
                return intent


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
    # TODO ...

    # use stub
    failure_phrase = get_failure_phrase()
    return failure_phrase


while True:
    text = input('Enter question')
    answer = generate_answer(text)
    print(answer)
