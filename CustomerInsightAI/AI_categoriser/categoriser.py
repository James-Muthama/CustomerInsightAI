from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from CustomerInsightAI.AI_categoriser.preprocessing_input import bag_of_words
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import labels
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import data
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import words
from CustomerInsightAI.AI_categoriser.loading_model import load_or_train_model

# Load or train the model
model = load_or_train_model()

stemmer = LancasterStemmer()


# prompts user for prompt to be used in the chatbot
def categorize(customer_conversation):
    # Split the customer text into sentences using a period as the delimiter
    sentences = customer_conversation.split('.')

    print(len(sentences))

    # arrays that hold the category and description chosen by the model
    category_arr = []
    description_arr = []

    for sentence in sentences:
        print(sentence)

        # takes in text from user passes it into the function bag_of_words
        results = model.predict([bag_of_words(sentence, words)])[0]

        print(results)

        # takes back the prediction with the highest probability
        results_index = np.argmax(results)

        # finds the tag with matching probability
        tag = labels[results_index]

        # only allows prediction with a 50% chance to be passed to the user any fewer passes an alternate response
        if results[results_index] > 0.5:
            category_arr.append(tag)
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    response = tg["responses"]
                    description_arr.append(response)

    return category_arr, description_arr


