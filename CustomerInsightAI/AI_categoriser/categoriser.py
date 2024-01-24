from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from CustomerInsightAI.AI_categoriser.preprocessing_input import bag_of_words
from CustomerInsightAI.AI_categoriser.training_the_model import model
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import labels
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import data
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import words

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


text = "I'm considering upgrading my current account to a premium one, and I'd like to understand the benefits and " \
       "the process involved. Can you provide information on the advantages of the upgraded account and how I can " \
       "proceed with the upgrade? Certainly! The premium account offers an array of benefits, including higher " \
       "interest rates, exclusive rewards, and priority customer support. To begin the upgrade process, " \
       "you can either visit your nearest branch or follow the online application procedure on our website. If you " \
       "choose to visit the branch, our staff will guide you through the necessary steps and ensure a smooth " \
       "transition to the premium account. Great, that sounds appealing. Regarding branch visits, are there any " \
       "specific documents I need to bring, and what safety measures are in place amid the ongoing situation? For a " \
       "branch visit, please bring a valid ID, proof of address, and your current account details. In terms of " \
       "safety, we've implemented strict hygiene measures at all our branches. This includes regular sanitization, " \
       "social distancing protocols, and the use of protective equipment by our staff. Your safety is our top " \
       "priority, and we want to ensure a secure and comfortable environment during your visit. Thank you for " \
       "clarifying. I'll gather the necessary documents and plan my visit accordingly. I appreciate your assistance. " \
       "You're welcome! If you have any more questions or need further guidance, feel free to reach out. We're here " \
       "to make the account upgrade process as seamless as possible for you. Have a wonderful day!"

categories, descriptions = categorize(text)
print(categories, descriptions)

if not categories or not descriptions:
    print("Unfortunately I was unable to classify the audio you put in")
else:
    for category, description in zip(categories, descriptions):
        print(f"Category: {category}. Description: {description[0]}")

