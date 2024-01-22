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

    # arrays that hold the category and description chosen by the model
    category_arr = []
    description_arr = []

    for sentence in sentences:
        # takes in text from user passes it into the function bag_of_words
        results = model.predict([bag_of_words(sentence, words)])[0]

        # takes back the prediction with the highest probability
        results_index = np.argmax(results)

        # finds the tag with matching probability
        tag = labels[results_index]

        # only allows prediction with a 70% chance to be passed to the user any fewer passes an alternate response
        if results[results_index] > 0.7:
            category_arr.append(tag)
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    description = tg["responses"]
                    description_arr.append(description)

    return category_arr, description_arr


text = "Hello, I hope you're well. I recently applied for a personal loan with ABSA, and I was wondering if there's " \
       "any update on the status of my application. Good day! I appreciate your inquiry. Let me check that for you. " \
       "Could you please provide me with your application reference number? Certainly, it's 123456789. Thank you. " \
       "I'll just take a moment to review the status. Please bear with me. I appreciate your patience. It appears " \
       "that your application is currently under review by our loan department. The decision process typically takes " \
       "a few business days. Is there anything specific you would like to know or any additional information you'd " \
       "like to provide? No, that's fine. I'll wait for the decision. Certainly. I've noticed three transactions in " \
       "the past week that I didn't authorize. The amounts are R500, R1,200, and R800. I don't recognize the " \
       "merchants associated with these transactions.. I'm sorry to hear that you're facing this issue. Your account " \
       "security is of utmost importance to us. Could you please provide more details about the transactions in " \
       "question? For instance, the transaction dates, amounts, or any merchant information would be helpful. I'm " \
       "sorry for any inconvenience this may have caused. I'll initiate an investigation into these transactions " \
       "immediately. In the meantime, I recommend placing a temporary hold on your account for added security. " \
       "Additionally, we'll issue a dispute for the unauthorized transactions. We take these matters seriously and " \
       "will work swiftly to resolve them. Thank you for your prompt action. I appreciate your assistance. You're " \
       "welcome. We're committed to ensuring the security of your account. If you have any further questions or " \
       "concerns, feel free to reach out. Is there anything else I can assist you with today? No, that covers " \
       "everything for now. Thank you for your help.  It's our pleasure. Have a great day, and we'll keep you updated " \
       "on both the loan application and the investigation into the unauthorized transactions."

categories, descriptions = categorize(text)

if not categories or not descriptions:
    print("Unfortunately I was unable to classify the audio you put in")
else:
    for category, description in zip(categories, descriptions):
        print(f"Category: {category}. Description: {description}")

