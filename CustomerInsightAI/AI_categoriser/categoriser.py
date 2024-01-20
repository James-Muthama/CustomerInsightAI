from nltk.stem.lancaster import LancasterStemmer
from CustomerInsightAI.AI_categoriser.converting_input_to_numpy_array import bag_of_words
from CustomerInsightAI.AI_categoriser.training_the_model import model
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import labels
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import data
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import words

stemmer = LancasterStemmer()


# prompts user for prompt to be used in the chatbot
def categoriser(customer_conversation):
    # takes in text from user passes it into the function bag_of_words
    results = model.predict([bag_of_words(customer_conversation, words)])[0]

    # stores tags and responses for results with probability > 0.7
    category = []

    # iterate through results
    for i, probability in enumerate(results):
        # check if probability is greater than 0.7
        if probability > 0.7:
            # retrieve corresponding tag
            tag = labels[i]

            # find tag in data["intents"]
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    # retrieve responses for the tag
                    responses = tg["responses"]
                    # add tag and responses to the list
                    category.append({"Category": tag, "Context of conversation": responses})

                    return category


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
response = categoriser(text)
print(response)
