from tensorflow.keras.models import load_model
from CustomerInsightAI.AI_categoriser.training_the_model import train_model


def load_or_train_model():
    try:
        model = load_model("CustomerInsightAI.tflearn")
    except:
        model = train_model()
    return model


