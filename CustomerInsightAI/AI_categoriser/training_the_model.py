from CustomerInsightAI.AI_categoriser.creating_model import model
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import training
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import output
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


# tries to load the model if it is not available we will need to train it
try:
    model.load("CustomerInsightAI.tflearn")

except:
    # Define a TensorBoard callback for visualizing metrics
    tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

    # Define an EarlyStopping callback to prevent overfitting
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # fitting the neural network with the training data, output, number of epochs, batch size and where it will show
    # metrics such as accuracy etc
    model.fit(training, output, epochs=75, batch_size=20, validation_split=0.2, callbacks=[tensorboard_callback, early_stopping_callback])

    # saving the model as SmartNannyBot.tflearn
    model.save("CustomerInsightAI.tflearn")
