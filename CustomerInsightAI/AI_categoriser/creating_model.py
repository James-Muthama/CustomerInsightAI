from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from nltk.stem.lancaster import LancasterStemmer
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import training
from CustomerInsightAI.AI_categoriser.preprocessing_json_file import output
from tensorflow.keras.optimizers import Adam

# Initialize Lancaster Stemmer
stemmer = LancasterStemmer()

# Create a Sequential model
model = Sequential()

# Add input layer specifying the input shape
model.add(Dense(80, input_shape=(training.shape[1],), activation='relu'))

# Add BatchNormalization for stabilization
model.add(BatchNormalization())

# Add dropout for regularization
model.add(Dropout(0.35))

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.35))



# Add output layer with sigmoid activation function
model.add(Dense(len(output[0]), activation='softmax'))

# Use the Adam optimizer with a moderate learning rate
optimizer = Adam(learning_rate=0.00065)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

