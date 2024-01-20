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
model.add(Dense(32, input_shape=(len(training[0]),), activation='relu'))

# Add BatchNormalization for stabilization
model.add(BatchNormalization())

# Add dropout for regularization
model.add(Dropout(0.45))

# Add hidden layers
model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.45))
model.add(Dense(16, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.45))


# Add output layer with sigmoid activation function
model.add(Dense(len(output[0]), activation='sigmoid'))

# Use the Adam optimizer with a moderate learning rate
optimizer = Adam(learning_rate=0.0005)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

