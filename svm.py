import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('spam_dataset.csv')

# Split the dataset into features and labels
features = data['email_text']
labels = data['spam']

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test case
test_email = ['Congratulations! You have won a free vacation. Click here to claim your prize.']

# Convert the test email to numerical features using the TF-IDF vectorizer
test_features = vectorizer.transform(test_email)

# Predict the label for the test email
prediction = classifier.predict(test_features)

# Print the predicted label
if prediction[0] == 0:
    print("The email is not spam.")
else:
    print("The email is spam.")
