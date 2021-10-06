import os
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import nltk
#nltk.download('punkt')
path_business = "business"
path_entertainment = "entertainment"
path_politics = "politics"
path_sport = "sport"
path_tech = "tech"
cwd = os.getcwd()
print(cwd)
train_data = []

# Function to read all the files in a location and add tags and create training data.
def training_function(file_path, category):
    with open(file_path, 'r') as f:
        tuple = f.readline().strip(), category
        train_data.append(tuple)
    f.close()
    print(train_data)
    return (NaiveBayesClassifier(train_data))



# Training data set for Business
os.chdir(path_business)
files = os.listdir('.')
print("Training Business data ...")
for file in files:
    classifier = training_function(file,'business')
print("Training Business data Completed Successfully!!")


# Training data set for Entertainment
os.chdir(cwd)
os.chdir(path_entertainment)
files = os.listdir('.')
print("Training Entertainment data ...")
for file in files:
    classifier = training_function(file, 'entertainment')
print("Training Entertainment data Completed Successfully!!")


# Training data set for Politics
os.chdir(cwd)
os.chdir(path_politics)
print("Training Politics data ...")
files = os.listdir('.')
for file in files:
    classifier = training_function(file, 'politics')
print("Training Politics data Completed Successfully!!")

# Training data set for Sports
os.chdir(cwd)
os.chdir(path_sport)
print("Training Sports data ...")
files = os.listdir('.')
for file in files:
    classifier = training_function(file, 'sports')
print("Training Sports data Completed Successfully!!")

# Training data set for Tech
os.chdir(cwd)
os.chdir(path_tech)
print("Training Technology data ...")
files = os.listdir('.')
for file in files:
    classifier = training_function(file, 'tech')
print("Training Technology data Completed Successfully!!")



print(classifier) # shows how many training data are involved
#### Testing the Trained Data
print(classifier.classify('Amarinder writes to Sonia, expresses reservation over Sidhu as Punjab Congress chief'))
print(classifier.classify('Russian miracle as passengers survive hard plane landing in Siberia'))
print(classifier.classify('White House slams Facebook as conduit for Covid-19 misinformation'))
print(classifier.classify('Hong Kong: US issues warning on business risks'))
print(classifier.classify('WhatsApp is testing a new feature that will let people message without using their phone for the first time'))










