import pandas as pd
import nltk
import string

# Step 1: Loading the Dataset
data_file = 'EmailSpamCollection.txt'
data = pd.read_csv(data_file, sep='\t', header=None, names=['label', 'email'])

# Step 2: Pre-Processing
# load stopwords and punctuation
nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

# pre-process function content
def pre_process(email):
    lowercase = "".join([char.lower() for char in email if char not in punctuation])
    tokenize = nltk.tokenize.word_tokenize(lowercase)
    remove_stopwords = [word for word in tokenize if word not in stopwords]
    return remove_stopwords

data['processed'] = data['email'].apply(lambda x: pre_process(x))


# Step 3: Categorizing and Counting Tokens
# categorizing ham /spam associated words 
def categorize_words():
    spam_words = []
    ham_words = []
    # spam associated words
    for email in data['processed'][data['label'] == 'spam']:
        for word in email:
            spam_words.append(word)
    # ham associated words
    for email in data['processed'][data['label'] == 'ham']:
        for word in email:
            ham_words.append(word)
    return spam_words, ham_words

spam_words, ham_words = categorize_words()

# Step 4: Predict Function
def predict(user_input):
    spam_counter = 0
    ham_counter = 0
    for word in user_input:
        spam_counter += spam_words.count(word)
        ham_counter += ham_words.count(word)
    print('******** Results ********')
    if ham_counter > spam_counter:
        accuracy = round((ham_counter / (ham_counter + spam_counter) * 100), 2)
        print('the email is not spam, with {}%  accuracy'.format(accuracy))
    elif spam_counter > ham_counter:
        accuracy = round((spam_counter / (ham_counter + spam_counter) * 100), 2)
        print('the email is spam, with {}%  accuracy'.format(accuracy))
    else:
        print('the email could be spam')

# Step 5: Collecting User Input
user_input = input('Please type a email \n')
processed_input = pre_process(user_input)
predict(processed_input)