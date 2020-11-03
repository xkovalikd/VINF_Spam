import re
import string
import email
import json
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#nespracovany = all_mails[0]['Body']
#print(nespracovany)
#spracovany = clean_up_pipeline(nespracovany)
#print(spracovany)

def process_email(message):
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    sw = stopwords.words('english')
    words = [word for word in words if word not in sw]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]   
    return words

def add_whitespace(word):
    result = word.replace(',',', ')
    result = result.replace('?','? ')
    result = result.replace('.','. ')
    result = result.replace('!','! ')
    return result

def remove_hyperlink(word):
    return  re.sub(r"http\S+", "", word)

def to_lower(word):
    result = word.lower()
    return result

def remove_number(word):
    result = re.sub(r'\d+', '', word)
    return result

def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result

def remove_whitespace(word):
    result = word.strip()
    return result

def replace_newline(word):
    return word.replace('\n','')

def clean_up_pipeline(sentence):
    cleaning_utils = [add_whitespace,
                      remove_hyperlink,
                      replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation,
                      remove_whitespace,
                      process_email]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence

with open(r'C:\Users\admin\Desktop\ING_1.ročník\ZS_2020-2021\VINF\trec07p\trec07p\full\index', 'r') as ifile:
    raw_labels = ifile.readlines()

all_mails = []
for label in raw_labels:
    mail = {}
    match_0 = re.search(r'((?:sp|h)am) ../data/inmail.(\d{1,})', label)
    if match_0:
        class_ = match_0.group(1)
        email_num = match_0.group(2)
        mail['Email_number'] = email_num
        mail['Class'] = class_
        x = mail['Email_number']
        
    with open(f'C:/Users/admin/Desktop/ING_1.ročník/ZS_2020-2021/VINF/trec07p/trec07p/data/inmail.{x}', 'rb') as email_file:
        message = email.message_from_binary_file(email_file)
        body = ''
        
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))
                
                if (content_type in ['text/html', 'text/txt'] and 'attachment' not in content_disposition):
                    body = part.get_payload(decode=True)
                    break
        else:
            body = message.get_payload(decode=True)
            
        mail['Body'] = BeautifulSoup(body, 'html.parser').get_text(strip=True)

    with open(f'C:/Users/admin/Desktop/ING_1.ročník/ZS_2020-2021/VINF/trec07p/trec07p/data/inmail.{x}', 'r', encoding='ISO-8859-1') as email_file:
        for riadok in email_file.readlines():
            match_1 = re.search(r'^From: (.*)$', riadok)
            if match_1:
                mail['Sender'] = match_1.group(1)
            match_2 = re.search(r'^To: (.*)$', riadok)
            if match_2:
                mail['Receiver'] = match_2.group(1)
            match_3 = re.search(r'^Date: (.*)$', riadok)
            if match_3:
                mail['Date'] = match_3.group(1)
            match_4 = re.search(r'^Subject: (.*)$', riadok)
            if match_4:
                mail['Subject'] = match_4.group(1)
    
    all_mails.append(mail)

with open("data_proccessed.json", 'w') as file:
    file.write('\n'.join(json.dumps(i) for i in all_mails))

## odfiltrovat riadky z emailu, ktore nie su konkretne odpovedou na predchadzajuci eamil, tyn. riadky zacinajuce znakom <
## vymysliet co s emailami, ktore obsahuju iba link, resp. obrazok, pretoze neviem ziskat nic z textu
## vziat do uvahy nie len cisty text, ale v pripade html sablony aj linky a odkazy na obrazky

## spracuj aj zvysne 2 subory
## pridaj nejaky ML model na porovnanie vysledkov

import pandas as pd
import numpy as np

emails = pd.read_json('data_proccessed.json', lines=True)
emails.drop(['Date', 'Receiver', 'Subject'], axis = 1, inplace = True)
emails['Class'] = emails['Class'].map({'ham': 0, 'spam': 1})

trainIndex, testIndex = list(), list()
for i in range(emails.shape[0]):
    if np.random.uniform(0, 1) < 0.80:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = emails.loc[trainIndex]
testData = emails.loc[testIndex]

trainData.reset_index(inplace = True)
trainData.drop(['index'], axis = 1, inplace = True)
trainData.head()

testData.reset_index(inplace = True)
testData.drop(['index'], axis = 1, inplace = True)
testData.head()

#testData['Class'].value_counts()
#trainData['Class'].value_counts()

from math import log, sqrt

class SpamClassifier(object):
    def __init__(self, trainData, method = 'tf-idf'):
        self.mails, self.labels = trainData['Body'], trainData['Class']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + \
                                                                len(list(self.tf_spam.keys())))
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words + \
                                                                len(list(self.tf_ham.keys())))
        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails 


    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(noOfMessages):
            message_processed = clean_up_pipeline(self.mails[i])
            count = list() 

            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_mails + self.ham_mails) \
                                                          / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
            
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails) \
                                                          / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
            
    
        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails 
                    
    def classify(self, processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:                
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                if self.method == 'tf-idf':
                    pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                else:
                    pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                if self.method == 'tf-idf':
                    pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys()))) 
                else:
                    pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam
    
    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)

sc_tf_idf = SpamClassifier(trainData, 'tf-idf')
sc_tf_idf.train()
preds_tf_idf = sc_tf_idf.predict(testData['Body'])
metrics(testData['Class'], preds_tf_idf)
