import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f=open('Desktop\Data\TacoStandCorpus.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "wazzup","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
similarityCutOff=0.3

lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    #print("cosine similarity vals")
    #print(vals)
    #get the id of the best response, [-1]=the actual typed in phrase so when sorted -2 is the highest match
    idx=vals.argsort()[0][-2]
    idx1=vals.argsort()[0][-3]
    idx2=vals.argsort()[0][-4]
    #however, we have to step back one more from the indices of the top matches to find the proper response to the match
    idList=[idx-1,idx1-1,idx2-1]
    #and we choose randomly from the top 3 matches to respond with
    idT=random.choice(idList)
    #we have to get our response similarity list into a form to sort 
    #based on the actual cosine similarity values and not ids
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    #print("flat[-2] value")
    #print(req_tfidf)
    if(req_tfidf<similarityCutOff):
        robo_response="Thing does not understand. Please rephrase."
        return robo_response
    else:
        #this is how we reference the matched response
        robo_response = sent_tokens[idT]
        return robo_response

flag=True
print("ChatBot: My name is Thing. Have a chat with me! If you want to exit, type 'Bye'")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' or user_response=='gracias' ):
            flag=False
            print("ChatBot: You are welcome..")
        else:
            if(user_response in GREETING_RESPONSE):
                print("ChatBot: "+greeting(user_response))
            else:
                print("ChatBot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ChatBot: Ciao!")
