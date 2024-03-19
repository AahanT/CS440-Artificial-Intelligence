import numpy as np
from collections import Counter
import copy
import math

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    frequency = {}
    for y in train.keys():
        c = Counter()
        for i in range(len(train[y])):
            c.update(train[y][i])
        frequency[y] = c
    return frequency
        

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    nonstop = copy.deepcopy(frequency)
    for y in frequency.keys():
        for x in frequency[y].keys():
            if(x in stopwords):
                if(x in nonstop[y]):
                    del nonstop[y][x]
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    likelihood = copy.deepcopy(nonstop)
    for y in nonstop.keys():
        no_token = 0
        no_type = 0
        for x in nonstop[y].keys():
            if(nonstop[y][x]!=0):
                no_type += 1
            no_token += nonstop[y][x]
        for x in nonstop[y].keys():
            likelihood[y][x] += smoothness
            likelihood[y][x] /= (no_token + smoothness*(no_type+1))
        likelihood[y]['OOV'] = smoothness/(no_token + smoothness*(no_type+1))

    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    for i in range(len(texts)): 
        p_prob = math.log(prior)
        n_prob = math.log(1-prior)
        for k in texts[i]:
            if k not in stopwords:
                if k in likelihood['pos'].keys():
                    p_prob += math.log(likelihood['pos'][k])
                else:
                    p_prob += math.log(likelihood['pos']['OOV'])
                if k in likelihood['neg'].keys():
                    n_prob += math.log(likelihood['neg'][k])
                else:
                    n_prob += math.log(likelihood['neg']['OOV'])
        if(p_prob > n_prob):
            hypotheses.append('pos')
        elif(n_prob > p_prob):
            hypotheses.append('neg')
        else:
            hypotheses.append('undecided')
    return hypotheses
                

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    temp = (len(priors),len(smoothnesses))
    accuracies = np.zeros(temp)
    for m in range(len(priors)):
        for n in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[n])
            hypotheses = naive_bayes(texts, likelihood, priors[m])
            correct_cnt = 0
            for (y,yhat) in zip(labels, hypotheses):
                if y==yhat:
                    correct_cnt += 1
            accuracies[m, n] = correct_cnt / len(labels)
    return accuracies
                          
