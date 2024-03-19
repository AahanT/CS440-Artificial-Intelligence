
import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    length = 0
    count0 = 0
    count1 = 0
    for i in range(len(texts)):
        tempcount0 = texts[i].count(word0)
        tempcount1 = texts[i].count(word1)
        newlen = len(texts[i])
        if(newlen >= length):
            length = newlen
        if(tempcount0 >= count0):
            count0 = tempcount0
        if(tempcount1 >= count1):
            count1 = tempcount1
    temp = [[0 for i in range(count1 + 1)] for j in range(count0 + 1)]
    Pjoint = np.array(temp)
    n0 = 0
    n1 = 0
    totaln = len(texts)
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            if(texts[i][j] == word0):
                n0 = n0+1
            if(texts[i][j] == word1):
                n1 = n1+1
        Pjoint[n0][n1] = Pjoint[n0][n1] + 1
        n0 = 0
        n1 = 0
    
    Pjoint = Pjoint / totaln
    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    temp = 0
    row , col = Pjoint.shape
    if(index == 0):
        Pmarginal = np.zeros(row)
        for i in range(row):
            for j in range(col):
                temp = temp + Pjoint[i][j]
            Pmarginal[i] = temp
            temp = 0
    if(index == 1):
        Pmarginal = np.zeros(col)
        for i in range(col):
            for j in range(row):
                temp = temp + Pjoint[j][i]
            Pmarginal[i] = temp
            temp = 0
            
    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    Pcond = np.zeros(Pjoint.shape)
    row, col = Pjoint.shape
    for i in range(row):
        Pcond[i] = Pjoint[i]/Pmarginal[i]         
    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    mu = 0
    for i in range(len(P)):
        mu = mu + i * P[i]
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    var = 0
    exs = 0
    mu = 0
    for i in range(len(P)):
        exs = exs + (i*i*P[i])
        mu = mu + (i*P[i])
    var = exs - mu**2
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    x0 = marginal_distribution_of_word_counts(P, 0)
    x1 = marginal_distribution_of_word_counts(P, 1)
    mu0 = 0
    mu1 = 0
    exy = 0
    
    for i in range(len(P)):
        for j in range(len(P[i])):
            exy = exy + (i*j*P[i][j])
    
    for i in range(len(x0)):
        mu0 = mu0 + (i*x0[i])
    for i in range(len(x1)):
        mu1 = mu1 + (i*x1[i])
        
    covar = exy - mu0*mu1
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    expected = 0
    for i in range(len(P)):
        for j in range(len(P[i])):
             expected = expected + f(i,j)*P[i][j]
    return expected
    
