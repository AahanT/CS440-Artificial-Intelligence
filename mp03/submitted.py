import numpy as np
from collections import Counter

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''
    dist = np.zeros(len(train_images))
    for i in range(len(train_images)):
        dist[i] = np.linalg.norm(image - train_images[i])
    sorted_dist = np.argsort(dist)
    index = sorted_dist[:k]
    neighbors = np.zeros((k, len(image)))
    labels = [0 for i in range(k)]
    for i in range(k):
        neighbors[i] = train_images[index[i]]
        labels[i] = train_labels[index[i]]
    return neighbors, labels

def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    neighbors = np.zeros((k, len(dev_images[0])))
    labels = [0 for i in range(k)]
    hypotheses = [0 for i in range(len(dev_images))]
    scores = [0 for i in range(len(dev_images))]
    for i in range(len(dev_images)):
        neighbors,labels= k_nearest_neighbors(dev_images[i], train_images, train_labels, k)
        true_cnt = 0
        false_cnt = 0
        for j in range(len(labels)):
            if(labels[j] == True):
                true_cnt += 1
            else:
                false_cnt += 1
        if(true_cnt > false_cnt):
            hypotheses[i] = True
            scores[i] = true_cnt
        elif(false_cnt > true_cnt):
            hypotheses[i] = False
            scores[i] = false_cnt
        else:
            hypotheses[i] = False
            scores[i] = false_cnt
    return hypotheses, scores


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''
    confusions = np.zeros((2,2))
    true_neg = 0
    false_pos = 0
    false_neg = 0
    true_pos = 0
    
    for i in range(len(hypotheses)):
        if(hypotheses[i] == True):
            if(hypotheses[i] == references[i]):
                true_pos += 1
            else:
                false_pos +=1
        else:
            if(hypotheses[i] == references[i]):
                true_neg += 1
            else:
                false_neg +=1
    confusions[0][0] = true_neg
    confusions[0][1] = false_pos
    confusions[1][0] = false_neg
    confusions[1][1] = true_pos
    
    accuracy = (true_pos + true_neg)/ (true_pos+true_neg+false_pos+false_neg)
    f1 = 2/ ((1/ (true_pos/(true_pos+false_neg))) + (1/ (true_pos/(true_pos+false_pos))))
    
    return confusions, accuracy, f1
           
    
    
