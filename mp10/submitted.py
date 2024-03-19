import numpy as np

epsilon = 1e-3

def find_prob(r, c, a, model):
    '''
    Calculate transition probabilities for each action from a given state in the grid world.

    Parameters:
    r - row index of the current state
    c - column index of the current state
    a - action to take (0: left, 1: up, 2: right, 3: down)
    model - the MDP model returned by load_MDP()

    Returns:
    probs[r', c'] - the probability of transitioning to state (r', c') from state (r, c) if action a is taken.
    '''
    
    probs = np.zeros((model.M, model.N))
    if a == 0: #left
        if c-1 < 0 or model.W[r, c-1]:
            probs[r, c] += model.D[r, c, 0]
        else:
            probs[r, c-1] = model.D[r, c, 0]
        
        if r+1 == model.M or model.W[r+1, c]:
            probs[r, c] += model.D[r, c, 1]
        else:
            probs[r+1, c] = model.D[r, c, 1]
        
        if r-1 < 0 or model.W[r-1, c]:
            probs[r, c] += model.D[r, c, 2]
        else:
            probs[r-1, c] = model.D[r, c, 2]
        
    elif a == 1: #up
        if r-1 < 0 or model.W[r-1, c]:
            probs[r, c] += model.D[r, c, 0]
        else:
            probs[r-1, c] = model.D[r, c, 0]
        
        if c-1 < 0 or model.W[r, c-1]:
            probs[r, c] += model.D[r, c, 1]
        else:
            probs[r, c-1] = model.D[r, c, 1]
        
        if c+1 == model.N or model.W[r, c+1]:
            probs[r, c] += model.D[r, c, 2]
        else:
            probs[r, c+1] = model.D[r, c, 2]
        
    elif a == 2: #right
        if c+1 == model.N or model.W[r, c+1]:
            probs[r, c] += model.D[r, c, 0]
        else:
            probs[r, c+1] = model.D[r, c, 0]
        
        if r-1 < 0 or model.W[r-1, c]:
            probs[r, c] += model.D[r, c, 1]
        else:
            probs[r-1, c] = model.D[r, c, 1]
        
        if r+1 == model.M or model.W[r+1, c]:
            probs[r, c] += model.D[r, c, 2]
        else:
            probs[r+1, c] = model.D[r, c, 2]
        
    else: # down
        if r+1 == model.M or model.W[r+1, c]:
            probs[r, c] += model.D[r, c, 0]
        else:
            probs[r+1, c] = model.D[r, c, 0]
        
        if c+1 == model.N or model.W[r, c+1]:
            probs[r, c] += model.D[r, c, 1]
        else:
            probs[r, c+1] = model.D[r, c, 1]
        
        if c-1 < 0 or model.W[r, c-1]:
            probs[r, c] += model.D[r, c, 2]
        else:
            probs[r, c-1] = model.D[r, c, 2]
            
    return probs



def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    #raise RuntimeError("You need to write this part!")
    P = np.zeros((model.M, model.N, 4, model.M, model.N))
    #dirs = ['up', 'down', 'left', 'right']
    
    for r in range(model.M):
        for c in range(model.N):
            for a in range(4):
                if not model.T[r,c]:
                    P[r, c, a] = find_prob(r, c, a, model)
    
    return P

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    U_next = np.zeros((U_current.shape[0], U_current.shape[1]))
    for r in range(U_current.shape[0]):
    	for c in range(U_current.shape[1]):
    		temp = 0
    		for a in range(P.shape[2]):
    			#x = np.dot(P[r,c,a,:,:] , U_current.T).sum()
    			x = 0 
    			for i in range(U_current.shape[0]):
    				for j in range(U_current.shape[1]):
    					x += P[r,c,a,i,j] * U_current[i,j]
    			if temp < x:
    				temp = x
    		U_next[r,c] = model.R[r,c] + model.gamma*temp
    return U_next



def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    P = compute_transition_matrix(model)
    U_current = np.zeros((model.M, model.N))
    U_next = np.zeros((model.M, model.N))
    for i in range(100):
    	U_next = update_utility(model, P, U_current)
    	check = 1
    	for r in range(U_current.shape[0]):
    		for c in range(U_current.shape[1]):
    			if abs(U_current[r,c] - U_next[r,c]) > epsilon:
    				check = 0
    	if check:
    		return U_next
    	U_current = U_next


if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
