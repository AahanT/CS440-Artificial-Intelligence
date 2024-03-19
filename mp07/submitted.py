import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    standardized_rules = copy.deepcopy(nonstandard_rules)
    variables = []
    d = -15

    for ruleIDs in standardized_rules:

      if len(standardized_rules[ruleIDs]['antecedents']) != 0:

        for i in range(len(standardized_rules[ruleIDs]['antecedents'])):
          for j in range(len(standardized_rules[ruleIDs]['antecedents'][i])):

            if standardized_rules[ruleIDs]['antecedents'][i][j] == 'something':
              standardized_rules[ruleIDs]['antecedents'][i][j] = 'x000' + str(d)
              if 'x000' + str(d) not in variables:
                variables.append('x000'+str(d))

      if len(standardized_rules[ruleIDs]['consequent']) != 0:

        for i in range(len(standardized_rules[ruleIDs]['consequent'])):

          if standardized_rules[ruleIDs]['consequent'][i] == 'something':
            standardized_rules[ruleIDs]['consequent'][i] = 'x000' + str(d)
            if 'x000' + str(d) not in variables:
              variables.append('x000'+str(d))

      d = d + 1
    return standardized_rules, variables

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    unification = []
    subs = dict()
    if query[1] == datum[1] and query[3] == datum[3]:

      if query[0] == query[2]:

        if query[0] in variables:
          if datum[0] in variables:
            if datum[2] not in variables:
              unification = [datum[2], datum[1], datum[2], datum[3]]
              if query[0] not in subs:
                subs[query[0]] = datum[0]
              if datum[0] not in subs:
                subs[datum[0]] = datum[2]

          elif datum[2] in variables:
            if datum[0] not in variables:
              unification = [datum[0], datum[1], datum[0], datum[3]]
              if query[0] not in subs:
                subs[query[0]] = datum[2]
              if datum[2] not in subs:
                subs[datum[2]] = datum[0]

        else:
          if datum[0] in variables:
            if datum[2] in variables:
              unification = copy.deepcopy(query)
              if datum[0] not in subs:
                subs[datum[0]] = query[0]
                subs[datum[2]] = query[2]


      elif datum[0] == datum[2]:

        if datum[0] in variables:
          if query[0] in variables:
            if query[2] not in variables:
              unification = [query[2], query[1], query[2], query[3]]
              if query[0] not in subs:
                subs[query[0]] = datum[0]
              if datum[2] not in subs:
                subs[datum[2]] = query[2]

          elif query[2] in variables:
            if query[0] not in variables:
              unification = [query[0], query[1], query[0], query[3]]
              if query[2] not in subs:
                subs[query[2]] = datum[2]
              if datum[0] not in subs:
                subs[datum[0]] = query[0]

        else:
          if query[0] in variables:
            if query[2] in variables:
              unification = copy.deepcopy(datum)
              if query[0] not in subs:
                subs[query[0]] = datum[0]
                subs[query[2]] = datum[2]

      elif query[0] in variables:

        if query[2] in variables:
          unification = copy.deepcopy(datum)
          if query[0] not in subs:
            subs[query[0]] = datum[0]
          if query[2] not in subs:
            subs[query[2]] = datum[2]

        elif datum[2] in variables:
          unification = [datum[0], datum[1], query[2], datum[3]]
          if query[0] not in subs:
            subs[query[0]] = datum[0]
          if datum[2] not in subs:
            subs[datum[2]] = query[2]

        elif datum[2] == query[2]:
          unification = [datum[0], datum[1], query[2], datum[3]]
          if query[0] not in subs:
            subs[query[0]] = datum[0]
          if datum[2] not in subs:
            subs[datum[2]] = query[2]

      elif datum[0] in variables:

        if datum[2] in variables:
          unification = copy.deepcopy(query)
          if datum[0] not in subs:
            subs[datum[0]] = query[0]
          if datum[2] not in subs:
            subs[datum[2]] = query[2]

        elif query[2] in variables:
          unification = [query[0], query[1], datum[2], query[3]]
          if datum[0] not in subs:
            subs[datum[0]] = query[0]
          if query[2] not in subs:
            subs[query[2]] = datum[2]

        elif datum[2] == query[2]:
          unification = [query[0], query[1], datum[2], query[3]]
          if query[0] not in subs:
            subs[query[0]] = datum[0]
          if datum[2] not in subs:
            subs[datum[2]] = query[2]
      
      elif datum[0] == query[0]:

        if query[2] in variables:
          unification = copy.deepcopy(datum)
          if query[0] not in subs:
            subs[query[0]] = datum[0]
          if query[2] not in subs:
            subs[query[2]] = datum[2]

        elif datum[2] in variables:
          unification = [datum[0], datum[1], query[2], datum[3]]
          if query[0] not in subs:
            subs[query[0]] = datum[0]
          if datum[2] not in subs:
            subs[datum[2]] = query[2]

        elif datum[2] == query[2]:
          unification = [datum[0], datum[1], query[2], datum[3]]
          if query[0] not in subs:
            subs[query[0]] = datum[0]
          if datum[2] not in subs:
            subs[datum[2]] = query[2]



    if len(unification) == 0:
      return None, None
    return unification, subs

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    applications = []
    goalsets = []
    for i in range(len(goals)):
      unification, subs = unify(goals[i], rule['consequent'], variables)
      
      if unification != None:
        newRule = copy.deepcopy(rule)
        newGoal = copy.deepcopy(goals)
        del newGoal[i]
        for j in range(len(newRule['antecedents'])):
          newvar0 = unification[0]
          newvar2 = unification[2]
          unificationa = []

          if newRule['antecedents'][j][0] == newRule['consequent'][0] and newRule['antecedents'][j][2] == newRule['consequent'][2] and newRule['antecedents'][j][2] in variables:
            unificationa = [unification[0], newRule['antecedents'][j][1], unification[2], newRule['antecedents'][j][3]]
          elif newRule['antecedents'][j][0] == newRule['consequent'][0] and newRule['antecedents'][j][0] in variables:
            unificationa = [unification[0], newRule['antecedents'][j][1], newRule['antecedents'][j][2], newRule['antecedents'][j][3]]
          elif newRule['antecedents'][j][2] == newRule['consequent'][2] and newRule['antecedents'][j][2] in variables:
            unificationa = [newRule['antecedents'][j][0], newRule['antecedents'][j][1], unification[2], newRule['antecedents'][j][3]]

          newRule['antecedents'][j] = unificationa
          newGoal.append(unificationa)
        unificationc, subsc = unify(goals[i], newRule['consequent'], variables)
        newRule['consequent'] = unificationc
        goalsets.append(newGoal)
        applications.append(newRule)
      
     


    return applications, goalsets

def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    #raise RuntimeError("You need to write this part!")
    # proof = []
    # if query[0] in variables or query[2] in variables:
    #   proof.append(query)
    #   return proof
    # for key in rules:
    #   print(rules[key])
    #   head = rules[key]['consequent']
    #   body = []
    #   for j in range(len(rules[key]['antecedents'])):
    #     body.append(rules[key]['antecedents'][j])
    #   print(body)
    #   if head == query:
    #     for condition in body:
    #       subproof = backward_chain(condition, rules, variables)
    #       if len(subproof) == 0:
    #         return None
    #       proof.append(subproof)
    #     proof.append(query)
    #     return proof

    proof = []
    for rule in rules:
      body = []
      head = rules[rule]['consequent']
      for j in range(len(rules[rule]['antecedents'])):
        body.append(rules[rule]['antecedents'][j])

      for j in range(len(body)):
        unifytemp, substemp = unify(query, body[j], variables)
        if unifytemp == query:
          proof.append(body[j])
          return(proof)

      if head == query:
        for j in range(len(body)):
          subproof = backward_chain(body[j], rules, variables)
          if len(subproof) != 0:
            proof.append(subproof)
        proof.append(query)
        return proof
    return None

