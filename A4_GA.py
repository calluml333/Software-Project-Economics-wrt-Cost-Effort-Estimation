import random
import arff
from operator import itemgetter


"""
Author: Callum Little
Class: CS547
Assignment: 4
"""

#===============================================================================
"""=========================================================================="""
#================================ The Data =====================================


#### NASA DATA ####

Nasa_data_dict = arff.load(open('nasa93-dem.arff', 'rb'))
# source: https://pypi.python.org/pypi/liac-arff
Nasa_data_list = Nasa_data_dict.items()
Nasa_data = Nasa_data_list[3][1]  # found this by looking at the data

#for dataset in data:
#    print(len(dataset)) # This is just to check each line is the same length

new_nasa = []            # this is to convert non-numerical nasa data to numerical
for row in Nasa_data:
    new_row = []
    for data in row:
        if data == u'vl':
            new_row.append(1.0)
        elif data == u'l':
            new_row.append(2.0)
        elif data == u'n':
            new_row.append(3.0)
        elif data == u'h':
            new_row.append(4.0)
        elif data == u'vh':
            new_row.append(5.0)
        elif data == u'xh':
            new_row.append(6.0)
        else:
            new_row.append(data)
    new_nasa.append(new_row)


#### Nasa93a
Nasa_attributes_a = []   #This will sort the data in to attributes and targets
Nasa_targets_a = []
for dataset in Nasa_data:
    sorted_list = []
    unsorted = dataset[-4:]
    sorted_list.extend([unsorted[0]]) # Extract relevant data
    sorted_list.extend(unsorted[2:])
    Nasa_attributes_a.append(sorted_list)
    Nasa_targets_a.append(unsorted[1])

### Nasa93b
Nasa_attributes_b = []
Nasa_targets_b = []
for dataset in new_nasa:
    sorted_list = []
    sorted_list.extend(dataset[1:24]) 
    sorted_list.extend(dataset[-2:])
    Nasa_attributes_b.append(sorted_list)
    Nasa_targets_b.append(dataset[-3])



#### China DATA ####

China_data_dict = arff.load(open('china.arff', 'rb'))
# source: https://pypi.python.org/pypi/liac-arff
China_data_list = China_data_dict.items()
China_data = China_data_list[3][1]  # found this by looking at the data
    
China_attributes = []
China_targets = []
for dataset in China_data:
    China_attributes.append(dataset[1:-2])
    China_targets.extend(dataset[-1:])


#===============================================================================
"""=========================================================================="""
#================================ Parameters ===================================


function_set = ['*', '/', '+', '-'] 
# If the function set is altered make sure to amend the interpreter function's
# solution section. 

Nasa_inputs_a = ['x', 'y', 'z']  #for the smaller nasa set
Nasa_inputs_b = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
China_inputs = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']
# Arbitrarily choses letters, the only important thing is that the number of
# inputs is equal to the number of attributes in the data. For Nasa data, there
# are three attributes.



### Data specific Parameters ###

attributes = Nasa_attributes_a
targets = Nasa_targets_a
inputs = Nasa_inputs_a   # Select the correct inputs for the data from the 2 above 
tree_depth = 2      # Depth of the initial tree population
max_tree_depth = 3  # Maximum depth any tree can reach
k = 10              # Number of k-folds
k_sub = 10          # Number of elements within each k-fold

#attributes = China_attributes
#targets = China_targets
#inputs = China_inputs
#tree_depth = 3      
#max_tree_depth = 4  
#k = 10          
#k_sub = 50          



### General Parameters ###

pop_size = 100
crossover_rate = 0.75
mutation_rate = 0.05
limit = 200         # Largest random number that an equation is allowed to include
generations = 500
selection_size = 0.5*pop_size
tree_type = "grow"  # Can be either "grow" or "full"
#tree_type = "full"


iterations = 10  # for ga_iteration function


#===============================================================================
"""=========================================================================="""
#====================== Tree Generator and Interpreter =========================


def terminal_branch(function_set, ext_inputs, limit):
    """
    Creates a terminal node.
    """
    node = []
    node.append(random.choice(function_set))
        
    choice_1 = random.randint(0,1)
    if choice_1 == 0:  # if it swings this way, add in an external input
        node.append(random.choice(ext_inputs))
    else:              # otherwise add in a float
        node.append(float(random.randint(0,limit)))

    choice_2 = random.randint(0,1) 
    if choice_2 == 0:  
        node.append(random.choice(ext_inputs))
    else:
        node.append(float(random.randint(0,limit))) 
    return node 
            

def syntax_tree(function_set, ext_inputs, tree_depth, limit):
    """
    Creates a syntax tree of depth "tree_depth", of type grow initialisation. 
    """

    tree = []
    tree.append(random.choice(function_set))
    i = 0
    previous_branch_depth = tree_depth-1 # This counter forces at least one branch reach the maximum depth 
    
    while i < 2:
        if previous_branch_depth != tree_depth-1:
            branch_depth = tree_depth-1
        else:
            branch_depth = random.randint(0, tree_depth-1) # use tree_depth-1 to account for root node        
        
        if branch_depth == 0:
            """
            This will just add in a value as the terminal.
            """
            if random.randint(0,1) == 0:
                tree.append(random.choice(ext_inputs)) #choose an external input
            else:
                tree.append(float(random.randint(0, limit)))  #choose a random number in the limit            
        
        elif branch_depth == 1:
            """
            This will add in a full terminal node. 
            """
            terminal = terminal_branch(function_set, ext_inputs, limit)
            tree.append(terminal)                       
        
        else:
            """
            This will create another tree.
            """
            branch_tree = syntax_tree(function_set, ext_inputs, branch_depth, limit)
            tree.append(branch_tree)  
            
        previous_branch_depth = branch_depth
        i += 1 
   
    return tree


def syntax_tree_full(function_set, ext_inputs, tree_depth, limit):
    """
    Creates a syntax tree of depth "tree_depth", of type full initialisation. 
    """

    tree = []
    tree.append(random.choice(function_set))
    i = 0
    while i < 2:
        branch_depth = tree_depth-1 # use tree_depth-1 to account for root node
        
        if branch_depth == 0:
            """
            This will just add in a value as the terminal.
            """
            if random.randint(0,1) == 0:
                tree.append(random.choice(ext_inputs)) #choose an external input
            else:
                tree.append(float(random.randint(0, limit)))  #choose a random number in the limit
            
        elif branch_depth == 1:
            """
            This will add in a full terminal node. 
            """
            terminal = terminal_branch(function_set, ext_inputs, limit)
            tree.append(terminal)
            
        else:
            """
            This will create another tree.
            """
            branch_tree = syntax_tree_full(function_set, ext_inputs, branch_depth, limit)
            tree.append(branch_tree)        
        i += 1 
   
    return tree


def syntax_tree_interpreter(syntax_tree, function_set, att_dict):
    """
    Takes in an equation tree and then solves it. The only operators that it
    accepts are +, -, / and *, however this can be changed. The only condition
    is that you know what operators it should be using to allow you to code it 
    in manually in the "Solution" section.
    """
    
    i = 0
    simplified_equation = ['','',''] # To progress to the solution
    ordered_equation = ['','','']    # To record the full equation
    while i < len(syntax_tree):
        equations = syntax_tree[i]
        if type(equations) == int or type(equations) == float:   #if it's an integer
            if simplified_equation[0] == '':

                simplified_equation[0] = equations
                ordered_equation[0] = equations
            else:
                simplified_equation[2] = equations
                ordered_equation[2] = equations
            """
            This will make sure that the values are added to the correct side of
            the operator in the list.
            """            
        
        elif type(equations) == tuple or type(equations) == list:
            # if its' a sub-equation, fire it back into this equation
            sub_eq = syntax_tree_interpreter(equations, function_set, att_dict)
            if simplified_equation[0] == '':

                simplified_equation[0] = sub_eq[1]
                ordered_equation[0] = sub_eq[2]
            else:
                simplified_equation[2] = sub_eq[1]
                ordered_equation[2] = sub_eq[2]
            """
            This makes sure that the sub-equations are added to the correct side
            of the operator in the list.
            """ 
                        
        else:
            if equations in function_set: # if it's an operator string
                """
                This adds the operator to the middle of the list.
                """
                simplified_equation[1] = equations
                ordered_equation[1] = equations 
            else:   # if its an external input string
                if simplified_equation[0] == '':
    
                    simplified_equation[0] = equations
                    ordered_equation[0] = equations
                else:
                    simplified_equation[2] = equations
                    ordered_equation[2] = equations                     
        i += 1
           
    # External input section
    """
    This section looks at the stings which represent the attributes, for 
    example 'x', 'y', ...
    It then determines what values these represent using the "att_dict" 
    dictionary, which contains this information. Fro example:
        att_dict = {'x': 1, 'y': 2, ... } 
    """
    if type(simplified_equation[0]) == str and len(simplified_equation[0]) == 1:
        # This will be an external input
        simplified_equation[0] = att_dict[simplified_equation[0]]
        
    if type(simplified_equation[2]) == str and len(simplified_equation[2]) == 1:
        simplified_equation[2] = att_dict[simplified_equation[2]]

    # Solution Section
    """
    This section now calculates the solution of the equation. 
    """
    if type(simplified_equation[0]) == str and len(simplified_equation[0]) > 1 or type(simplified_equation[2]) == str and len(simplified_equation[2]) > 1:
        solution = "Invalid" # In case previous solutions were "Invalid" 
    elif simplified_equation[1] == '+':
        solution = simplified_equation[0] + simplified_equation[2]
    elif simplified_equation[1] == '-':
        solution = simplified_equation[0] - simplified_equation[2]
    elif simplified_equation[1] == '/':
        if simplified_equation[2] == 0.0:
            solution = "Invalid"  # If there is a devision by zero
        else:
            solution = simplified_equation[0] / simplified_equation[2]        
    elif simplified_equation[1] == '*':
        solution = simplified_equation[0] * simplified_equation[2]
                
    return [simplified_equation, solution, ordered_equation, syntax_tree]
                        # DON'T CHANGE THIS OUTPUT!!!!


def depth_calculator(syntax_tree, previous_count=None):
    if previous_count == None:
        count = 0
    else:
        count = previous_count
    if type(syntax_tree) == list or type(syntax_tree) == tuple:
        count1 = depth_calculator(syntax_tree[1], count+1) # branch 1
        count2 = depth_calculator(syntax_tree[2], count+1) # branch 2
        count = max(count1, count2)  # this finds the max depth from the two branches
    return count


#===============================================================================
"""=========================================================================="""
#=============================== Initial Pop ===================================


def initial_population(pop_size, function_set, ext_inputs, tree_depth,  tree_type, limit=None):
    """
    Initialises a population pop_size number of syntax trees of tree_type 
    "full" or "grow" and a depth of tree_depth. ext_input's are the attributes 
    of the data and limit is the maximum number that can be randomly added in to
    the trees. The function set consits of *, /, - and +. The output is a list 
    of syntax tree.
    """
    
    population = []
    if limit == None:
        limit = 100
    while len(population) < pop_size:
        if tree_type == "full":
            tree = syntax_tree_full(function_set, ext_inputs, tree_depth, limit)
        elif tree_type == "grow":
            tree = syntax_tree(function_set, ext_inputs, tree_depth, limit)
        population.append(tree)
    return population


#===============================================================================
"""=========================================================================="""
#============================= Fitness Function ================================


def fitness_function(equation_array, attribute_array, target_array, 
                     function_set, ext_inputs, n=None):
    """
    Calculates the fitness of the input syntax tree.
    """

    fit_list = []    
    for equation in equation_array:
        i = 0
        error_array = []
        while i < len(attribute_array):
            j = 0
            attributes = {}
            while j < len(ext_inputs):  # This creates the dictionary
                attributes[ext_inputs[j]] = attribute_array[i][j]
                j += 1
            target = target_array[i]
            equation_solved = syntax_tree_interpreter(equation, function_set, 
                                                      attributes)         
             
            if equation_solved[1] == "Invalid":
                error_array.append(1000000)
            else:
                error = abs(target - equation_solved[1])#/target # abs error: 0 is best
                error_array.append(error)
            i += 1
        if n == None:
            fitness = sum(error_array) # just the sum
        else:
            fitness = sum(error_array)/len(attribute_array) # takes the mean of the abs error
                                                       # Used for the crossvalidation
        
        
        fit_list.append([equation, fitness])
    
    fit_list_sorted = sorted(fit_list, key=itemgetter(1))  
    return fit_list_sorted   
            # Needs to have the equation, fitness and original representation 
            # for the crossover. 


#===============================================================================
"""=========================================================================="""
#=============================== Selection =====================================


def tournament(fit_list, select_size):
    """
    Standard tournament selection process.
    """
    
    select_pop = []
    top_10_percent = int(0.1*len(fit_list))
    for vec in fit_list[:top_10_percent]:
        select_pop.append(vec[0])         # Fittest 10% of the population    
    rest_of_pop = fit_list[top_10_percent:]
        
    while len(select_pop) < select_size:
        choice1 = random.choice(rest_of_pop)
        choice2 = random.choice(rest_of_pop)

        # Now compare fitness values of each choice and choose the fitter one
        if choice1[1] < choice2[1]:
            select_pop.append(choice1[0])
        elif choice1[1] == choice2[1]:
            choice = random.choice([choice1,choice2])
            select_pop.append(choice[0]) 
        else:
            select_pop.append(choice2[0])
               
    return select_pop   # needs to just be syntax trees


#===============================================================================
"""=========================================================================="""
#=============================== Crossover =====================================


#syntax trees go in
def crossover_simple(select_pop, new_pop_size, crossover_rate):
    """
    Simple crossover function. It chooses which branch from the first parent 
    tree that has to be removed for crossing. It then attaches the opposite
    branch to the tree. 
    """
    
    new_pop = []
    random.shuffle(select_pop)
    non_cross = int((1-crossover_rate)*new_pop_size)
    if non_cross != 0.0:
        new_pop.extend(select_pop[:non_cross])
    while len(new_pop) < new_pop_size:
        
        parent1 = random.choice(select_pop)
        parent2 = random.choice(select_pop)
        child = []
        break_position = random.randint(1,2)
        if break_position == 1:  #if we select the first branch as the cross point   
            #choose which parent's root operator to use
            if random.random() < 0.5:
                child.append(parent1[0])
            else:
                child.append(parent2[0])           
            child.append(parent2[1]) #first element from parent 2     
            child.append(parent1[2]) #last element from parent 1
            
        elif break_position == 2: #if we select the first branch as the cross point           
            #choose which parent's root operator to use   
            if random.random() < 0.5:
                child.append(parent1[0])
            else:
                child.append(parent2[0])
            child.append(parent1[1]) #first element from parent 1 
            child.append(parent2[2]) #last element from parent 2
                             
        new_pop.append(child)
    return new_pop
    

####### More Advanced Crossover #######


def cross_choice(tree):
    
    choice = random.randint(1,2)
    if type(tree[choice]) == float or type(tree[choice]) == int or type(tree[choice]) == str:
        part = tree[choice]
    elif type(tree[choice]) == list or type(tree[choice]) == tuple:
        if random.random() < 0.3:
            part = tree[choice]
        else:
            part = cross_choice(tree[choice])        
    return part


def cross_point(tree, new_part):
    
    cross_tree = tree[:]
    choice = random.randint(1,2)
    if type(cross_tree[choice]) == float or type(cross_tree[choice]) == int or type(cross_tree[choice]) == str:
        cross_tree[choice] = new_part
    elif type(cross_tree[choice]) == list or type(cross_tree[choice]) == tuple:
        if random.random() < 0.5:
            cross_tree[choice] = new_part     
        else:
            cross_tree[choice] = cross_point(cross_tree[choice], new_part)   
    return cross_tree


def crossover(select_pop, new_pop_size, crossover_rate, max_tree_depth):
    """
    Advanced crossover function. It chooses a random section of parent 2 and 
    replaces a random section of parent 1 with that part to create the 
    child. Only one child is created in each crossover operation. 
    """
    
    new_pop = []
    random.shuffle(select_pop)
    non_cross = int((1-crossover_rate)*new_pop_size)
    if non_cross != 0.0:
        new_pop.extend(select_pop[:non_cross])
    while len(new_pop) < new_pop_size:
        
        parent1 = random.choice(select_pop)
        parent2 = random.choice(select_pop)
        # First pick ot the part from parent 2 that we want to swap in to 
        # parent 1
        part = cross_choice(parent2)
        part_depth = depth_calculator(part)
        # Then replace a part of parent 1 with the part selected from 
        # parent 2
        child = cross_point(parent1, part)
        child_depth = depth_calculator(child)
        if child_depth <= max_tree_depth: # If the depth is okay
            new_pop.append(child)
    return new_pop
    
    
#===============================================================================
"""=========================================================================="""
#================================ Mutation =====================================


def mutation_simple(new_pop, mutation_rate, function_set, ext_inputs, 
             tree_depth, limit):
    """
    Changes one of the branches in the tree to eiter a number (random or input)
    or another tree.
    """
    
    random.shuffle(new_pop)
    mutate_number = int(mutation_rate*len(new_pop))
    pop_to_mutate = new_pop[:mutate_number]
    mutated_pop = new_pop[(mutate_number):]
    
    for equation in pop_to_mutate:
        choice = random.randint(0,1)
        if choice == 0:    
            #makes mutation a randomly generated "grow" tree
            mutation = syntax_tree(function_set, ext_inputs, tree_depth, limit)
            #mutation = terminal_branch(function_set, ext_inputs, limit)
        else:   
            #makes mutation either a random number or an external input
            if random.randint(0,1) == 0:
                mutation = float(random.randint(0,limit))
            else:
                mutation = random.choice(ext_inputs)
                        
        new_equation = []
        mutation_position = random.randint(1,2)
        if mutation_position == 1:           
            new_equation.append(equation[0])
            new_equation.append(mutation) # mutation replaces first branch   
            new_equation.append(equation[2]) 
            
        elif mutation_position == 2:            
            new_equation.append(equation[0])
            new_equation.append(equation[1])    
            new_equation.append(mutation) # mutation replaces second branch
                        
        mutated_pop.append(new_equation)
    random.shuffle(mutated_pop)
    return mutated_pop
   
     
def mutation(new_pop, mutation_rate, function_set, ext_inputs, 
             tree_depth, limit, max_tree_depth):
    """
    Advanced mutation function. It chooses a random part of the tree and mutates
    it, replacing it with either a random number, and input or a tree. 
    """
    
    mutate_number = int(mutation_rate*len(new_pop))
    pop_to_mutate = new_pop[:mutate_number]
    mutated_pop = new_pop[(mutate_number):]
    for equation in pop_to_mutate:
        while True:    # While the depth of the mutated element is allowed...
            choice = random.randint(0,1)
            if choice == 0:    
                #makes mutation a randomly generated "grow" tree
                new_tree_depth = random.randint(1,tree_depth)
                mutation = syntax_tree(function_set, ext_inputs, new_tree_depth, limit)
            
            else:   
                #makes mutation either a random number or an external input
                if random.randint(0,1) == 0:
                    mutation = float(random.randint(0,limit))
                else:
                    mutation = random.choice(ext_inputs)  
            mutated_equation = cross_point(equation, mutation)
            mutated_equation_depth = depth_calculator(mutated_equation)
            #Checks that the depth of the mutated equation is allowed
            if mutated_equation_depth <= max_tree_depth:
                #If so, it will add to the new population
                mutated_pop.append(mutated_equation)
                break
            
    random.shuffle(mutated_pop)
    return mutated_pop  
    
    
#===============================================================================
"""=========================================================================="""
#============================== Implementation =================================


def genetic_algorithm(pop_size, ext_inputs, attributess, 
                      tree_depth, tree_type, target_array, crossover_rate=None, 
                      mutation_rate=None, generations=None, select_size=None,
                      limit=None, max_tree_depth=None, function_set=None):
    """
    This is the GA function. The GA will run for as many gererations as stated.
    The outpur is the evolved equation and it's mean absolute error.
    """
    
    if crossover_rate == None:
        crossover_rate = 0.75
    if mutation_rate == None:
        mutation_rate = 0.1
    if generations == None:
        generations = 100
    if select_size == None:
        select_size = int(0.5*pop_size)
    if limit == None:
        limit = 100  
    if max_tree_depth == None:
        max_tree_depth = 3 
    if function_set == None:
        function_set = ['*', '/', '+', '-']   
    
    
    population = initial_population(pop_size, function_set, ext_inputs, 
                                    tree_depth,tree_type, limit=limit)    
    i = 0
    while i < generations:
        print("Generation", i+1)
        fitness = fitness_function(population, attributess, target_array, 
                  function_set, ext_inputs)
        if fitness[0] == 0:
            break
        #print(" best of Gen", i+1,  fitness[0]) 
        winners = tournament(fitness, select_size)
        crossed_pop = crossover(winners, pop_size, crossover_rate, max_tree_depth)
        population = mutation(crossed_pop, mutation_rate, function_set, 
                     ext_inputs, tree_depth, limit, max_tree_depth)
        i += 1
    fitness = fitness_function(population, attributess, target_array, 
              function_set, ext_inputs, n=1)
    return fitness[0]
    
    
### Uncomment below to run the GA

#GA = genetic_algorithm(pop_size, inputs, attributes, 
#                                     tree_depth, tree_type, targets, 
#                                     crossover_rate, mutation_rate, generations, 
#                                     selection_size, limit, max_tree_depth, 
#                                     function_set)
#print("\n \n \n", GA) 


#===============================================================================
"""=========================================================================="""
#================================== K-Folds ====================================


def k_fold(function_set, pop_size, ext_inputs, attributes_list, target_list,
           tree_depth, tree_type, crossover_rate=None, mutation_rate=None, 
           select_size=None, generations=None, limit=None, max_tree_depth=None,
           k=None, k_sub=None):
    """
    K-folds equaiton. Breaks the data up in to k sections containing k_sub 
    amount of data. It will then run the GA k times using k-1 of the data for
    traiing and the other data for testing. The output is an array of the 
    evolved equations and their mean absolute errors. 
    """
    if k == None and k_sub == None:
        k = 10
        k_sub = 10    #arbitrarily picked
    if k != None and k_sub == None:
        raise ValueError("k_sub value is required for a pre-determined k value")
        #this is so that the data is efficiently and accurately distributed       
                
    #### k-fold setup #### 
    i = 0
    k_folds_attributes = []
    k_folds_targets = []
    while i < k:
        k_folds_attributes.append([]) # creates k empty arrays
        k_folds_targets.append([]) 
        i += 1
    
    j = 0
    l = 0
    while j < k:
        count = 0
        while l < len(attributes_list) and count < k_sub:
            k_folds_attributes[j].append(attributes_list[l])
            k_folds_targets[j].append(target_list[l]) 
            #fills the k empty arrays with (roulghly) same number of elements until the limit
            count += 1
            l += 1
        j += 1
    
    
    ### Cross Validation ###
    cross_val = 0
    fitness_array = []  
    equation_array = []
    
    while cross_val < k:
        # Attributes
        train_atts_non_merged = k_folds_attributes[:cross_val] + k_folds_attributes[(cross_val+1):] #chooses k-1 groups (lists) to train from the data
        train_attributes = []
        for attribute_group in train_atts_non_merged: #makes the training data one group (list)
            train_attributes.extend(attribute_group)
        test_attributes = k_folds_attributes[cross_val]

        # Targets
        train_targets_non_merged = k_folds_targets[:cross_val] + k_folds_targets[(cross_val+1):] 
        train_targets = []
        for target in train_targets_non_merged: 
            train_targets.extend(target)
        test_train = k_folds_targets[cross_val] 
        print("\nCross Validation", cross_val+1)
        best_equation = genetic_algorithm(pop_size, ext_inputs, train_attributes, 
                                     tree_depth, tree_type, train_targets, 
                                     crossover_rate, mutation_rate, generations, 
                                     select_size, limit, max_tree_depth, 
                                     function_set)
                     
        fitness = fitness_function([best_equation[0]], test_attributes, test_train, 
                                    function_set, ext_inputs, n=1)  

        equation_array.append(fitness[0][0])
        fitness_array.append(fitness[0][1])

        cross_val += 1
  
    return [equation_array, fitness_array]
    
  

   
cross_validation = k_fold(function_set, pop_size, inputs, attributes, targets, 
                          tree_depth, tree_type, crossover_rate, mutation_rate, 
                          selection_size, generations, limit, max_tree_depth, 
                          k, k_sub) 
print("\n \n \n", cross_validation) 
         
 
#===============================================================================
"""=========================================================================="""
#================================ Iteration ====================================
                   
                                                    
def ga_iteration(iteration_limit, pop_size, inputs, attributes, tree_depth, 
                 tree_type, targets, crossover_rate, mutation_rate, generations, 
                 selection_size, limit, max_tree_depth, function_set):
    """
    This function was used for various investigations on the GA, such as how 
    the depth of the tree's affect the accuracy of the evolved equations...
    """
    i = 0
    fitness_array = []
    depth = max_tree_depth
    fittest_depth = []
    while i < iteration_limit:
        #if depth == 2:
        #    tree_depth = 2
        #else:
        #    tree_depth = 3
        equations = genetic_algorithm(pop_size, inputs, attributes, 
                                     tree_depth, tree_type, targets, 
                                     crossover_rate, mutation_rate, generations, 
                                     selection_size, limit, depth, 
                                     function_set)
        fitness_array.append(equations[1])
        fittest_depth.append(depth_calculator(equations[0]))
        
        #depth += 1   # for investigating how depth affects accuracy
        print(fitness_array, fittest_depth)
        i += 1
    return [fitness_array, fittest_depth]
          
             
## Uncomment below to run the Iterations              
                   
#Investigations = ga_iteration(iterations, pop_size, inputs, attributes, 
#                                     tree_depth, tree_type, targets, 
#                                     crossover_rate, mutation_rate, generations, 
#                                     selection_size, limit, max_tree_depth, 
#                                     function_set)                                              
#print("\n \n \n", Investigations) 















