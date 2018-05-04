# Software-Project-Economics-wrt-Cost-Effort-Estimation
The aim of this assignment was to design a genetic algorithm that would evolve a function to predict the total cost/effort of a project, given various attributes about the project, and compare the accuracy of the results with existing cost/effort prediction techniques.


I used python 2.7 for this task. 


_________________Set-up_________________

The program is currently set up to run the k-folds cross-validation function for the nasa93a data, where k = 10 and generations = 500.



_________________Structure of data_________________

The data is seperated into differnt sections which are labelled. There sections are:

### The Data ###
This section is where the data is read in an manipulated. 

### Parameters ###
This section contains the parametes for each datset. There are two chunks of parameters specific to each dataset, and then under this are general parameters applicable to both. 

NOTE: the parameters are currently set to the nasa datset.

### Tree Generator and Interpreter ###

This section contains all of the functions that are relevant to constructing ,interpreting and solving the syntax trees. 

### Initial Pop ###

This function generates the initail population of the GA


### Fitness Function ###

This section contains the fitness function


### Selection ###

This section contains the selection function


### Crossover ###

This section contains two different crossover functions - "crossover_simple" and "crossover". It also contains two other functions that assist the "crossover" function.


### Mutation ###

This section contains two different mutation functions - "mutation_simple" and "mutation".


### Implementation ###

This section contains the GA function. You can uncomment the specified code at the bottom of this section to run the GA.


### K-folds ### 

This section contains the k-fols cross-validation function. 

NOTE: it is this function that is set to run by default.


### Itertion ###

This section contains the ga_iteration function, which was used to carry out different investigations of the GA's funcitonality, where the GA was required to be run repeatedly a certain amount of times. You can uncomment the specified code at the bottom of this section to run the iteration function.

