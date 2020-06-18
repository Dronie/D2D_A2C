import numpy as np
import rand_prob as prob
import math

n = 3
no_of_trials = 500000
trials = []
len_c = 5

# compute the actual prob using formula ---------------------------------------------------------------
form = []
for i in range(0, n):
    form.append( math.log( (len_c-i)/len_c ) )

prob = math.pow( math.e, sum(form) )
print("actual probability of all agents taking unique actions: ", prob)


# run a number of trials ---------------------------------------------------------------------
for i in range(0, no_of_trials):
    #if i % 1000 == 0:
        #print("Trials ", (i / no_of_trials) * 100, " percent complete")
    c = [0,0,0,0,0]
    for j in range(0, n):
        choice = np.random.randint(0,len(c))

        c[choice] = 1
    
    trials.append(c)

print("example trial: ", trials[np.random.randint(0, 100000)])


# tally the trials -------------------------------------------------------------------------------------
entries = []
count = []

for i in range(0, len(trials)):
    #if i % 1000 == 0:
        #print("Checks ", (i / no_of_trials) * 100, " percent complete")
    if trials[i] not in entries:
        entries.append(trials[i])
        count.append(1)
    else:
        ind = entries.index(trials[i])
        count[ind] += 1

# count the number of examples we're looking for ----------------------------------------------------------
unique_count = 0

for i in range(0, len(entries)):
    if sum(entries[i]) == 1:
        unique_count += count[i]

print("estimated prob of all agents taking unique actions: ", (unique_count / len(trials)))
