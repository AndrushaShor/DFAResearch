from Transition_Structure import *
import numpy
import string
import csv
import gc
from random import choice


MAX_LENGTH = 26
def experiments(transition_size): # let length vary 
    data = []
    # size of transition structure
    for trial in range(100):
            
        lengths = [2,4,6,8,10]
        for length in lengths:
            alphabet = random_string(length)
            print(f"Trial: {trial}. NUMBER OF STATES: {transition_size}. LENGTH OF ALPHABET: {length}. ALPHABET: {alphabet}")
            Transition_struct = RandomTransitionStructure(transition_size, alphabet)
            Transition_struct.generateRandomTransitionStructure(trial) #TODO only for debugging purposes, remove times parameter later 
            strings, filtered, isInfinite = Transition_struct.BFSString(m=transition_size, trial=trial)
            data.append([transition_size, length, len(strings), isInfinite]) # num of states, num of strings to find, num strings found, is_infinite
            
            
        print("-----------------------------------------------------------------------------------------")
            
        del Transition_struct
        gc.collect()
    
    
    header = ['num_states', 'length_alphabet', 'num_strings_found', 'is_Infinite']
    writeToCsv('results.csv', header, data)



def random_string(length):
    stringAlpha = string.ascii_lowercase
    
    #alphabet = list(''.join(random.choice(string.ascii_lowercase) for i in range(length)))
    
    #print(alphabet)
    return random.sample(stringAlpha, length)


def writeToCsv(filename, header, data):
    with open(filename, 'a', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        writer.writerows(data)
        #writer.close()
if __name__ == '__main__':
    experiments(4)