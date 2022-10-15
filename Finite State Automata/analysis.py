import pandas as pd
import matplotlib.pyplot as plt
def analysis1(df): # analyze what percentage of DFA contained infinite languages based on states.
    d = {}
    states = list(df.num_states.unique())
    print(states)
    #del states[1]

    for state in states:
        s = df.loc[df['num_states'] == state] # all rows containing the given state
        #print(s)
        finite = s.loc[s['is_Infinite'] == False]
        #print(finite)
        infinite = s.loc[s['is_Infinite'] == True]
        num_finite = finite[finite.columns[0]].count()
        num_infinite = infinite[infinite.columns[0]].count()

        print(f"Percentage of Infinite Langauges for state {state}: {num_infinite / (num_infinite + num_finite) * 100}%")
        print(f"Percentage of finite Langauges for state {state}: {num_finite / (num_infinite + num_finite) * 100}%")
        print("-----------------------------------------------------------------------------------------")
        d[state] = (num_infinite / (num_infinite + num_finite) * 100, num_finite / (num_infinite + num_finite) * 100)

    
    return d # key: state, value: (num_infinite, num_finite)

def analysis2(df): # analyze what percentage of DFA contained ininite languages based on length of language
    d = {}
    alpha_lengths = list(df.length_alphabet.unique())
    print(alpha_lengths)

    for length in alpha_lengths:
        l = df.loc[df['length_alphabet'] == length]
        finite = l.loc[l['is_Infinite'] == False]
        #print(finite)
        infinite = l.loc[l['is_Infinite'] == True]
        num_finite = finite[finite.columns[0]].count()
        num_infinite = infinite[infinite.columns[0]].count()

        print(f"Percentage of Infinite Langauges for alphabet length {length}: {num_infinite / (num_infinite + num_finite) * 100}%")
        print(f"Percentage of finite Langauges for alphabet length {length}: {num_finite / (num_infinite + num_finite) * 100}%")
        print("-----------------------------------------------------------------------------------------")
        d[length] = (num_infinite / (num_infinite + num_finite) * 100, num_finite / (num_infinite + num_finite) * 100)
    return d


def analysis3(df): 
    d = {}
    states = list(df.num_states.unique())
    alpha_lengths = list(df.length_alphabet.unique())
    print(states)
    print(alpha_lengths)
    for state in states:
        num_inf = []
        for length in alpha_lengths:
            l = df[(df['length_alphabet'] == length) & (df['num_states'] == state)]


            finite = l.loc[l['is_Infinite'] == False]
        #print(finite)
            infinite = l.loc[l['is_Infinite'] == True]
            num_finite = finite[finite.columns[0]].count()
            num_infinite = infinite[infinite.columns[0]].count()
            
            print(f"Percentage of Infinite Langauges for alphabet length {length} and state {state}: {num_infinite / (num_infinite + num_finite) * 100}%")
            print(f"Percentage of finite Langauges for alphabet length {length} and state {state}: {num_finite / (num_infinite + num_finite) * 100}%")
            

            num_inf.append((num_infinite / (num_infinite + num_finite) * 100, num_finite / (num_infinite + num_finite) * 100)) #num_inf, num_finite
        
        print("-----------------------------------------------------------------------------------------")
        d[state] = num_inf
    
    return d

def visualize2(d, df, xlabel, ylabel):
    states = list(df.num_states.unique())
    lengths = [2,4,6,8,10]
    for state in states:
        L = d[state]
        inf = []
        fin = []
        for tup in L:
            inf.append(tup[0])
            fin.append(tup[1])
        
        t = 'Percentage of Infinite Languages vs Finite Languages for State ' + str(state)
        plt.title(t, fontsize=10)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(lengths, inf, label = "Infinite Languages (%)")
        plt.plot(lengths, fin, label = "Finite Languages (%)")
        plt.legend()
        
        fp = 'results/states_' + str(state) +  '.png'
        plt.savefig(fp)
        plt.clf()

def visualize(d, df, title, xlabel, ylabel, filepath, type=1):
    if type == 1:
        states = list(df.num_states.unique()) # independent
    else: 
        states = list(df.length_alphabet.unique())
    infinite = []
    finite = []
    for state in states:
        infinite.append(d[state][0])
        finite.append(d[state][1])
    
    plt.title(title, fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(states, infinite, label = "Infinite Languages (%)")
    plt.plot(states, finite, label = "Finite Languages (%)")
    plt.legend()
    plt.savefig(filepath)
    plt.clf()

       


if __name__ == '__main__':
    df = pd.read_csv('results/results.csv')
    d = analysis1(df)
    l = analysis2(df)
    s = analysis3(df)
    visualize2(s, df, xlabel='Alphabet Length', ylabel='Percentage')
    visualize(d, df, 'Percentage of Infinite Languages vs Finite Languages for Arbitrary States', 'States', 'Percentage', 'results/states.png')
    visualize(l, df, 'Percentage of Infinite Languages vs Finite Languages for Arbitrary Alphabet Length', 'Alphabet Lengths', 'Percentage', 'results/alphabet_length.png',2)
    # Understand second graph more by analyzing random DFA Transition structures -> fix state and vary alphabet length and see the results.
