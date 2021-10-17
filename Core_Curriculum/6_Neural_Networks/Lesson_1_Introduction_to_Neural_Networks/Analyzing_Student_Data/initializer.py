import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data preb
def data_preb(file_name):
    data = pd.read_csv(file_name)
    # print(data.head())

    return data

# ploting data
def plot_points(data):
    X = np.array(data[['gre', 'gpa']])
    y = np.array(data['admit'])

    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]

    plt.scatter(
        [s[0][0] for s in rejected], [s[0][1] for s in rejected], color='red', 
        edgecolors='k'
        )
    plt.scatter(
        [s[0][0] for s in admitted], [s[0][1] for s in admitted], color='cyan',  
        edgecolors='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')

file = 'Lesson_1_Introduction_to_Neural_Networks/Analyzing_Student_Data/student_data.csv'
data = data_preb(file)

if __name__ == '__main__':
    # plot data
    plot_points(data)

    # make data separable. 
    # we'll take the rank into account with making 4 plots, each one for each rank.
    data_rank_1 = data[data['rank'] == 1]
    data_rank_2 = data[data['rank'] == 2]
    data_rank_3 = data[data['rank'] == 3]
    data_rank_4 = data[data['rank'] == 4]

    fig = plt.figure(figsize=[20, 20])
    plt.subplot(2, 2, 1)
    plot_points(data_rank_1)
    plt.title("Rank 1", loc='left')
    plt.subplot(2, 2, 2)
    plot_points(data_rank_2)
    plt.title("Rank 2", loc='left')
    plt.subplot(2, 2, 3)
    plot_points(data_rank_3)
    plt.title("Rank 3", loc='left')
    plt.subplot(2, 2, 4)
    plot_points(data_rank_4)
    plt.title("Rank 4", loc='left')

    plt.show()
