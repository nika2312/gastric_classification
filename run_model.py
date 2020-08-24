from lib import models, graph, coarsening, utils
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)

def build_laplacian(k):
    fullgraph = pickle.load(open(r"C:\Users\veronica\Desktop\study\Deep Learning\Project\full_interactions_graph", 'rb'))[0:k, 0:k]
    A = csr_matrix(fullgraph).astype(np.float32)
    graphs, perm = coarsening.coarsen(A, levels=3, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]
    pickle.dump(L, open("L"+str(k), 'wb'))
    pickle.dump(perm, open("prem"+str(k), 'wb'))
    return L, perm

def get_graph_parameters(k):
    L = pickle.load(open("L"+str(k), 'rb'))
    perm = pickle.load(open("prem"+str(k), 'rb'))
    return L, perm

def get_data(k):
    all_data = pd.read_csv("all_data.csv")
    all_data = all_data.drop(["Unnamed: 0", "barcode"], axis = 1)
    #x_train, x_test = train_test_split(all_data, test_size=0.2)
    #y_train, y_test = x_train.pop("label") , x_test.pop("label")
    #return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    y = all_data.pop("label")
    return np.array(all_data)[:, 0:k], np.array(y)


def build_params():
    n_train = 245 // 2
    params = dict()
    params['dir_name']       = 'demo'
    params['num_epochs']     = 28
    params['batch_size']     = 8
    params['eval_frequency'] = 100

    # Building blocks.
    params['filter']         = 'chebyshev5'
    params['brelu']          = 'b1relu'
    params['pool']           = 'mpool1'

    # Number of classes.
    C = 4

    # Architecture.
    params['F']              = [32, 32]  # Number of graph convolutional filters.
    params['K']              = [10, 4]  # Polynomial orders.
    params['p']              = [4, 2]    # Pooling sizes.
    params['M']              = [1024, 512, C]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['regularization'] = 5e-4
    params['dropout']        = 0.95
    params['learning_rate']  = 1e-3
    params['decay_rate']     = 0.95
    params['momentum']       = 0.9
    params['decay_steps']    = n_train / params['batch_size']
    return params

run_model = True

if run_model:
    k=2000
    if k in [200, 600, 2000]:
        L, perm = get_graph_parameters(k)
    else:
        L, perm = build_laplacian(k)
    x, y = get_data(k)
    kf = KFold(n_splits=5, random_state=0)
    final_accuracy = 0
    train_accuracy = 0
    all_loss, all_acc = 0, 0
    for train_index, test_index in kf.split(x):
        x_train, y_train, x_test, y_test = x[train_index], y[train_index], x[test_index], y[test_index]
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_train = coarsening.perm_data(x_train, perm)
        x_test = coarsening.perm_data(x_test, perm)
        model = models.cgcnn(L, **build_params())
        accuracy, loss, t_step = model.fit(x_train, y_train, x_test, y_test)
        # all_acc += accuracy[-1]
        # all_loss += loss[-1]
        final_accuracy += accuracy[-1]
        train_accuracy += model.evaluate(x_train, y_train)[1]
    final_accuracy /= 5
    train_accuracy /= 5
    all_loss /= 5
    all_acc /= 5
    print("final accuracy:", final_accuracy)
    print("training accuracy:", train_accuracy)
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(accuracy, 'b.-')
    ax1.set_ylabel('validation accuracy', color='b')
    ax2 = ax1.twinx()
    ax2.plot(loss, 'g.-')
    ax2.set_ylabel('training loss', color='g')
    plt.savefig('plot.png')


