import numpy as np
import matplotlib.pyplot as plt

from sympy import isprime

from matplotlib import pyplot as plt


def figure_features(tex=True, font="serif", dpi=300):
    """Customize figure settings.

    Args:
        tex (bool, optional): use LaTeX. Defaults to True.
        font (str, optional): font type. Defaults to "serif".
        dpi (int, optional): dots per inch. Defaults to 180.
    """
    plt.rcParams.update(
        {   
            "font.size": 25,
            "font.family": font,
            "text.usetex": tex,
            "figure.subplot.top": 0.9,
            "figure.subplot.right": 0.9,
            "figure.subplot.left": 0.15,
            "figure.subplot.bottom": 0.12,
            "figure.subplot.hspace": 0.4,
            "savefig.dpi": dpi,
            "savefig.format": "pdf",
            "axes.titlesize": 27,
            "axes.labelsize": 20,
            "axes.axisbelow": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "xtick.minor.size": 2.25,
            "xtick.major.pad": 7.5,
            "xtick.minor.pad": 7.5,
            "ytick.major.pad": 7.5,
            "ytick.minor.pad": 7.5,
            "ytick.major.size": 5,
            "ytick.minor.size": 2.25,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 13,
            "legend.framealpha": 1,
            "figure.titlesize": 25,
            "lines.linewidth": 1,

            "text.latex.preamble": r"\usepackage{amsmath}" + "\n"
                                   r"\usepackage{physics}"
        }
    )


def plot_complexity(Numbers:np.ndarray, N_simulations:int, path:str='Results/'): 
    '''
    This function plots the results of the simulations.

    Parameters
    ----------
    Numbers : np.ndarray
        Array with the numbers to be written as the sum of two primes.
    N_simulations : int
        Number of simulations.
    path : str, optional
        Path to the results. The default is 'Results/'.
    '''
    n_simulations = np.arange(1, N_simulations + 1)

    results = np.zeros((len(Numbers), N_simulations, 2))
    quantum_complexity = np.zeros((len(Numbers), N_simulations, 2))
    problem_size = []

    for n in Numbers:
        # Calculate the problem size, it is the number of primes between 1 and n
        list_primes = [1] + [n if isprime(n) else None for n in range(2, n+1)]
        list_primes = [n for n in list_primes if n is not None]
        problem_size.append(len(list_primes))

    for n in n_simulations:
        # Load the results
        with open(path + 'results_' + str(n) + '.npy', 'rb') as file:
            res = np.load(file)
            quantum_complexity[:, n-1, :] = res[:, 2:]  # The first two columns are the results n = p + q
            results[:, n-1, :] = res[:, :2]             # The last two columns are the quantum complexity (additions and searchs)

    problem_size = np.array(problem_size)   # Problem sizes
    N_size = np.unique(problem_size)        # Unique problem sizes

    # Check if all the results are correct
    results_sums = np.mean( np.sum(results, axis=2), axis=1 )
    print(f'All the resuls are correct: {np.all(results_sums == Numbers)}')

    QC_additions = np.zeros((len(N_size), 2))
    QC_searchs = np.zeros((len(N_size), 2))

    aux_values = None
    count_index = 0

    for i, res_n in enumerate(quantum_complexity):
        # Calculate the mean and std of the quantum complexity
        # for each problem size
        n_size = N_size[count_index]
        if i == 0:
            aux_values = list(res_n)
            continue

        if i != len(quantum_complexity)-1 and problem_size[i] == n_size:
            aux_values += list(res_n)
            continue
        
        aux_values = np.array(aux_values)
        QC_additions[count_index, :] = np.array([np.mean(aux_values[:, 0]), np.std(aux_values[:, 0])])
        QC_searchs[count_index, :] = np.array([np.mean(aux_values[:, 1]), np.std(aux_values[:, 1])])
        aux_values = list(res_n)
        count_index += 1

    QC_additions = np.array(QC_additions)
    QC_searchs = np.array(QC_searchs)
    QC_complexity = [QC_additions, QC_searchs]

    classical_complexity = N_size ** 2
    quantum_complexity_theory = N_size

    ''' ------------------- Plot the results ------------------- '''

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.plot(N_size, classical_complexity, label='$\mathcal{O}(N^2)$', color='red')
    ax.plot(N_size, quantum_complexity_theory, label='$\mathcal{O}(N)$', color='blue')

    labels = ['Additions', 'Searchs']
    colors = ['orange', 'green']
    markers = ['o', 's']
    
    for i, QC in enumerate(QC_complexity):
        #for n in range(len(Numbers)):
        #    ax.scatter([problem_size[n]] * N_simulations + np.random.normal(-0.1, 0.1, N_simulations), quantum_complexity[n, :, i], color='gray', alpha=0.5, label=f'Data Points ({labels[i]})', marker=markers[i])
        ax.errorbar(N_size, QC[:, 0], yerr=QC[:, 1], label=labels[i], fmt='o', capsize=5, color=colors[i])
    
    ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=20)
    ax.set_xlabel('$N$', fontsize=30)

    fig.tight_layout()
    fig.savefig(path + f'Results.pdf', bbox_inches='tight')
    
if __name__ == '__main__':
    N_simulations = 10
    Numbers = np.arange(2, 102, 2)

    figure_features()
    plot_complexity(Numbers, N_simulations)
    plt.show()

