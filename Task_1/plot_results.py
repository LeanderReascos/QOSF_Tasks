import numpy as np
import matplotlib.pyplot as plt

from sympy import isprime

path = 'Results/'

# Load data
Numbers = np.arange(2, 100 + 2, 2)
N = 10
n_simulations = np.arange(1, N + 1, 1)

results = np.zeros((len(Numbers), N, 2))
quantum_complexity = np.zeros((len(Numbers), N, 2))
problem_size = []

for n in Numbers:
    list_primes = [1] + [n if isprime(n) else None for n in range(2, n+1)]
    list_primes = [n for n in list_primes if n is not None]
    problem_size.append(len(list_primes))

for n in n_simulations:
    with open(path + 'results_' + str(n) + '.npy', 'rb') as file:
        res = np.load(file)
        quantum_complexity[:, n-1, :] = res[:, 2:]
        results[:, n-1, :] = res[:, :2]

problem_size = np.array(problem_size)
size = np.unique(problem_size)

classical_complexity = np.array(size) ** 2
quantum_complexity_theory = np.array(size)

results_sums = np.mean( np.sum(results, axis=2), axis=1 )
print(f'All the resuls are correct: {np.all(results_sums == Numbers)}')

QC_additions = np.zeros((len(size), 2))
QC_searchs = np.zeros((len(size), 2))

aux_values = None
n_size = size[0]
count_index = 0
for i, res_n in enumerate(quantum_complexity):
    if i == 0:
        aux_values = list(res_n)
        continue
    if problem_size[i] == n_size:
        aux_values += list(res_n)
    else:
        aux_values = np.array(aux_values)
        QC_additions[count_index, :] = np.array([np.mean(aux_values[:, 0]), np.std(aux_values[:, 0])])
        QC_searchs[count_index, :] = np.array([np.mean(aux_values[:, 1]), np.std(aux_values[:, 1])])
        aux_values = list(res_n)
        count_index += 1
        n_size = size[count_index]

QC_additions = np.array(QC_additions)
QC_searchs = np.array(QC_searchs)

# Plot results

fig_sums, ax_sums = plt.subplots(1, 1, figsize=(16, 4))

ax_sums.plot(size, classical_complexity, label='Classical complexity')
ax_sums.plot(size, quantum_complexity_theory, label='Quantum complexity Theory')

for n in range(len(Numbers)):
    ax_sums.scatter([problem_size[n]] * N + np.random.normal(-0.1, 0.1, N), quantum_complexity[n, :, 0], color='gray', alpha=0.5, label='Data Points')
ax_sums.errorbar(size, QC_additions[:, 0], yerr=QC_additions[:, 1], label='Quantum complexity additions', fmt='o', capsize=10)
# Log scale
ax_sums.set_yscale('log')
handles, labels = ax_sums.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_sums.legend(by_label.values(), by_label.keys())
ax_sums.set_xlabel('Problem size')

fig_sums.savefig(path + 'sums.png')


fig_sums, ax_sums = plt.subplots(1, 1, figsize=(16, 4))

ax_sums.plot(size, classical_complexity, label='Classical complexity')
ax_sums.plot(size, quantum_complexity_theory, label='Quantum complexity Theory')

for n in range(len(Numbers)):
    ax_sums.scatter([problem_size[n]] * N + np.random.normal(-0.1, 0.1, N), quantum_complexity[n, :, 1], color='gray', alpha=0.5, label='Data Points')
ax_sums.errorbar(size, QC_searchs[:, 0], yerr=QC_searchs[:, 1], label='Quantum complexity searchs', fmt='o', capsize=10)
# Log scale
ax_sums.set_yscale('log')
# Plot unique legends
handles, labels = ax_sums.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_sums.legend(by_label.values(), by_label.keys())
ax_sums.set_xlabel('Problem size')

fig_sums.savefig(path + 'searchs.png')