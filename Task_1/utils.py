import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit import execute, Aer


def execute_circuit(qc:QuantumCircuit, N_shoots:int = 1000):
    '''
    This function executes a quantum circuit and returns the counts of the measurements.
    
    Parameters:
    -----------
    qc: QuantumCircuit
        The quantum circuit to be executed.
    N_shoots: int
        The number of shots to be used in the execution.

    Returns:
    --------
    counts: dict
        A dictionary containing the counts of the measurements.
    '''
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=N_shoots)
    result = job.result()
    counts = result.get_counts(qc)
    return counts


def single_qubit_state_preparation(psi_1:np.ndarray):
    '''
    Function that generates quantum circuit to prepare a given single-qubit state.

    Parameters:
    -----------
    psi_1: array
        A 1-dimensional array containing the amplitudes of the state to be prepared.
    '''
    
    theta = 2*np.arctan(np.absolute(psi_1[1])/np.absolute(psi_1[0]))
    phi = np.angle(psi_1[1]) - np.angle(psi_1[0])
    
    q = QuantumRegister(1)
    qc = QuantumCircuit(q)
    qc.ry(theta, q[0])
    qc.rz(phi, q[0])
    
    return qc

def Schmidt_decomposition_based_preparation(psi:np.ndarray, n:int, all_parts:bool=False):
    '''
    Function that implements Schmidt-decomposition-based method to
    prepare arbitrary statevectors. Description of method can be found
    in Section 19 of arXiv:1804.03719. Although method is only discussed
    for an even number of qubits in this reference, it can be easily
    adapted to an odd number of qubits as well. This function applies
    to both even and odd numbers of qubits.

    Parameters:
    -----------
    psi: array
        A 1-dimensional array containing the amplitudes of the state to be prepared.
        It contains 2^n complex numbers, where n is the number of qubits.
    n: int
        The number of qubits.
    all_parts: bool
        If True, the function returns the gates corresponding to the
        decomposition. Otherwise, it returns an unique gate.

    Returns:
    --------
    qc_gate: Quantum Gate
        The gate corresponding to the decomposition. If all_parts is True,
        it returns a tuple containing the gates corresponding to the
        decomposition. Otherwise, it returns an unique gate.

    References:
    -----------
    [1] https://arxiv.org/abs/1804.03719
    '''
    
    if n == 1:
        qc = single_qubit_state_preparation(psi)
    else:
        M = np.reshape(psi,(2**(int(n/2) + n % 2),2**(int(n/2))))
        U, s, Vdagger = np.linalg.svd(M)

        U_circ = Operator(U)
        Vconj_circ = Operator(np.transpose(Vdagger))

        if int(n/2) == 1:
            B_circ = single_qubit_state_preparation(s).to_gate()
        else:
            B_circ = Schmidt_decomposition_based_preparation(s, int(n/2))
        q = QuantumRegister(int(n))
        qc = QuantumCircuit(q)
        qc.append(B_circ, q[:int(n/2)])
        for i in range(int(n/2)):
            qc.cx(q[i],q[int(n/2)+i])

        qc.unitary(U_circ, q[int(n/2):])      # most significant qubits for U
        qc.unitary(Vconj_circ, q[:int(n/2)])  # least significant qubits for Vconj
        
    if all_parts:
        return B_circ, U_circ, Vconj_circ
    else:
        state_preparation_gate = qc.to_gate()
        state_preparation_gate.name = "State Preparation"
        return state_preparation_gate
