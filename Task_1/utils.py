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
    

import time
import logging
from pathlib import Path


def prepare_message(method):
    def wrapper(ref, *messages):
        message = ''
        if len(messages) == 0:
            method(ref, message)
        for m in messages:
            message += str(m) + ' '
        method(ref, message)
    return wrapper

class CustomLoggerFilter(logging.Filter):
    def __init__(self, logger_name, name: str = "") -> None:
        super().__init__(name)
        self.logger_name = logger_name

    def filter(self, record):
        return record.name == self.logger_name

class log:
    def __init__(self, program, title, level=logging.INFO, flush: bool = False):
        Path(program).parent.mkdir(parents=True, exist_ok=True)

        self.program = program
        self.title = title
        self.level = level
        self.flush = flush
        logging.basicConfig(filename=program+'.log',
                            filemode='w',
#                            encoding='utf-8',
                            format='%(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=level)
        self.logger = logging.getLogger(program)
        self.filter = CustomLoggerFilter(self.logger.name)

        self.STARTTIME = time.time()

        self.logger.addFilter(self.filter)

        ch = logging.StreamHandler()
        ch.addFilter(self.filter)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

    @prepare_message
    def debug(self, message):
        if self.flush:
            print(message, flush=True)
        self.logger.debug(message)

    @prepare_message
    def info(self, message):
        if self.flush:
            print(message, flush=True)
        self.logger.info(message)

    @prepare_message
    def error(self, message):
        if self.flush:
            print(message, flush=True)
        self.logger.error(message)

    @prepare_message
    def warning(self, message):
        if self.flush:
            print(message, flush=True)
        self.logger.warning(message)
    
    def footer(self):
        time_sec = time.time() - self.STARTTIME
        time_hour = time_sec//3600
        time_min = (time_sec - time_hour*3600)//60
        time_sec = time_sec - time_hour*3600 - time_min*60

        self.info(f'Program {self.program} finished in {time_hour:.0f}h {time_min:.0f}m {time_sec:.0f}s')

import os
from multiprocessing import Process, Manager
from functools import partial
from typing import Callable, Union

def parallelize(process_name: str, f: Callable, iterator: Union[list, np.ndarray], *args, n_process=os.cpu_count(), logger=None) -> np.ndarray:
        '''
        Create processes for some function f over an iterator.
        Parameters
            process_name : string
                The name of the process to be parallelized.
            f : Callable
                Function f(iterator: array_like, *args) to be applied.
                This function gets another iterator as a parameter and will compute the result for each element in the iterator.
            iterator : array_like
                The function f is applied to elements in the iterator array.
            verbose : bool
                If this flag is true the progress bar is shown.
        
        Return
            result : array_like
                It contains the result for each value inside the iterator.
                The information is not sorted.
        '''
        process = []
        iterator = list(iterator)
        N = len(iterator)

        ###########################################################################
        # Debug information and Progress bar
        ###########################################################################

        logger = log('parallelize', process_name, level=logging.INFO) if logger is None else logger
        logger.info(f'Starting Parallelization for {process_name} with {N} values. Number of processes: {n_process}')

        ###########################################################################
        # Processes management
        ###########################################################################
        def parallel_f(result: np.ndarray, per: list[int], iterator: Union[list, np.ndarray], *args) -> None:
            '''
            Auxiliar function to help the parallelization
            Parameters:
                result : array_like
                    It is a shared memory list where each result is stored.
                per : list[int]
                    It is a shared memory list that contais the number of elements solved.
                iterator : array_like
                    The function f is applied to elements in the iterator array.
            '''
            value = f(iterator, *args)              # The function f is applied to the iterator
            if value is not None:
                # The function may not return anything
                result += value        # Store the output into result array
            per[0] += len(iterator)                 # The counter is actualized
            
            logger.debug(f'Process {process_name} finished {per[0]} of {N} values.')
        
        result = Manager().list([])             # Shared Memory list to store the result
        per = Manager().list([0])               # Shared Memory to countability the progress
        f_ = partial(parallel_f,  result, per)  # Modified function used to create processes

        n = N//n_process                                                   # Number or processes
        for i_start in range(n_process):
            # Division of the iterator array into n smaller arrays
            j_end = n*(i_start+1) if i_start < n_process-1\
                else n*(i_start+1) + N % n_process
            i_start = i_start*n
            p = Process(target=f_, args=(iterator[i_start: j_end], *args))      # Process creation
            p.start()                                                           # Initialize the process
            process.append(p)

        while len(process) > 0:
            p = process.pop(0)
            p.join()
        return np.array(result)