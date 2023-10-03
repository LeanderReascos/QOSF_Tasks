import numpy as np
from sympy import isprime
import logging
import os

from qiskit.extensions import ZGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from utils import execute_circuit, Schmidt_decomposition_based_preparation
from utils import log, parallelize

def quantum_adder(n_bits:int):
    '''
    This function creates a quantum circuit that adds two n-qbit binary numbers.
    
    Parameters:
    ----------
    n_bits: int
        The number of bits in the binary numbers.
    
    Returns:
    -------
    qc: Quantum Gate
        The quantum circuit in a gate format that adds two n-bit binary numbers.
    '''
    q_r_numberA = QuantumRegister(n_bits, name='A')
    q_r_numberB = QuantumRegister(n_bits, name='B')
    q_r_carry = QuantumRegister(n_bits, name='Carry')

    qc = QuantumCircuit(q_r_numberA, q_r_numberB, q_r_carry, name='Quantum_Adder')

    # Add both registers
    for i in range(n_bits):
        qc.ccx(q_r_numberA[i], q_r_numberB[i], q_r_carry[i])
        qc.cx(q_r_numberA[i], q_r_numberB[i])
        if i > 0:
            qc.ccx(q_r_numberB[i], q_r_carry[i-1], q_r_carry[i])
            qc.cx(q_r_carry[i-1], q_r_numberB[i])
    
    # The result is encoded in the last carry qubit (the most significant bit) and the B register

    adder_gate = qc.to_gate()
    adder_gate.name = 'Quantum_Adder'
    return adder_gate

def A(input_state_gate, n_qubits:int):
    '''
    This function creates the quantum operator A that prepares the initial superposition state for Grover's algorithm.
    The superposition state is a sum of all the possible combinations of two n-qbit binary numbers. The sum is performed
    using a quantum adder circuit.

    Parameters:
    ----------
    input_state_gate: Quantum Gate
        The gate that prepares the initial state for each register. This corresponds to an equal superposition of all
        the numbers to be added.
    n_qubits: int
        The number of qubits in each register.

    Returns:
    -------
    A_gate: Quantum Gate
        The gate that prepares the initial superposition state for Grover's algorithm.
    '''
    q_r_numberA = QuantumRegister(n_qubits, name='A')           # Register for the first summand
    q_r_numberB = QuantumRegister(n_qubits, name='B')           # Register for the second summand
    q_r_carry = QuantumRegister(n_qubits, name='Carry')         # Register for the carry bits

    qc = QuantumCircuit(q_r_numberA, q_r_numberB, q_r_carry)

    qc.append(input_state_gate, q_r_numberA[:])
    qc.append(input_state_gate, q_r_numberB[:])
    qc.append(quantum_adder(n_qubits), q_r_numberA[:] + q_r_numberB[:] + q_r_carry[:])

    A_gate = qc.to_gate()
    A_gate.name = '$A$'
    return A_gate


def S_f(number:int, num_qubits:int):
    '''
    This function creates the oracle gate S_f that marks the desired solution in Grover's algorithm.
    The oracle in this case marks the state corresponding to the sum of the two numbers equal to the
    desired number.

    Parameters:
    ----------
    number: int
        The desired number to be found.
    num_qubits: int
        The number of qubits in each register.

    Returns:
    -------
    S_f_gate: Quantum Gate
        The oracle gate that marks the desired solution in Grover's algorithm.
    '''
    q_r = QuantumRegister(num_qubits, name='q_r')
    qc = QuantumCircuit(q_r, name='$S_f$')
    
    bit_string = np.binary_repr(number, width=num_qubits)

    if bit_string[0] == '0':
        # Apply the X gate to the last qubit if the first bit of the binary representation of the number is 0
        qc.x(q_r[-1])

    # Apply the Control Z gate to the last qubit
    qc.append(ZGate().control(len(bit_string)-1, ctrl_state=bit_string[1:]), q_r)

    if bit_string[0] == '0':
        # Apply the X gate to the last qubit if the first bit of the binary representation of the number is 0
        qc.x(q_r[-1])

    S_f_gate = qc.to_gate()
    S_f_gate.name = '$S_f$'

    return S_f_gate

def Grover(A_gate, S_f_gate, n_qubits:int):
    '''
    This function creates the Grover gate that implements Grover's algorithm.

                        $$ G = A S_0 A^{\dagger} S_f $$

    Parameters:
    ----------
    A_gate: Quantum Gate
        The gate that prepares the initial superposition state for Grover's algorithm.
    S_f_gate: Quantum Gate
        The oracle gate that marks the desired solution in Grover's algorithm.
    n_qubits: int
        The number of qubits in each register.

    Returns:
    -------
    Grover_gate: Quantum Gate
        The gate that implements a Grover's algorithm iteration.
    '''
    q_r_numberA = QuantumRegister(n_qubits, name='A')       # Register for the first summand
    q_r_numberB = QuantumRegister(n_qubits, name='B')       # Register for the second summand
    q_r_carry = QuantumRegister(n_qubits, name='Carry')     # Register for the carry bits

    qc = QuantumCircuit(q_r_numberA, q_r_numberB, q_r_carry)

    q_r_number = q_r_numberB[:] + [q_r_carry[-1]]           # Register for the sum of the two numbers

    qc.append(S_f_gate, q_r_number)
    qc.append(A_gate.inverse(), q_r_numberA[:] + q_r_numberB[:] + q_r_carry[:])
    # ---------------- S_0 ------------------- #
    [qc.x(q) for q in q_r_number]
    qc.append(ZGate().control(len(q_r_number)-1), q_r_number)
    [qc.x(q) for q in q_r_number]
    # ---------------------------------------- #
    qc.append(A_gate, q_r_numberA[:] + q_r_numberB[:] + q_r_carry[:])

    Grover_gate = qc.to_gate()
    Grover_gate.name = 'Grover'

    return Grover_gate


def find_the_primes_numbers(number_1:int, list_primes:list[int], max_tries:int=10, N_shoots:int=5, max_iterations:int=5, logger=None):
    '''
    This function finds the two prime numbers that make up the number_1. (Goldbach's conjecture)

    Parameters:
    ----------
    number_1 : int
        The number to be decomposed into prime numbers.
    list_primes : list[int]
        List of prime numbers.
    max_tries : int (default=10)
        Maximum number of tries to find the results.
    N_shoots : int (default=5)
        Number of shoots of the circuit.
    max_iterations : int (default=5)
        Maximum number of iterations of the Grover's algorithm.
    
    Returns:
    -------
    list[int]
        List of two prime numbers that make up the number_1.
    '''

    ''' ---------------------- Problem Setup ---------------------- '''

    logger = logger if logger is not None else log('Task_1', 'TASK 1 - Goldbach\'s conjecture')
    logging_block = '\n\tNumber to be decomposed: ' + str(number_1)

    # Identify the number of qubits needed to represent the largest number in the list of prime numbers.
    max_number = np.max(list_primes)
    num_qubits_max_number = int(np.ceil( np.log2( max_number ) )) +1
    
    # Compute the initial state as an uniform superposition of all possible prime numbers contained in the list.
    input_state = np.zeros( 2**num_qubits_max_number )
    for prime in list_primes:
        input_state[prime] = 1
    input_state /= np.sqrt( len(list_primes) )         

    # Obtain the gate that prepares the equal superposition of all the prime numbers in the list.
    input_primes_state = Schmidt_decomposition_based_preparation(input_state, num_qubits_max_number)

    # Obtain the gates that implement the Grover's algorithm
    A_gate = A(input_primes_state, num_qubits_max_number)
    S_f_gate = S_f(number_1, num_qubits_max_number + 1)
    Grover_gate = Grover(A_gate, S_f_gate, num_qubits_max_number)

    # Number of iterations of the Grover's algorithm
    n_iterations = 1
    n_try = 0
    
    number_of_operations = np.zeros( 2 , dtype=int) # Number of Addition Operation, Number of search operation

    ''' ----------------------------------------------------------- '''

    ''' ----------------------------------------------------------------
    Iteration of the Grover's algorithm until the desired result is found 
    or the maximum number of iterations is reached.
    ---------------------------------------------------------------- '''

    while n_try < max_tries:
        logging_block += f'\n\tTry {n_try+1} of {max_tries} tries.'
        logging_block += f'\n\tNumber of iterations: {n_iterations}'

        number_of_operations += np.array([ 2*n_iterations + 1, n_iterations]) * N_shoots

        q_r_numberA = QuantumRegister(num_qubits_max_number, name='A')      # Register for the first summand
        q_r_numberB = QuantumRegister(num_qubits_max_number, name='B')      # Register for the second summand
        q_r_carry = QuantumRegister(num_qubits_max_number, name='Carry')    # Register for the carry bits
        q_r_number = q_r_numberB[:] + [q_r_carry[-1]]                       # Register for the sum of the two numbers

        c_r_number = ClassicalRegister(num_qubits_max_number * 2 + 1, name='Result')    # Register for the result
        qc = QuantumCircuit(q_r_numberA, q_r_numberB, q_r_carry, c_r_number)  

        ''' ---------------------- Grover's Algorithm ---------------------- '''
        
        # Prepare the equal superposition of all the possible sums of the two numbers in registers A and B
        qc.append(A_gate, q_r_numberA[:] + q_r_numberB[:] + q_r_carry[:])

        for _ in range(n_iterations):
            qc.append(Grover_gate, q_r_numberA[:] + q_r_numberB[:] + q_r_carry[:])

        # Measure the result
        qc.measure(q_r_numberA, c_r_number[:num_qubits_max_number])
        qc.measure(q_r_number, c_r_number[num_qubits_max_number:])

        ''' ---------------------------------------------------------------- '''
        
        ''' ---------------------- Process the results ---------------------- '''

        counts = execute_circuit(qc, N_shoots)
        
        # Convert the keys from binary to decimal. (A+B, A)
        keys = [(key[:num_qubits_max_number + 1], key[num_qubits_max_number + 1:]) for key in counts.keys()]
        keys = list(map( lambda x: (int(x[0], 2), int(x[1], 2)), keys))

        results_values = None
        for n, a in keys:
            if n == number_1:
                results_values = (n, a)
                break
        
        number_of_operations += np.array([0, len(keys)])

        ''' ---------------------------------------------------------------- '''

        if results_values is not None:
            logging_block += f'\n\tNumber of Addition Operations: {number_of_operations[0]}'
            logging_block += f'\n\tNumber of Search Operations: {number_of_operations[1]}'
            logging_block += f'\n\tResults: {number_1 - results_values[1]}, {results_values[1]}'
            logger.info(logging_block)
            return number_1 - results_values[1], results_values[1], *number_of_operations

        if n_iterations == max_iterations:
            logging_block += '\n\tThe maximum number of iterations is reached.'
            n_iterations = 1
            n_try += 1
            continue

        n_iterations += 1
        n_try += 1

    # If the results are not found,
    logging_block += '\n\tThe maximum number of tries is reached.'
    logging_block += '\n\tThe results were not found.'
    logger.info(logging_block)
    return None

if __name__ == '__main__':

    qiskit_logger = logging.getLogger('qiskit')
    qiskit_logger.setLevel(logging.ERROR)

    logger = log('Task_1', 'TASK 1 - Goldbach\'s conjecture')

    max_tries = 10
    N_shoots = 5
    max_iterations = 5

    # Numbers of repeatition of the algorithm
    N = 10
    N_iterator = np.arange(1, N+1)

    # Numbers to be decomposed into prime numbers
    Numbers = np.arange(2, 102, 2)


    def iterator_function(N_iterator):
        for it in N_iterator:
            # Make header for iteration
                             
            result_list = []
            number_solved = 0
            n_to_solve = len(Numbers)

            for i, n in enumerate(Numbers):
                logger.info(f'\n ------ Iteration: {it}/{N} Number solved: {number_solved}/{n_to_solve} Correctly Solved: {number_solved}/{i}')
                list_primes = [1] + [n if isprime(n) else None for n in range(2, n+1)]
                list_primes = [n for n in list_primes if n is not None]
                res = find_the_primes_numbers(n, list_primes, max_tries=max_tries, N_shoots=N_shoots, max_iterations=max_iterations, logger=logger)
                result_list.append(res)
                number_solved += 1 if res is not None else 0

            results = np.array(result_list)

            path_to_save = 'Results/'

            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            filename = f'results_{it}.npy'

            with open(path_to_save + filename, 'wb') as f:
                logger.info(f'\n  Saving results in {path_to_save + filename}')
                np.save(f, results)

            print(f'\n ------ Iteration: {it}/{N} Number solved: {number_solved}/{n_to_solve} Correctly Solved: {number_solved}/{n_to_solve}')
    

    parallelize('Task_1', iterator_function, N_iterator, n_process=5, logger=logger)
    logger.footer()
