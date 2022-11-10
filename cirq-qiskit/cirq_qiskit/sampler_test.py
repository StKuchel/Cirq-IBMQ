
import cirq
from cirq_qiskit.sampler import get_qiskit_sampler, cirq2qasm, qc_exe
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
import numpy as np
from unittest.mock import patch, MagicMock, ANY
import pytest


def test_cirq2qasm_toffoli():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.decompose(cirq.TOFFOLI(q0, q1, q2)))

    qc = QuantumCircuit(3)
    # Toffoli with control qubits
    qc.ccx(0,1,2)
    assert Statevector.from_instruction(cirq2qasm(circuit)).equiv(Statevector.from_instruction(qc)),\
            f"Statevector of is different:\n Converted Circuit:\n{cirq2qasm(circuit)}\n Expected Circuit:\n{qc}"
            
def test_cirq2qasm_inverseQFT():
    
    def make_qft_inverse(qubits):
        """Generator for the inverse QFT on a list of qubits."""
        qreg = list(qubits)[::-1]
        while len(qreg) > 0:
            q_head = qreg.pop(0)
            yield cirq.H(q_head)
            for i, qubit in enumerate(qreg):
                yield (cirq.CZ ** (-1 / 2 ** (i + 1)))(qubit, q_head)

    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(make_qft_inverse(qubits))


    def nBitQFT(n):
        circuit = QuantumCircuit(n)
        #
        # We start with the most significant bit
        #
        for k in range(n):
            j = n - k
            # Add the Hadamard to qubit j-1
            circuit.h(j-1)
            #
            # there is one conditional rotation for
            # each qubit with lower significance
            for i in reversed(range(j-1)):
                circuit.cu1(2*np.pi/2**(j-i),i, j-1)
        #
        # Finally we need to swap qubits
        #
        for i in range(n//2):
            circuit.swap(i, n-i-1)
        return circuit
    qc = nBitQFT(4)

    assert Statevector.from_instruction(cirq2qasm(circuit)).equiv(Statevector.from_instruction(qc)),\
            f"Statevector of is different:\n Converted Circuit:\n{cirq2qasm(circuit)}\n Expected Circuit:\n{qc}"

def test_qc_exe():
    with (
        patch('cirq_qiskit.sampler.IBMQ') as mock_IBMQ,
        patch('cirq_qiskit.sampler.IBMQJobManager') as mock_job_manager,
        patch('cirq_qiskit.sampler.transpile') as mock_transpile,
        patch('cirq_qiskit.sampler.transform_output') as mock_transform,
    ):
        mock_IBMQ.active_account.return_value = True
        mock_IBMQ.get_provider().backends.return_value = ['test_backend']
        
        #assert 0, mock_IBMQ.get_provider().backends()
        circuits = MagicMock()
        resolver = MagicMock()
        
        result = qc_exe(circuits=circuits, backend='test_backend', resolvers=resolver, repetitions=[100, 150])
        
        mock_transpile.assert_called_once_with(circuits=[circuits], backend='test_backend')
        mock_job_manager().run.assert_called_once_with(mock_transpile(), 'test_backend', shots=150)
        mock_transform.assert_called_once_with(ANY, mock_transpile(), resolver)
        assert result == mock_transform()
        
        #mock_transform.assert_called_once_with(mock.Any)
        
    #mock ibmq active account
    #mock ibmq backends
    #mock transform output
    #mock transpile
    

def test_transform_output():
    pass

def test_run_sweep():
    
    bell_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    bell_circuit.append(cirq.H(qubits[0]))
    bell_circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    bell_circuit.append(cirq.measure(qubits[0], qubits[1], key='m'))
    
    sampler = get_qiskit_sampler('Backend')
    sampler.executor = MagicMock('executor')
    sampler.run_sweep(bell_circuit, params=[cirq.ParamResolver({})], repetitions=10)
    
    sampler.executor.assert_called_once()
    _, kwargs = sampler.executor.call_args_list[0]
    assert isinstance(kwargs['circuits'][0], QuantumCircuit)
    assert len(kwargs['circuits']) == 1
    assert kwargs['resolvers'] == [[cirq.ParamResolver({})]]
    assert kwargs['repetitions'] == [10]
    assert kwargs['backend'] == 'Backend'
    

def test_run_batch():
    bell_circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    bell_circuit.append(cirq.H(qubits[0]))
    bell_circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    bell_circuit.append(cirq.measure(qubits[0], qubits[1], key='m'))
    
    sampler = get_qiskit_sampler('Backend')
    sampler.executor = MagicMock('executor')
    sampler.run_batch([bell_circuit, bell_circuit], params_list=[cirq.ParamResolver({}), cirq.ParamResolver({})], repetitions=[10, 15])
    
    sampler.executor.assert_called_once()
    _, kwargs = sampler.executor.call_args_list[0]
    assert isinstance(kwargs['circuits'][0], QuantumCircuit)
    assert isinstance(kwargs['circuits'][1], QuantumCircuit)
    assert len(kwargs['circuits']) == 2
    assert kwargs['resolvers'] == [[cirq.ParamResolver({})], [cirq.ParamResolver({})]]
    assert kwargs['repetitions'] == [10, 15]
    assert kwargs['backend'] == 'Backend'

def test_get_qiskit_sampler():
    sampler = get_qiskit_sampler('Backend')
    assert sampler.backend == 'Backend'
    assert sampler.executor == qc_exe
    assert sampler.transformer == cirq2qasm
