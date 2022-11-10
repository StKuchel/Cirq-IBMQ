
from typing import Optional, Union, Sequence

import numpy as np
import cirq
from qiskit import QuantumCircuit, execute, IBMQ
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.compiler import transpile


WRITE_CIRCUIT = False

def get_labels(ibm_result, qasm_circuit):
    """_summary_

    Args:
        ibm_result (_type_): _description_
        qasm_circuit (_type_): _description_

    Returns:
        _type_: _description_
    """
    res_list = ibm_result.to_dict()["results"]
    # now find the element with name=qasm_circuit
    result = list(filter(lambda x: x["header"]["name"] == qasm_circuit.name, res_list))[0]
    return [e[0][2:] for e in result["header"]["clbit_labels"]]

def transform_output(results, circuits, resolvers):
    """_summary_

    Args:
        results (_type_): _description_
        circuits (_type_): _description_
        resolvers (_type_): _description_

    Returns:
        _type_: _description_
    """
    output = []
    for i, circuit in enumerate(circuits):
        labels = get_labels(results, circuit)
        reformated_result = {e: [] for e in labels}

        counts = results.get_counts(circuit)
        for key, duplications in counts.items():
            measurements = key.split()
            for j, measurement in enumerate(measurements):
                reformated_result[labels[j]].extend(
                    [[int(measurement)]] * duplications
                )

        output.append(
            cirq.ResultDict(
                params=resolvers[i],
                measurements={
                    k: np.array(v) for k, v in reformated_result.items()
                },
            )
        )
    return output

# Backend can be actual hardware or simulator e.g. 'qasm_simulator'
def qc_exe(circuits, backend, resolvers, repetitions):
    """_summary_

    Args:
        circuits (_type_): _description_
        backend (_type_): _description_
        resolvers (_type_): _description_
        repetitions (_type_): _description_
    """
    if not isinstance(circuits, list):
        circuits=[circuits]

# coupling map specified for 4 qubits on ibmq_belem (related error appears to be backend specific)
    if IBMQ.active_account():
        provider = IBMQ.get_provider()
        if backend in provider.backends():
            # need to map circuits onto backend's gate set
            circuits = transpile(circuits = circuits, backend = backend)
            # IBMQJobManger manages job sizes w.r.t. max_experiments of selected IBMQ backend
            job_manager = IBMQJobManager()
            current_job = job_manager.run(circuits, backend, shots=max(repetitions))
            results = current_job.results()
            results = results.combine_results()
        else:
            raise NameError('The provided IBMQ backend does not exist.')
    else:
        current_job = execute(circuits, backend, shots=max(repetitions))
        results = current_job.result()
    # now reformat for cirq

    return transform_output(results, circuits, resolvers)

# This is the transformer
def cirq2qasm(circuit):
    """Representation of a cirq.Circuit in QASM format via cirq.QasmOutput.
    Args:
      operations: Tree of operations to insert.
      qubits: The qubits used in the operations.
      header: A multi-line string that is placed in a comment at the top
            of the QASM.
        precision: The number of digits after the decimal to show for
          numbers in the QASM code.
        version: The QASM version to target. Objects may return different
            QASM depending on version.
    """
    qasm_output = cirq.QasmOutput(
        circuit.all_operations(), circuit.all_qubits()
    )
    qasm_circuit = QuantumCircuit().from_qasm_str(str(qasm_output))

    if WRITE_CIRCUIT:
        qasm_circuit.qasm(True, qasm_circuit.name + ".qasm")
    return qasm_circuit


class QiskitSampler(cirq.Sampler):
    def __init__(
        self,
        backend,
        executor,
        transformer,
    ):
        """Initializes a QiskitSampler.
        Args:
            backend: A QuantumComputer against which to run the
                cirq.Circuits. -> provider.get_backend('your prefered backend')
            executor: A callable that first uses the below transformer` on cirq.Circuit s and
                then executes the transformed circuit on the quantum_computer.
                Qiskit's standard execute command execute(circuit, backend, shots) should do.
            transformer: Transforms the cirq.Circuit into the QASM format.
                You can use the callable 'cirq2qasm' provided below.
        """

        self.backend = backend
        self.executor = executor
        self.transformer = transformer

    def run_sweep(
        self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int = 1
    ):
        """This will evaluate results on the circuit for every set of parameters in `params`.
        Args:
            program: Circuit to evaluate for each set of parameters in `params`.
            params: `cirq.Sweepable` of parameters which this function passes to
                `cirq.protocols.resolve_parameters` for evaluating the circuit.
            repetitions: Number of times to run each iteration through the `params`. For a given
                set of parameters, the `cirq.Result` will include a measurement for each repetition.
        Returns:
            A list of `cirq.Result` s.
        """

        # abstract_circuit = program.unfreeze(copy=False)
        # resolvers = list(cirq.to_resolvers(params))
        # circuits = [
        #     cirq.protocols.resolve_parameters(abstract_circuit, resolver)
        #     for resolver in resolvers
        # ]
        # qasm_circuits = [self.transformer(circuit) for circuit in circuits]

        # output = self.executor(
        #     circuits=qasm_circuits,
        #     backend=self.backend,
        #     resolvers=resolvers,
        #     repetitions=repetitions,
        # )
        # return output
        return self.run_batch([program], repetitions, [params])[0]

    def run_batch(
        self,
        programs: Sequence["cirq.AbstractCircuit"],
        repetitions: Union[int, Sequence[int]],
        params_list: Optional[Sequence["cirq.Sweepable"]] = None,
    ) -> Sequence[Sequence["cirq.Result"]]:
        params_list, repetitions = self._normalize_batch_args(
            programs, params_list, repetitions
        )

        abstract_circuits = [
            program.unfreeze(copy=False) for program in programs
        ]
        resolvers = [list(cirq.to_resolvers(params)) for params in params_list]
        circuits = [
            cirq.protocols.resolve_parameters(abstract_circuit, resolver)
            for abstract_circuit, resolver in zip(abstract_circuits, resolvers)
        ]

        qasm_circuits = [self.transformer(circuit) for circuit in circuits]

        output = self.executor(
            circuits=qasm_circuits,
            backend=self.backend,
            resolvers=resolvers,
            repetitions=repetitions,
        )

        return output

# Callable of the sampler
def get_qiskit_sampler(
    backend):
    """Initialize QiskitSampler.
    Args:
        quantum_computer: The name of the desired quantum computer. Need to check what exaxtly to
        call for IBMQ. Maybe IBMQ.get_provider()
        executor: A callable that first uses the below transformer on cirq.Circuit s and
            then executes the transformed circuit on the quantum_computer. You may pass your
            own callable or any static method on CircuitSweepExecutors.
        transformer: A callable that transforms the cirq.Circuit into a pyquil.Program.
            You may pass your own callable or any static method on CircuitTransformers.
    Returns:
        A QiskitSampler with the specified quantum processor, executor, and transformer.
    """
    return QiskitSampler(backend, qc_exe, cirq2qasm)
