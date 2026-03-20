import os
from dotenv import load_dotenv
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def debug_sampler_result():
    load_dotenv()
    api_key = os.getenv("IBM_API_KEY")
    service = QiskitRuntimeService(token=api_key)
    # Using local simulator for speed of structure check if possible, 
    # but the user has real devices. Let's use a simulator if available in Runtime.
    # Actually, let's just use the first available backend.
    backend = service.backends()[0]
    print(f"Using backend: {backend.name}")
    
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    
    isa_qc = transpile(qc, backend=backend)
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_qc], shots=10)
    result = job.result()
    
    pub_result = result[0]
    print(f"\nPub result data type: {type(pub_result.data)}")
    print(f"Pub result data attributes/keys: {dir(pub_result.data)}")
    
    # Try common access patterns
    try:
        print(f"data.meas: {pub_result.data.meas}")
    except Exception as e:
        print(f"Failed data.meas: {e}")
        
    try:
        # Check for bitstrings
        # In newer Qiskit, it might be result.data.c (default reg name) or similar
        for key in pub_result.data:
            print(f"Found key: {key}")
    except Exception as e:
        print(f"Failed iterating keys: {e}")

if __name__ == "__main__":
    debug_sampler_result()
