import os
import time
from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def test_submit_torino():
    # Load .env file
    load_dotenv()
    
    api_key = os.getenv("IBM_API_KEY")
    if not api_key:
        print("❌ Error: IBM_API_KEY not found in .env file.")
        return

    print("✅ IBM_API_KEY found. Authenticating...")
    
    try:
        # Initialize the service
        print("Initializing QiskitRuntimeService...")
        service = QiskitRuntimeService(token=api_key)
        
        # Target backend
        backend_name = "ibm_torino"
        
        print("\nSearching for backend...")
        # Try to list backends first to trigger discovery (like in check_ibm_devices)
        backends = service.backends()
        print(f"Discovered {len(backends)} backends.")
        
        print(f"Connecting to {backend_name}...")
        backend = service.backend(backend_name)
        print(f"✅ Backend {backend_name} found!")
        
        # Create a simple Bell state circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        print("\nTranspiling circuit for backend...")
        from qiskit import transpile
        isa_circuit = transpile(qc, backend=backend)
        
        print("\n=== Submitting Test Circuit ===")
        print(isa_circuit.draw())
        
        # Initialize Sampler
        print("Initializing Sampler (V2)...")
        sampler = Sampler(mode=backend)
        
        # Submit the job
        print("Submitting job...")
        job = sampler.run([(isa_circuit)], shots=128)
        print(f"\n🚀 Job submitted successfully!")
        print(f"   Job ID: {job.job_id()}")
        print(f"   Status: {job.status()}")
        print(f"\nYou can track your job at: https://quantum.ibm.com/jobs/{job.job_id()}")

    except Exception as e:
        print(f"❌ Failed to submit job: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_submit_torino()
