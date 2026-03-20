import os
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

def check_devices():
    # Load .env file
    load_dotenv()
    
    api_key = os.getenv("IBM_API_KEY")
    if not api_key:
        print("❌ Error: IBM_API_KEY not found in .env file.")
        return

    print("✅ IBM_API_KEY found. Authenticating...")
    
    try:
        # Initialize the service
        service = QiskitRuntimeService(token=api_key)
        
        print("\n=== Available IBM Quantum Backends ===")
        backends = service.backends()
        
        if not backends:
            print("No backends found.")
            return

        # Sort backends by qubit count for better readability
        backends.sort(key=lambda x: x.num_qubits, reverse=True)

        header = f"{'Backend Name':<25} | {'Qubits':<8} | {'Status':<15} | {'Queue'}"
        print(header)
        print("-" * len(header))

        for b in backends:
            status = "Online" if b.status().operational else "Offline"
            pending_jobs = b.status().pending_jobs
            print(f"{b.name:<25} | {b.num_qubits:<8} | {status:<15} | {pending_jobs} jobs")

    except Exception as e:
        print(f"❌ Authentication failed: {e}")

if __name__ == "__main__":
    check_devices()
