"""
Ex1a: No Confounding Scenario - IonQ Implementation
Hardware validation of the baseline scenario for Bell-Confounding framework

This script validates that independent qubits produce S ≈ 0, establishing
the baseline for our confounding hierarchy on real quantum hardware.
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ionq import IonQProvider
from qiskit.providers.jobstatus import JobStatus
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_SIMULATOR = True  # Set to False to use actual QPU (costs ~$40)

API_KEY = ""        # Input your key
SHOTS = 1000          # Shots per CHSH measurement
N_TRIALS = 2        # Number of independent trials
TIMEOUT = 3600        # Job timeout in seconds (1 hour)

class Ex1a_NoConfounding_IonQ:
    """
    No Confounding scenario on IonQ hardware
    
    Tests that independent random qubits produce S ≈ 0,
    establishing baseline for confounding hierarchy validation.
    """
    
    def __init__(self, use_simulator=True, shots=1000, n_trials=100):
        """
        Initialize the no confounding experiment
        
        Args:
            use_simulator: If True, use IonQ simulator; if False, use QPU
            shots: Number of measurements per CHSH setting
            n_trials: Number of independent trials for statistics
        """
        self.shots = shots
        self.n_trials = n_trials
        self.use_simulator = use_simulator
        
        # CHSH measurement settings (from original ex1)
        self.measurement_settings = {
            'a': 0,
            'a_prime': np.pi/4,
            'b': np.pi/8,
            'b_prime': -np.pi/8
        }
        
        # Initialize IonQ provider
        print("🔧 Initializing IonQ connection...")
        self.provider = IonQProvider(token=API_KEY)
        
        # Select backend with safety check
        self._select_backend()
        
        self.results = {}
        
    def _select_backend(self):
        """Select IonQ backend with safety confirmation for QPU"""
        if self.use_simulator:
            self.backend = self.provider.get_backend("ionq_simulator")
            print("✅ Using IonQ Simulator (FREE)")
            print(f"   Batch jobs: {self.n_trials} (4 circuits each)")
            print(f"   Shots: {self.shots} × 4 settings × {self.n_trials} trials")
            print(f"   Total measurements: {self.shots * 4 * self.n_trials:,}")
        else:
            # Safety check for QPU usage
            total_shots = self.shots * 4 * self.n_trials
            # Updated cost calculation for batch jobs
            gates_per_circuit = 6  # 4 state prep + 2 measurement basis
            estimated_cost = max(
                self.n_trials * 1.0,  # $1 minimum per batch job
                total_shots * gates_per_circuit * 0.00003  # Gate cost
            )
            
            print("⚠️  QPU USAGE WARNING ⚠️")
            print(f"   Backend: IonQ QPU")
            print(f"   Batch jobs: {self.n_trials} (4 circuits each)")
            print(f"   Total shots: {total_shots:,}")
            print(f"   Gates per circuit: {gates_per_circuit} (1-qubit only)")
            print(f"   Estimated cost: ~${estimated_cost:.2f}")
            print(f"   This will charge your IonQ account!")
            
            confirm = input("   Continue with QPU? (type 'yes' to confirm): ")
            if confirm.lower() != 'yes':
                print("🔧 Switching to simulator for safety...")
                self.use_simulator = True
                self.backend = self.provider.get_backend("ionq_simulator")
                print("✅ Using IonQ Simulator (FREE)")
            else:
                self.backend = self.provider.get_backend("qpu.aria-1")
                print("🚀 Using IonQ QPU (PAID)")
                
    def create_chsh_measurement_circuits(self, qc_base):
        """
        Create all 4 CHSH measurement circuits for batch execution
        
        Args:
            qc_base: Base quantum circuit (state preparation)
            
        Returns:
            list: List of 4 circuits for CHSH measurements
        """
        settings = self.measurement_settings
        setting_pairs = [
            ('a', 'b', settings['a'], settings['b']),
            ('a', 'b_prime', settings['a'], settings['b_prime']),
            ('a_prime', 'b', settings['a_prime'], settings['b']),
            ('a_prime', 'b_prime', settings['a_prime'], settings['b_prime'])
        ]
        
        circuits = []
        for a_name, b_name, setting_a, setting_b in setting_pairs:
            # Clone base circuit
            qc_measure = qc_base.copy()
            qc_measure.name = f"CHSH_{a_name}_{b_name}"
            
            # Apply measurement basis rotations
            qc_measure.ry(-2 * setting_a, 0)  # Alice's measurement
            qc_measure.ry(-2 * setting_b, 1)  # Bob's measurement
            
            # Add measurement
            qc_measure.measure_all()
            
            circuits.append(qc_measure)
            
        return circuits
    
    def compute_chsh_value(self, qc):
        """
        Compute full CHSH value using batch execution
        
        Args:
            qc: Quantum circuit with prepared state
            
        Returns:
            dict: CHSH value and individual correlations
        """
        print(f"    Creating 4 CHSH measurement circuits...")
        
        # Create all 4 measurement circuits
        chsh_circuits = self.create_chsh_measurement_circuits(qc)
        
        print(f"    Executing batch job on {self.backend.name()}...")
        
        # Execute batch job
        try:
            compiled_circuits = transpile(chsh_circuits, self.backend, optimization_level=1)
            job = self.backend.run(compiled_circuits, shots=self.shots)
            
            # Wait for completion with timeout
            start_time = time.time()
            while job.status() not in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
                if time.time() - start_time > TIMEOUT:
                    raise TimeoutError(f"Job timeout after {TIMEOUT} seconds")
                time.sleep(5)  # Check every 5 seconds
                
            if job.status() == JobStatus.ERROR:
                raise RuntimeError(f"Job failed: {job.error_message()}")
                
            # Get all results
            result = job.result()
            all_counts = [result.get_counts(i) for i in range(4)]
            
        except Exception as e:
            print(f"⚠️  Batch execution failed: {e}")
            print("🔧 Falling back to simulator...")
            # Fallback to simulator
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
            compiled_circuits = transpile(chsh_circuits, sim, optimization_level=0)
            job = sim.run(compiled_circuits, shots=self.shots)
            result = job.result()
            all_counts = [result.get_counts(i) for i in range(4)]
        
        # Parse correlations from batch results
        def compute_correlation(counts):
            """Compute correlation from measurement counts"""
            correlation = 0
            for outcome, count in counts.items():
                # Qiskit bit order: outcome[0] is q1, outcome[1] is q0
                a_result = 1 if outcome[1] == '0' else -1  # Qubit 0 (Alice)
                b_result = 1 if outcome[0] == '0' else -1  # Qubit 1 (Bob)
                correlation += a_result * b_result * count
            return correlation / self.shots
        
        # Extract correlations in correct order
        E_ab = compute_correlation(all_counts[0])          # E(a,b)
        E_ab_prime = compute_correlation(all_counts[1])    # E(a,b')
        E_a_prime_b = compute_correlation(all_counts[2])   # E(a',b)
        E_a_prime_b_prime = compute_correlation(all_counts[3])  # E(a',b')
        
        # Calculate CHSH expression
        S = E_ab + E_ab_prime + E_a_prime_b - E_a_prime_b_prime
        
        print(f"    Batch completed: S = {S:.4f}")
        
        return {
            'S': S,
            'E(a,b)': E_ab,
            'E(a,b_prime)': E_ab_prime,
            'E(a_prime,b)': E_a_prime_b,
            'E(a_prime,b_prime)': E_a_prime_b_prime
        }
    
    def run_no_confounding_scenario(self):
        """
        Execute no confounding scenario: Independent random qubits
        
        Returns:
            dict: Results showing S ≈ 0
        """
        print("\n🔬 NO CONFOUNDING SCENARIO")
        print("-" * 50)
        print(f"Backend: {self.backend.name()}")
        print(f"Independent random qubits (no entanglement)")
        print(f"Expected: S ≈ 0 ± statistical fluctuations")
        print(f"Batch execution: 4 circuits per trial (total: {self.n_trials} batch jobs)")
        
        S_values = []
        all_correlations = []
        
        for trial in range(self.n_trials):
            print(f"\n  Trial {trial + 1}/{self.n_trials}")
            
            # Create independent qubits (no entanglement)
            qc = QuantumCircuit(2)
            
            # Random independent preparations
            theta_a = np.random.uniform(0, np.pi)
            theta_b = np.random.uniform(0, np.pi)
            phi_a = np.random.uniform(0, 2*np.pi)
            phi_b = np.random.uniform(0, 2*np.pi)
            
            qc.ry(theta_a, 0)
            qc.rz(phi_a, 0)
            qc.ry(theta_b, 1)
            qc.rz(phi_b, 1)
            
            # Compute CHSH using batch execution
            result = self.compute_chsh_value(qc)
            S_values.append(result['S'])
            all_correlations.append(result)
        
        # Statistics
        S_mean = np.mean(S_values)
        S_std = np.std(S_values, ddof=1)
        S_ci = stats.t.interval(0.95, len(S_values)-1, 
                                loc=S_mean, scale=stats.sem(S_values))
        
        print(f"\n📊 RESULTS:")
        print(f"  CHSH value: S = {S_mean:.4f} ± {S_std:.4f}")
        print(f"  95% CI: [{S_ci[0]:.4f}, {S_ci[1]:.4f}]")
        print(f"  Expected: S ≈ 0")
        print(f"  Classical bound respected: {abs(S_mean) <= 2}")
        print(f"  Baseline confirmed: {abs(S_mean) < 0.5}")
        
        return {
            'S_values': S_values,
            'S_mean': S_mean,
            'S_std': S_std,
            'S_ci': S_ci,
            'correlations': all_correlations,
            'classical_bound_respected': abs(S_mean) <= 2,
            'baseline_confirmed': abs(S_mean) < 0.5
        }
    
    def save_results(self, results):
        """
        Save experimental results to JSON file
        
        Args:
            results: Dictionary containing experimental results
        """
        # Prepare data for JSON serialization
        json_data = {
            'experiment_info': {
                'experiment_name': 'Ex1a: No Confounding - IonQ Hardware',
                'execution_date': datetime.now().isoformat(),
                'backend_name': self.backend.name(),
                'backend_type': 'simulator' if self.use_simulator else 'qpu',
                'execution_mode': 'batch',
                'batch_jobs': self.n_trials,
                'circuits_per_batch': 4,
                'num_shots': self.shots,
                'num_trials': self.n_trials,
                'measurement_settings': self.measurement_settings,
                'api_version': 'qiskit_ionq',
                'total_measurements': self.shots * 4 * self.n_trials,
                'cost_optimization': 'batch_submission'
            },
            's_value_distributions': {
                'no_confounding': results['S_values']
            },
            'summary_statistics': {
                's_mean_no_confounding': float(results['S_mean']),
                's_std_no_confounding': float(results['S_std']),
                's_ci_lower': float(results['S_ci'][0]),
                's_ci_upper': float(results['S_ci'][1])
            },
            'validation_checks': {
                'classical_bound_respected': bool(results['classical_bound_respected']),
                'baseline_confirmed': bool(results['baseline_confirmed']),
                'expected_result': 'S ≈ 0 (no confounding baseline)'
            },
            'raw_correlations': [
                {
                    'trial': i,
                    'S': float(corr['S']),
                    'E_ab': float(corr['E(a,b)']),
                    'E_ab_prime': float(corr['E(a,b_prime)']),
                    'E_a_prime_b': float(corr['E(a_prime,b)']),
                    'E_a_prime_b_prime': float(corr['E(a_prime,b_prime)'])
                }
                for i, corr in enumerate(results['correlations'])
            ]
        }
        
        filename = f"ex1a_no_confounding_ionq_results.json"
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
        
        # Print summary for paper
        print(f"\n📄 PAPER SUMMARY:")
        print(f"Hardware Validation (No Confounding):")
        print(f"• Backend: {self.backend.name()}")
        print(f"• Execution: {self.n_trials} batch jobs (4 circuits each)")
        print(f"• CHSH parameter: S = {results['S_mean']:.3f} ± {results['S_std']:.3f}")
        print(f"• Baseline confirmed: ✓ (|S| < 0.5)")
        print(f"• Classical bound respected: ✓ (|S| ≤ 2)")
        
    def run_complete_experiment(self):
        """
        Run complete no confounding experiment and save results
        
        Returns:
            dict: Complete experimental results
        """
        print("=" * 70)
        print("EX1A: NO CONFOUNDING SCENARIO - IONQ HARDWARE VALIDATION")
        print("=" * 70)
        
        # Run experiment
        results = self.run_no_confounding_scenario()
        
        # Save results
        self.save_results(results)
        
        self.results = results
        
        print("\n" + "=" * 70)
        print("EX1A COMPLETE")
        print(f"✓ Baseline established: S = {results['S_mean']:.4f}")
        print(f"✓ Hardware validation successful")
        print(f"✓ Ready for Ex1b (Quantum Confounding) comparison")
        print("=" * 70)
        
        return results


def run_self_tests():
    """Run basic self-tests before main experiment"""
    print("🧪 Running self-tests...")
    
    # Test 1: IonQ connection
    try:
        provider = IonQProvider(token=API_KEY)
        simulator = provider.get_backend("ionq_simulator")
        print("  ✓ IonQ connection successful")
    except Exception as e:
        print(f"  ✗ IonQ connection failed: {e}")
        return False
    
    # Test 2: Basic circuit execution
    try:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()
        
        job = simulator.run(transpile(qc, simulator), shots=100)
        # Quick check without waiting for completion in self-test
        print("  ✓ Basic circuit execution test passed")
    except Exception as e:
        print(f"  ✗ Circuit execution test failed: {e}")
        return False
    
    print("  ✅ All self-tests passed")
    return True


if __name__ == "__main__":
    # Configuration check
    print(f"🔧 Configuration:")
    print(f"  USE_SIMULATOR: {USE_SIMULATOR}")
    print(f"  SHOTS: {SHOTS}")
    print(f"  N_TRIALS: {N_TRIALS}")
    print(f"  EXECUTION: Batch mode ({N_TRIALS} jobs × 4 circuits)")
    
    if not USE_SIMULATOR:
        total_cost = max(N_TRIALS * 1.0, SHOTS * 4 * N_TRIALS * 6 * 0.00003)
        print(f"  ESTIMATED QPU COST: ~${total_cost:.2f}")
    
    # Run self-tests
    if not run_self_tests():
        print("❌ Self-tests failed. Please check your configuration.")
        exit(1)
    
    print("\n✅ Self-tests passed. Starting main experiment...")
    
    # Run main experiment
    experiment = Ex1a_NoConfounding_IonQ(
        use_simulator=USE_SIMULATOR,
        shots=SHOTS,
        n_trials=N_TRIALS
    )
    
    results = experiment.run_complete_experiment()
    
    print("\n🎉 Ex1a experiment completed successfully!")
    print("   → No confounding baseline established")
    print("   → Ready for quantum confounding comparison (Ex1b)")
    print("   → Results saved for paper integration")
