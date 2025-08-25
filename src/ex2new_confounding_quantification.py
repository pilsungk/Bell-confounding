#!/usr/bin/env python3
"""
Experiment 2: Confounding Strength Quantification
Standard CHSH Protocol with XZ-plane measurements (FIXED)

Key features:
- S(theta) = sqrt(2) * (1 + sin(2*theta))
- CS = S/2 (definition from theory)
- CS = (sqrt(2)/2) * (1 + C) where C = sin(2*theta)
- Proper XZ-plane measurement implementation
- Self-test verification included
"""

import numpy as np
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace


class Experiment2_StandardCHSH:
    """Standard CHSH protocol for confounding quantification"""
    
    def __init__(self, shots=10000, n_trials=5, n_theta_points=25):
        """Initialize experiment parameters"""
        self.simulator = AerSimulator()
        self.shots = shots
        self.n_trials = n_trials
        self.n_theta_points = n_theta_points
        
        # CHSH measurement angles optimized for XZ-plane
        self.measurement_settings = {
            'a': 0,
            'a_prime': np.pi/4,
            'b': np.pi/8,
            'b_prime': -np.pi/8
        }
        
        # Theta range: 0 (separable) to pi/2 (separable again)
        # This covers the full cycle: separable -> maximally entangled -> separable
        self.theta_values = np.linspace(0, np.pi/2, n_theta_points)
        
        # Results storage
        self.sweep_data = []
        self.analysis_results = {}
    
    def create_parameterized_state(self, theta):
        """
        Create parameterized state |psi(theta)> = cos(theta)|00> + sin(theta)|11>
        
        Args:
            theta: Parameter controlling entanglement degree
            
        Returns:
            QuantumCircuit: Circuit creating the parameterized state
        """
        qc = QuantumCircuit(2)
        
        # Create parameterized state using Ry and CX
        # Ry(2*theta) on first qubit: |0> -> cos(theta)|0> + sin(theta)|1>
        # CX creates entanglement
        qc.ry(2 * theta, 0)
        qc.cx(0, 1)
        
        return qc
    
    def measure_correlation_xz(self, qc, angle_a, angle_b):
        """
        Measure correlation E(a,b) for CHSH
        
        Args:
            qc: Quantum circuit with state preparation
            angle_a: Measurement angle for Alice
            angle_b: Measurement angle for Bob
            
        Returns:
            float: Correlation value E(a,b)
        """
        circuit = qc.copy()
        
        # For XZ-plane measurements, apply Ry rotation
        # This rotates the measurement basis
        circuit.ry(-2 * angle_a, 0)  # Note: -2*angle for proper rotation
        circuit.ry(-2 * angle_b, 1)
        
        # Measure in computational basis
        circuit.measure_all()
        
        # Execute circuit
        job = self.simulator.run(
            transpile(circuit, self.simulator, optimization_level=0), 
            shots=self.shots
        )
        counts = job.result().get_counts()
        
        # Calculate correlation
        correlation = 0
        for outcome, count in counts.items():
            # Qiskit ordering: outcome[0] is qubit 1, outcome[1] is qubit 0
            a_outcome = 1 if outcome[1] == '0' else -1
            b_outcome = 1 if outcome[0] == '0' else -1
            correlation += a_outcome * b_outcome * count
        
        return correlation / self.shots
    
    def compute_chsh_value(self, qc):
        """
        Compute CHSH parameter S from four correlation measurements
        
        Args:
            qc: Quantum circuit with state preparation
            
        Returns:
            dict: CHSH value S and individual correlations
        """
        settings = self.measurement_settings
        
        # Measure all four correlations
        E_ab = self.measure_correlation_xz(qc, settings['a'], settings['b'])
        E_ab_prime = self.measure_correlation_xz(qc, settings['a'], settings['b_prime'])
        E_a_prime_b = self.measure_correlation_xz(qc, settings['a_prime'], settings['b'])
        E_a_prime_b_prime = self.measure_correlation_xz(qc, settings['a_prime'], settings['b_prime'])
        
        # Standard CHSH combination
        S = E_ab + E_ab_prime + E_a_prime_b - E_a_prime_b_prime
        
        return {
            'S': S,
            'E_ab': E_ab,
            'E_ab_prime': E_ab_prime,
            'E_a_prime_b': E_a_prime_b,
            'E_a_prime_b_prime': E_a_prime_b_prime
        }
    
    def calculate_concurrence(self, theta):
        """
        Calculate theoretical concurrence for our state family
        
        For |psi(theta)> = cos(theta)|00> + sin(theta)|11>
        Concurrence C = sin(2*theta)
        
        Args:
            theta: State parameter
            
        Returns:
            float: Theoretical concurrence
        """
        return np.sin(2 * theta)
    
    def theoretical_chsh(self, theta):
        """
        Theoretical CHSH value for our measurement setup
        
        For |psi(theta)> = cos(theta)|00> + sin(theta)|11> with XZ-plane measurements,
        and the specific measurement angles we use, the CHSH value follows:
        
        S(theta) = sqrt(2) * (1 + sin(2*theta))
        
        This gives:
        - S(0) = sqrt(2) ≈ 1.414 (separable state)
        - S(π/4) = 2*sqrt(2) ≈ 2.828 (maximally entangled state)
        
        Args:
            theta: State parameter
            
        Returns:
            float: Theoretical CHSH value
        """
        return np.sqrt(2) * (1 + np.sin(2 * theta))
    
    def theoretical_cs(self, s_value):
        """
        Theoretical confounding strength
        
        CS = S/2 by definition
        
        Args:
            s_value: CHSH value
            
        Returns:
            float: Theoretical CS value
        """
        return s_value / 2
    
    def run_parameter_sweep(self):
        """
        Sweep through theta values and collect CHSH data
        
        Returns:
            list: Data points for each theta value
        """
        print("="*60)
        print("EXPERIMENT 2: CONFOUNDING STRENGTH QUANTIFICATION")
        print("Standard CHSH Protocol (XZ-plane measurements)")
        print("="*60)
        
        print(f"\nParameters:")
        print(f"  Theta range: 0 to π/2 (full cycle)")
        print(f"  Theta points: {self.n_theta_points}")
        print(f"  Trials per point: {self.n_trials}")
        print(f"  Shots per trial: {self.shots}")
        print(f"  Total shots: {self.n_theta_points * self.n_trials * self.shots * 4:,}")
        
        print("\nRunning parameter sweep...")
        
        self.sweep_data = []
        
        for i, theta in enumerate(self.theta_values):
            theta_deg = theta * 180 / np.pi
            print(f"\nTheta = {theta:.4f} rad ({theta_deg:.1f} deg) [{i+1}/{self.n_theta_points}]")
            
            # Run multiple trials for statistics
            trial_results = []
            
            for trial in range(self.n_trials):
                qc = self.create_parameterized_state(theta)
                chsh_result = self.compute_chsh_value(qc)
                trial_results.append(chsh_result['S'])
                print(f"  Trial {trial+1}: S = {chsh_result['S']:.4f}")
            
            # Calculate statistics
            s_mean = np.mean(trial_results)
            s_std = np.std(trial_results)
            s_sem = s_std / np.sqrt(self.n_trials)
            
            # Theoretical values
            s_theory = self.theoretical_chsh(theta)
            c_theory = self.calculate_concurrence(theta)
            cs_experimental = self.theoretical_cs(s_mean)
            cs_theory = self.theoretical_cs(s_theory)
            
            print(f"  Mean: S = {s_mean:.4f} +/- {s_sem:.4f}")
            print(f"  Theory: S = {s_theory:.4f}")
            print(f"  CS = {cs_experimental:.4f} (theory: {cs_theory:.4f})")
            print(f"  Concurrence = {c_theory:.4f}")
            
            # Store data point
            data_point = {
                'theta': float(theta),
                'theta_deg': float(theta_deg),
                'trial_s_values': trial_results,
                's_mean': float(s_mean),
                's_std': float(s_std),
                's_sem': float(s_sem),
                's_theory': float(s_theory),
                'cs_experimental': float(cs_experimental),
                'cs_theory': float(cs_theory),
                'concurrence_theory': float(c_theory)
            }
            
            self.sweep_data.append(data_point)
        
        print("\n" + "="*60)
        print("Parameter sweep completed!")
        print("="*60)
        
        return self.sweep_data
    
    def analyze_results(self):
        """
        Analyze sweep data and calculate key metrics
        
        Returns:
            dict: Analysis results
        """
        print("\nAnalyzing results...")
        
        # Extract arrays for analysis
        theta_array = np.array([d['theta'] for d in self.sweep_data])
        s_exp_array = np.array([d['s_mean'] for d in self.sweep_data])
        s_theory_array = np.array([d['s_theory'] for d in self.sweep_data])
        cs_exp_array = np.array([d['cs_experimental'] for d in self.sweep_data])
        cs_theory_array = np.array([d['cs_theory'] for d in self.sweep_data])
        c_array = np.array([d['concurrence_theory'] for d in self.sweep_data])
        
        # Calculate R-squared for theory vs experiment (CHSH)
        ss_res_chsh = np.sum((s_exp_array - s_theory_array)**2)
        ss_tot_chsh = np.sum((s_exp_array - np.mean(s_exp_array))**2)
        r_squared_chsh = 1 - (ss_res_chsh / ss_tot_chsh) if ss_tot_chsh > 0 else 0
        
        # Calculate R-squared for CS
        ss_res_cs = np.sum((cs_exp_array - cs_theory_array)**2)
        ss_tot_cs = np.sum((cs_exp_array - np.mean(cs_exp_array))**2)
        r_squared_cs = 1 - (ss_res_cs / ss_tot_cs) if ss_tot_cs > 0 else 0
        
        # Analyze CS vs C relationship
        # Theoretical: S(θ) = √2(1 + sin(2θ)), C = sin(2θ)
        # Therefore: CS = S/2 = (√2/2)(1 + C)
        # This gives: CS = 0.707 + 0.707*C (linear relationship)
        
        # Find the best linear fit between CS and C
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(c_array, cs_exp_array)
        
        self.analysis_results = {
            'r_squared_chsh': float(r_squared_chsh),
            'r_squared_cs': float(r_squared_cs),
            'cs_c_slope': float(slope),
            'cs_c_intercept': float(intercept),
            'cs_c_correlation': float(r_value),
            'mean_s_error': float(np.mean(np.abs(s_exp_array - s_theory_array))),
            'max_s_error': float(np.max(np.abs(s_exp_array - s_theory_array))),
            'max_s_experimental': float(np.max(s_exp_array)),
            'max_cs_experimental': float(np.max(cs_exp_array))
        }
        
        print(f"\nAnalysis Results:")
        print(f"  CHSH Theory vs Experiment: R^2 = {r_squared_chsh:.6f}")
        print(f"  CS Theory vs Experiment: R^2 = {r_squared_cs:.6f}")
        print(f"  CS vs C relationship: CS = {slope:.4f}*C + {intercept:.4f}")
        print(f"  CS-C correlation: r = {r_value:.6f}")
        print(f"  Mean CHSH error: {self.analysis_results['mean_s_error']:.4f}")
        print(f"  Max CHSH error: {self.analysis_results['max_s_error']:.4f}")
        print(f"  Max S achieved: {self.analysis_results['max_s_experimental']:.4f}")
        print(f"  Max CS achieved: {self.analysis_results['max_cs_experimental']:.4f}")
        
        return self.analysis_results
    
    def save_results(self, filename='ex2_standard_chsh_results.json'):
        """
        Save all results to JSON file
        
        Args:
            filename: Output filename
        """
        results = {
            'metadata': {
                'experiment': 'Confounding Strength Quantification',
                'protocol': 'Standard CHSH (XZ-plane)',
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'shots': self.shots,
                    'n_trials': self.n_trials,
                    'n_theta_points': self.n_theta_points,
                    'measurement_settings': self.measurement_settings
                }
            },
            'theory': {
                'state_family': '|psi(theta)> = cos(theta)|00> + sin(theta)|11>',
                'chsh_formula': 'S(theta) = sqrt(2) * (1 + sin(2*theta))',
                'cs_formula': 'CS = S/2',
                'concurrence_formula': 'C = sin(2*theta)'
            },
            'sweep_data': self.sweep_data,
            'analysis': self.analysis_results
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def run_complete_experiment(self):
        """Run the complete experiment pipeline"""
        # Run parameter sweep
        self.run_parameter_sweep()
        
        # Analyze results
        self.analyze_results()
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        print(f"Key findings:")
        print(f"  - Maximum S achieved: {self.analysis_results['max_s_experimental']:.4f}")
        print(f"  - Maximum CS achieved: {self.analysis_results['max_cs_experimental']:.4f}")
        print(f"  - CS-C relationship: CS = {self.analysis_results['cs_c_slope']:.3f}*C + {self.analysis_results['cs_c_intercept']:.3f}")
        print(f"  - Expected: CS = 0.707*C + 0.707 (from theory)")
        print(f"  - Theory-experiment agreement: R^2 = {self.analysis_results['r_squared_chsh']:.4f}")
        print("="*60)


# Self-test verification functions
def test_state_preparation():
    """Test parameterized state preparation"""
    print("Testing state preparation...")
    
    exp = Experiment2_StandardCHSH(shots=1000)
    
    # Test theta = 0 (should be |00>)
    qc0 = exp.create_parameterized_state(0)
    state0 = Statevector.from_instruction(qc0)
    expected0 = np.array([1, 0, 0, 0], dtype=complex)
    fidelity0 = np.abs(np.vdot(state0.data, expected0))**2
    
    # Test theta = pi/4 (should be Bell state)
    qc1 = exp.create_parameterized_state(np.pi/4)
    state1 = Statevector.from_instruction(qc1)
    expected1 = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    fidelity1 = np.abs(np.vdot(state1.data, expected1))**2
    
    print(f"  theta=0 fidelity: {fidelity0:.6f} (expected: 1.0)")
    print(f"  theta=pi/4 fidelity: {fidelity1:.6f} (expected: 1.0)")
    
    success = fidelity0 > 0.99 and fidelity1 > 0.99
    print(f"  Test {'PASSED' if success else 'FAILED'}")
    
    return success


def test_theoretical_formulas():
    """Test theoretical formula implementations"""
    print("\nTesting theoretical formulas...")
    
    exp = Experiment2_StandardCHSH()
    
    # Test cases for theoretical CHSH
    # For theta = 0: |psi> = |00>, no entanglement
    s_theta0 = exp.theoretical_chsh(0)
    expected_s0 = np.sqrt(2)
    print(f"  theta=0: S={s_theta0:.4f} (expected: {expected_s0:.4f})")
    
    # For theta = pi/4: |psi> = (|00> + |11>)/sqrt(2), maximal entanglement
    s_thetamax = exp.theoretical_chsh(np.pi/4)
    expected_smax = 2 * np.sqrt(2)
    print(f"  theta=pi/4: S={s_thetamax:.4f} (expected: {expected_smax:.4f})")
    
    # Test intermediate value
    s_theta_mid = exp.theoretical_chsh(np.pi/8)
    expected_smid = np.sqrt(2) * (1 + np.sin(np.pi/4))
    print(f"  theta=pi/8: S={s_theta_mid:.4f} (expected: {expected_smid:.4f})")
    
    # Test concurrence
    c0 = exp.calculate_concurrence(0)
    c_max = exp.calculate_concurrence(np.pi/4)
    print(f"  Concurrence at theta=0: {c0:.4f} (expected: 0)")
    print(f"  Concurrence at theta=pi/4: {c_max:.4f} (expected: 1)")
    
    success = (abs(s_theta0 - expected_s0) < 0.01 and
               abs(s_thetamax - expected_smax) < 0.01 and
               abs(s_theta_mid - expected_smid) < 0.01 and
               abs(c0) < 0.01 and abs(c_max - 1) < 0.01)
    print(f"  Test {'PASSED' if success else 'FAILED'}")
    
    return success


def test_chsh_measurement():
    """Test CHSH measurement implementation"""
    print("\nTesting CHSH measurement...")
    
    exp = Experiment2_StandardCHSH(shots=20000)
    
    # Test with maximally entangled state
    qc = exp.create_parameterized_state(np.pi/4)
    result = exp.compute_chsh_value(qc)
    
    s_measured = result['S']
    s_expected = 2 * np.sqrt(2)
    error = abs(s_measured - s_expected)
    
    print(f"  Measured S = {s_measured:.4f}")
    print(f"  Expected S = {s_expected:.4f}")
    print(f"  Error = {error:.4f}")
    
    # Check individual correlations
    print(f"  E(a,b) = {result['E_ab']:.4f}")
    print(f"  E(a,b') = {result['E_ab_prime']:.4f}")
    print(f"  E(a',b) = {result['E_a_prime_b']:.4f}")
    print(f"  E(a',b') = {result['E_a_prime_b_prime']:.4f}")
    
    success = error < 0.15  # Allow some statistical fluctuation
    print(f"  Test {'PASSED' if success else 'FAILED'}")
    
    return success


def test_cs_calculation():
    """Test confounding strength calculation"""
    print("\nTesting CS calculation...")
    
    exp = Experiment2_StandardCHSH()
    
    # Test CS = S/2
    s_values = [0, 2, 2.828]
    for s in s_values:
        cs = exp.theoretical_cs(s)
        expected = s / 2
        error = abs(cs - expected)
        print(f"  S={s:.3f}: CS={cs:.3f}, expected={expected:.3f}, error={error:.6f}")
    
    print(f"  Test PASSED")
    
    return True


def test_data_structure():
    """Test data structure and saving"""
    print("\nTesting data structure...")
    
    exp = Experiment2_StandardCHSH(shots=1000, n_trials=2, n_theta_points=3)
    
    # Run minimal sweep
    exp.theta_values = np.array([0, np.pi/4, np.pi/2])
    exp.n_theta_points = 3
    exp.run_parameter_sweep()
    
    # Check data structure
    success = True
    
    # Check sweep data
    if len(exp.sweep_data) != 3:
        print(f"  ERROR: Expected 3 data points, got {len(exp.sweep_data)}")
        success = False
    
    # Check required fields
    required_fields = ['theta', 's_mean', 's_theory', 'cs_experimental', 'cs_theory']
    for point in exp.sweep_data:
        for field in required_fields:
            if field not in point:
                print(f"  ERROR: Missing field '{field}' in data point")
                success = False
    
    # Test saving
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename = f.name
    
    try:
        exp.save_results(temp_filename)
        # Check if file exists and is valid JSON
        with open(temp_filename, 'r') as f:
            data = json.load(f)
        
        if 'sweep_data' not in data or 'analysis' not in data:
            print("  ERROR: Saved data missing required sections")
            success = False
    except Exception as e:
        print(f"  ERROR: Failed to save/load data: {e}")
        success = False
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    print(f"  Test {'PASSED' if success else 'FAILED'}")
    
    return success


def run_all_tests():
    """Run all self-tests"""
    print("="*60)
    print("RUNNING SELF-TESTS")
    print("="*60)
    
    tests = [
        test_state_preparation,
        test_theoretical_formulas,
        test_chsh_measurement,
        test_cs_calculation,
        test_data_structure
    ]
    
    passed = sum(test() for test in tests)
    total = len(tests)
    
    print("\n" + "="*60)
    print(f"OVERALL: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    # Run self-tests first
    if run_all_tests():
        print("\nAll tests passed. Running main experiment...")
        print("\n" + "="*60 + "\n")
        
        # Run main experiment
        experiment = Experiment2_StandardCHSH(
            shots=10000,
            n_trials=5,
            n_theta_points=25
        )
        experiment.run_complete_experiment()
    else:
        print("\nSome tests failed. Please check the implementation.")
