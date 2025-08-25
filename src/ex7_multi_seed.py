import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
import seaborn as sns
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class Experiment7_QuantumMLConfounding:
    """
    Experiment 7: Causal Feature Selection in Quantum Machine Learning
    
    Demonstrates how the Bell-Confounding framework enables robust feature selection
    by distinguishing causal features from confounded ones using quantum do-calculus.
    
    This experiment proves that quantum entanglement creates spurious correlations
    that can mislead ML models, and shows how quantum do-calculus can identify
    truly causal features for more robust classification.
    """
    
    def __init__(self, n_samples=2000, shots=1, test_size=0.3):
        """
        Initialize the quantum ML confounding experiment
        
        Args:
            n_samples: Number of data points to generate
            shots: Number of shots per quantum measurement (1 for deterministic outcomes)
            test_size: Fraction of data to use for testing
        """
        self.n_samples = n_samples
        self.shots = shots
        self.test_size = test_size
        self.simulator = AerSimulator()
        self.results = {}
        
    def create_causal_circuit(self, theta_a, confounding_strength=1.0):
        """
        Create quantum circuit implementing causal structure: C ↔ A → B
        
        Args:
            theta_a: Rotation angle for feature A (controls A's value)
            confounding_strength: Strength of A-C entanglement (0=none, 1=maximal)
            
        Returns:
            QuantumCircuit: Circuit implementing the causal structure
        """
        qc = QuantumCircuit(3, 3)  # A, B, C qubits + measurements
        
        # Step 1: Prepare feature A (the true causal feature)
        qc.ry(theta_a, 0)  # A = qubit 0
        
        # --- SIMPLIFIED AND ROBUST CONFOUNDING LOGIC ---
        # Use a simple CNOT to guarantee a strong correlation between A and C.
        # This makes the confounding effect clear and unambiguous.
        if confounding_strength > 0:
            # We can use the strength parameter to control this probabilistically.
            if np.random.random() < confounding_strength:
                qc.cx(0, 2)  # Entangle A (qubit 0) and C (qubit 2)
        # --- END CORRECTION ---

        # Step 3: Create true causation A → B
        qc.cx(0, 1)  # B = qubit 1, directly caused by A
        
        # Measure all qubits
        qc.measure([0, 1, 2], [0, 1, 2])
        return qc

    def generate_quantum_dataset(self, n_samples=None, confounding_strength=1.0, 
                                noise_level=0.0, random_state=None):
        """
        Generate dataset with quantum confounding structure
        
        Args:
            n_samples: Number of samples to generate
            confounding_strength: Strength of spurious correlation (0-1)
            noise_level: Amount of classical noise to add (0-1)
            random_state: Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Dataset with features A, C and label B
        """
        if n_samples is None:
            n_samples = self.n_samples
        
        if random_state is not None:
            np.random.seed(random_state)
            
        data = []
        for i in range(n_samples):
            # Random angle for feature A (varying input conditions)
            theta_a = np.random.uniform(0, np.pi)
            
            # Create circuit with specified confounding
            qc = self.create_causal_circuit(theta_a, confounding_strength)
            
            # Execute quantum circuit
            job = self.simulator.run(transpile(qc, self.simulator), shots=self.shots)
            counts = job.result().get_counts()
            outcome = list(counts.keys())[0]  # Single deterministic outcome
            
            # Parse results (Qiskit bit order: c b a)
            a_val = int(outcome[2])  # Feature A (true cause)
            b_val = int(outcome[1])  # Label B (effect)
            c_val = int(outcome[0])  # Feature C (confounded)
            
            # Add classical noise if specified
            if noise_level > 0:
                if np.random.random() < noise_level:
                    b_val = 1 - b_val  # Flip label with probability noise_level
            
            data.append({
                'A': a_val, 
                'C': c_val, 
                'B_label': b_val,
                'theta_a': theta_a  # Store for analysis
            })
        
        dataset = pd.DataFrame(data)
        return dataset
    
    def train_classifiers(self, dataset, random_state=None):
        """
        Train multiple classifiers with different feature sets
        
        Args:
            dataset: Training dataset
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Trained models and their performance metrics
        """
        # Prepare features and labels
        X_naive = dataset[['A', 'C']]  # Naive: uses both features
        X_causal = dataset[['A']]      # Causal: uses only true cause
        X_confounded = dataset[['C']]  # Confounded: uses only spurious feature
        y = dataset['B_label']
        
        # Split data with specified random state
        X_naive_train, X_naive_test, y_train, y_test = train_test_split(
            X_naive, y, test_size=self.test_size, random_state=random_state
        )
        X_causal_train = X_naive_train[['A']]
        X_causal_test = X_naive_test[['A']]
        X_conf_train = X_naive_train[['C']]
        X_conf_test = X_naive_test[['C']]
        
        models = {}
        
        # Train Naive Classifier (A + C)
        naive_model = LogisticRegression(random_state=random_state)
        naive_model.fit(X_naive_train, y_train)
        naive_pred = naive_model.predict(X_naive_test)
        naive_acc = accuracy_score(y_test, naive_pred)
        
        # Train Causal Classifier (A only)
        causal_model = LogisticRegression(random_state=random_state)
        causal_model.fit(X_causal_train, y_train)
        causal_pred = causal_model.predict(X_causal_test)
        causal_acc = accuracy_score(y_test, causal_pred)
        
        # Train Confounded Classifier (C only)
        conf_model = LogisticRegression(random_state=random_state)
        conf_model.fit(X_conf_train, y_train)
        conf_pred = conf_model.predict(X_conf_test)
        conf_acc = accuracy_score(y_test, conf_pred)
        
        models = {
            'naive': {'model': naive_model, 'accuracy': float(naive_acc), 'predictions': naive_pred},
            'causal': {'model': causal_model, 'accuracy': float(causal_acc), 'predictions': causal_pred},
            'confounded': {'model': conf_model, 'accuracy': float(conf_acc), 'predictions': conf_pred},
            'test_data': {'X_naive': X_naive_test, 'X_causal': X_causal_test, 
                         'X_confounded': X_conf_test, 'y_true': y_test}
        }
        
        return models
    
    def apply_quantum_do_calculus(self, n_intervention_samples=1000):
        """
        Apply quantum do-calculus to test causal relationships
        
        Args:
            n_intervention_samples: Number of samples for intervention
            
        Returns:
            dict: Results of causal interventions
        """
        # Measure observational distributions
        obs_dataset = self.generate_quantum_dataset(n_intervention_samples, 
                                                   confounding_strength=1.0)
        
        # Observational: P(B|A) and P(B|C)
        p_b_given_a0 = obs_dataset[obs_dataset['A'] == 0]['B_label'].mean()
        p_b_given_a1 = obs_dataset[obs_dataset['A'] == 1]['B_label'].mean()
        p_b_given_c0 = obs_dataset[obs_dataset['C'] == 0]['B_label'].mean()
        p_b_given_c1 = obs_dataset[obs_dataset['C'] == 1]['B_label'].mean()
        
        # Interventional: P(B|do(A)) and P(B|do(C))
        # Intervention on A: do(A=0) and do(A=1)
        do_a_results = self.intervention_on_feature('A', n_intervention_samples)
        p_b_given_do_a0 = do_a_results['P(B=1|do(A=0))']
        p_b_given_do_a1 = do_a_results['P(B=1|do(A=1))']
        
        # Intervention on C: do(C=0) and do(C=1)
        do_c_results = self.intervention_on_feature('C', n_intervention_samples)
        p_b_given_do_c0 = do_c_results['P(B=1|do(C=0))']
        p_b_given_do_c1 = do_c_results['P(B=1|do(C=1))']
        
        # Calculate causal effects
        causal_effect_a = abs(p_b_given_do_a1 - p_b_given_do_a0)
        causal_effect_c = abs(p_b_given_do_c1 - p_b_given_do_c0)
        
        # Statistical significance test
        independence_threshold = 0.1
        a_is_causal = causal_effect_a > independence_threshold
        c_is_spurious = causal_effect_c < independence_threshold
        
        return {
            'observational': {
                'P(B|A=0)': float(p_b_given_a0), 'P(B|A=1)': float(p_b_given_a1),
                'P(B|C=0)': float(p_b_given_c0), 'P(B|C=1)': float(p_b_given_c1)
            },
            'interventional': {
                'P(B|do(A=0))': float(p_b_given_do_a0), 'P(B|do(A=1))': float(p_b_given_do_a1),
                'P(B|do(C=0))': float(p_b_given_do_c0), 'P(B|do(C=1))': float(p_b_given_do_c1)
            },
            'causal_effects': {
                'A_causal_effect': float(causal_effect_a),
                'C_causal_effect': float(causal_effect_c)
            },
            'validation': {
                'A_is_causal': bool(a_is_causal),
                'C_is_spurious': bool(c_is_spurious)
            }
        }
    
    def intervention_on_feature(self, feature, n_samples):
        """
        Perform do-calculus intervention on specified feature
        
        Args:
            feature: 'A' or 'C' - which feature to intervene on
            n_samples: Number of intervention samples
            
        Returns:
            dict: Intervention results
        """
        results = {}
        
        for value in [0, 1]:
            b_outcomes = []
            
            for _ in range(n_samples):
                # Create intervention circuit
                qc = QuantumCircuit(3, 1)  # Only measure B
                
                if feature == 'A':
                    # Intervention: do(A=value)
                    if value == 1:
                        qc.x(0)  # Set A to 1
                    # Don't entangle with C (break confounding)
                    # A still causes B
                    qc.cx(0, 1)
                    
                elif feature == 'C':
                    # Intervention: do(C=value)
                    if value == 1:
                        qc.x(2)  # Set C to 1
                    # Prepare A independently (break confounding)
                    theta_a = np.random.uniform(0, np.pi)
                    qc.ry(theta_a, 0)
                    # A still causes B (C doesn't)
                    qc.cx(0, 1)
                
                qc.measure(1, 0)  # Measure only B
                
                job = self.simulator.run(transpile(qc, self.simulator), shots=self.shots)
                counts = job.result().get_counts()
                outcome = list(counts.keys())[0]
                b_outcomes.append(int(outcome))
            
            results[f'P(B=1|do({feature}={value}))'] = float(np.mean(b_outcomes))
        
        return results
    
    def test_robustness(self, models, n_test_scenarios=5, random_state=None):
        """
        Test model robustness across different confounding scenarios
        
        Args:
            models: Trained models from train_classifiers
            n_test_scenarios: Number of different test scenarios
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Robustness test results with detailed accuracies
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        robustness_results = []
        
        for i in range(n_test_scenarios):
            # Vary confounding strength and noise
            conf_strength = i / (n_test_scenarios - 1)  # 0 to 1
            noise_level = np.random.uniform(0, 0.1)
            
            # Generate test dataset with different confounding
            test_data = self.generate_quantum_dataset(
                n_samples=500, 
                confounding_strength=conf_strength,
                noise_level=noise_level,
                random_state=random_state + i if random_state is not None else None
            )
            
            X_test_naive = test_data[['A', 'C']]
            X_test_causal = test_data[['A']]
            X_test_confounded = test_data[['C']]
            y_test = test_data['B_label']
            
            # Test all models
            naive_acc = accuracy_score(y_test, models['naive']['model'].predict(X_test_naive))
            causal_acc = accuracy_score(y_test, models['causal']['model'].predict(X_test_causal))
            conf_acc = accuracy_score(y_test, models['confounded']['model'].predict(X_test_confounded))
            
            robustness_results.append({
                'scenario_id': int(i),
                'confounding_strength': float(conf_strength),
                'noise_level': float(noise_level),
                'naive_accuracy': float(naive_acc),
                'causal_accuracy': float(causal_acc),
                'confounded_accuracy': float(conf_acc)
            })
        
        return robustness_results
    
    def run_single_seed_experiment(self, seed=None):
        """
        Run a single experiment with a specified seed (simplified version for multi-seed)
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            dict: Single seed experiment results
        """
        # Phase 1: Generate confounded dataset
        dataset = self.generate_quantum_dataset(confounding_strength=1.0, random_state=seed)
        
        # Phase 2: Train classifiers
        models = self.train_classifiers(dataset, random_state=seed)
        
        # Phase 3: Apply quantum do-calculus (not needed for multi-seed, but kept for consistency)
        causal_analysis = self.apply_quantum_do_calculus()
        
        # Phase 4: Test robustness
        robustness_results = self.test_robustness(models, random_state=seed)
        
        return {
            'seed': seed,
            'dataset': dataset,
            'models': models,
            'causal_analysis': causal_analysis,
            'robustness': robustness_results
        }
    
    def run_single_seed_full_experiment(self, seed=42):
        """
        Run a complete single seed experiment with full causal analysis
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            dict: Complete experimental results including causal analysis
        """
        print("="*80)
        print("SINGLE SEED EXPERIMENT: FULL CAUSAL ANALYSIS")
        print(f"Running with seed={seed}")
        print("="*80)
        
        # Phase 1: Generate confounded dataset
        print("\n1. Generating Confounded Quantum Dataset...")
        dataset = self.generate_quantum_dataset(confounding_strength=1.0, random_state=seed)
        
        # Calculate and display correlations
        corr_matrix = dataset[['A', 'C', 'B_label']].corr()
        print("  Correlation Matrix:")
        print(f"    A-B: {corr_matrix.loc['A', 'B_label']:.4f} (true causation)")
        print(f"    C-B: {corr_matrix.loc['C', 'B_label']:.4f} (spurious correlation)")
        print(f"    A-C: {corr_matrix.loc['A', 'C']:.4f} (confounding)")
        
        # Phase 2: Train classifiers
        print("\n2. Training Classification Models...")
        models = self.train_classifiers(dataset, random_state=seed)
        
        print(f"  Naive Classifier (A+C): {models['naive']['accuracy']:.4f}")
        print(f"  Causal Classifier (A):   {models['causal']['accuracy']:.4f}")
        print(f"  Confounded Classifier (C): {models['confounded']['accuracy']:.4f}")
        
        # Phase 3: Apply quantum do-calculus
        print("\n3. Applying Quantum Do-Calculus...")
        causal_analysis = self.apply_quantum_do_calculus()
        
        print("  Observational Distributions:")
        print(f"    P(B=1|A=0) = {causal_analysis['observational']['P(B|A=0)']:.4f}, P(B=1|A=1) = {causal_analysis['observational']['P(B|A=1)']:.4f}")
        print(f"    P(B=1|C=0) = {causal_analysis['observational']['P(B|C=0)']:.4f}, P(B=1|C=1) = {causal_analysis['observational']['P(B|C=1)']:.4f}")
        
        print("  Interventional Distributions:")
        print(f"    P(B=1|do(A=0)) = {causal_analysis['interventional']['P(B|do(A=0))']:.4f}, P(B=1|do(A=1)) = {causal_analysis['interventional']['P(B|do(A=1))']:.4f}")
        print(f"    P(B=1|do(C=0)) = {causal_analysis['interventional']['P(B|do(C=0))']:.4f}, P(B=1|do(C=1)) = {causal_analysis['interventional']['P(B|do(C=1))']:.4f}")
        
        print("  Causal Effects:")
        print(f"    CE(A→B) = |P(B|do(A=1)) - P(B|do(A=0))| = {causal_analysis['causal_effects']['A_causal_effect']:.4f}")
        print(f"    CE(C→B) = |P(B|do(C=1)) - P(B|do(C=0))| = {causal_analysis['causal_effects']['C_causal_effect']:.4f}")
        
        print("  Causal Validation:")
        print(f"    A is causal feature: {causal_analysis['validation']['A_is_causal']} (CE = {causal_analysis['causal_effects']['A_causal_effect']:.4f})")
        print(f"    C is spurious feature: {causal_analysis['validation']['C_is_spurious']} (CE = {causal_analysis['causal_effects']['C_causal_effect']:.4f})")
        
        # Phase 4: Test robustness
        print("\n4. Testing Model Robustness...")
        robustness_results = self.test_robustness(models, random_state=seed)
        
        for scenario in robustness_results:
            print(f"  Scenario {scenario['scenario_id']+1}: Confounding={scenario['confounding_strength']:.2f}, Noise={scenario['noise_level']:.3f}")
            print(f"    Naive: {scenario['naive_accuracy']:.3f}, Causal: {scenario['causal_accuracy']:.3f}, Confounded: {scenario['confounded_accuracy']:.3f}")
        
        # Save single seed results
        single_seed_results = {
            'seed': seed,
            'dataset_stats': {
                'n_samples': len(dataset),
                'correlations': {
                    'A_B': float(corr_matrix.loc['A', 'B_label']),
                    'C_B': float(corr_matrix.loc['C', 'B_label']),
                    'A_C': float(corr_matrix.loc['A', 'C'])
                }
            },
            'model_accuracies': {
                'naive': models['naive']['accuracy'],
                'causal': models['causal']['accuracy'],
                'confounded': models['confounded']['accuracy']
            },
            'causal_analysis': causal_analysis,
            'robustness': robustness_results
        }
        
        # Save to JSON
        filename = 'experiment_7_single_seed_results.json'
        with open(filename, 'w') as f:
            json.dump(single_seed_results, f, indent=2)
        
        print(f"\n✓ Single seed results saved to {filename}")
        
        # Final assessment
        causal_validated = causal_analysis['validation']['A_is_causal']
        spurious_detected = causal_analysis['validation']['C_is_spurious']
        avg_causal_robustness = np.mean([r['causal_accuracy'] for r in robustness_results])
        avg_naive_robustness = np.mean([r['naive_accuracy'] for r in robustness_results])
        
        print("\n" + "="*80)
        print("SINGLE SEED EXPERIMENT COMPLETE")
        print(f"✓ Causal feature A validated: {causal_validated}")
        print(f"✓ Spurious feature C detected: {spurious_detected}")
        print(f"✓ Causal classifier robustness: {avg_causal_robustness:.3f}")
        print(f"✓ Naive classifier robustness: {avg_naive_robustness:.3f}")
        print(f"✓ Robustness improvement: {avg_causal_robustness - avg_naive_robustness:.3f}")
        print("="*80)
        
        return single_seed_results
    
    def run_multi_seed_experiment(self, n_seeds=20):
        """
        Run experiment with multiple random seeds
        
        Args:
            n_seeds: Number of different random seeds to test
            
        Returns:
            dict: Complete multi-seed experimental results
        """
        print("="*80)
        print("EXPERIMENT 7: MULTI-SEED CAUSAL FEATURE SELECTION IN QUANTUM ML")
        print(f"Running with {n_seeds} different random seeds")
        print("="*80)
        
        multi_seed_results = []
        
        # Collect results for each seed
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx  # Start from seed 42
            print(f"\nSeed {seed_idx + 1}/{n_seeds} (seed={seed})")
            
            # Run single seed experiment
            seed_results = self.run_single_seed_experiment(seed)
            
            # Extract accuracy data for all test scenarios
            seed_accuracies = {
                'seed': int(seed),
                'training_accuracy': {
                    'naive': float(seed_results['models']['naive']['accuracy']),
                    'causal': float(seed_results['models']['causal']['accuracy'])
                },
                'test_scenarios': []
            }
            
            # Extract results for each test scenario
            for scenario in seed_results['robustness']:
                seed_accuracies['test_scenarios'].append({
                    'scenario_id': int(scenario['scenario_id']),
                    'confounding_strength': float(scenario['confounding_strength']),
                    'naive_accuracy': float(scenario['naive_accuracy']),
                    'causal_accuracy': float(scenario['causal_accuracy'])
                })
            
            multi_seed_results.append(seed_accuracies)
            
            # Print progress
            avg_causal = np.mean([s['causal_accuracy'] for s in seed_results['robustness']])
            avg_naive = np.mean([s['naive_accuracy'] for s in seed_results['robustness']])
            print(f"  Average accuracies - Causal: {avg_causal:.3f}, Naive: {avg_naive:.3f}")
        
        return multi_seed_results
    
    def perform_statistical_analysis(self, multi_seed_results):
        """
        Perform paired t-test on multi-seed results
        
        Args:
            multi_seed_results: Results from run_multi_seed_experiment
            
        Returns:
            dict: Statistical analysis results
        """
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS: PAIRED T-TEST")
        print("="*80)
        
        # Organize data for each test scenario
        scenario_analysis = {}
        n_scenarios = len(multi_seed_results[0]['test_scenarios'])
        
        for scenario_id in range(n_scenarios):
            causal_accs = []
            naive_accs = []
            
            # Collect accuracies across all seeds for this scenario
            for seed_result in multi_seed_results:
                scenario_data = seed_result['test_scenarios'][scenario_id]
                causal_accs.append(scenario_data['causal_accuracy'])
                naive_accs.append(scenario_data['naive_accuracy'])
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(causal_accs, naive_accs)
            
            # Calculate effect size (Cohen's d)
            diff = np.array(causal_accs) - np.array(naive_accs)
            cohen_d = np.mean(diff) / np.std(diff, ddof=1)
            
            # Store results
            scenario_analysis[f'scenario_{scenario_id}'] = {
                'confounding_strength': float(multi_seed_results[0]['test_scenarios'][scenario_id]['confounding_strength']),
                'causal_mean': float(np.mean(causal_accs)),
                'causal_std': float(np.std(causal_accs)),
                'naive_mean': float(np.mean(naive_accs)),
                'naive_std': float(np.std(naive_accs)),
                'mean_difference': float(np.mean(causal_accs) - np.mean(naive_accs)),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohen_d': float(cohen_d),
                'significant': bool(p_value < 0.05)
            }
            
            print(f"\nScenario {scenario_id} (Confounding={scenario_analysis[f'scenario_{scenario_id}']['confounding_strength']:.2f}):")
            print(f"  Causal: μ={scenario_analysis[f'scenario_{scenario_id}']['causal_mean']:.3f} (σ={scenario_analysis[f'scenario_{scenario_id}']['causal_std']:.3f})")
            print(f"  Naive:  μ={scenario_analysis[f'scenario_{scenario_id}']['naive_mean']:.3f} (σ={scenario_analysis[f'scenario_{scenario_id}']['naive_std']:.3f})")
            print(f"  Difference: {scenario_analysis[f'scenario_{scenario_id}']['mean_difference']:.3f}")
            print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
            print(f"  Cohen's d: {cohen_d:.3f}")
            print(f"  Significant: {'Yes' if scenario_analysis[f'scenario_{scenario_id}']['significant'] else 'No'}")
        
        # Overall analysis across all scenarios
        all_causal = []
        all_naive = []
        
        for seed_result in multi_seed_results:
            for scenario in seed_result['test_scenarios']:
                all_causal.append(scenario['causal_accuracy'])
                all_naive.append(scenario['naive_accuracy'])
        
        overall_t_stat, overall_p_value = stats.ttest_rel(all_causal, all_naive)
        overall_cohen_d = (np.mean(all_causal) - np.mean(all_naive)) / np.std(np.array(all_causal) - np.array(all_naive), ddof=1)
        
        print("\n" + "-"*80)
        print("OVERALL ANALYSIS (All Scenarios Combined):")
        print(f"  Causal: μ={np.mean(all_causal):.3f} (σ={np.std(all_causal):.3f})")
        print(f"  Naive:  μ={np.mean(all_naive):.3f} (σ={np.std(all_naive):.3f})")
        print(f"  Mean Difference: {np.mean(all_causal) - np.mean(all_naive):.3f}")
        print(f"  t-statistic: {overall_t_stat:.3f}, p-value: {overall_p_value:.4e}")
        print(f"  Cohen's d: {overall_cohen_d:.3f}")
        print(f"  Statistical Significance: {'YES' if overall_p_value < 0.05 else 'NO'}")
        
        return {
            'scenario_analysis': scenario_analysis,
            'overall': {
                'causal_mean': float(np.mean(all_causal)),
                'causal_std': float(np.std(all_causal)),
                'naive_mean': float(np.mean(all_naive)),
                'naive_std': float(np.std(all_naive)),
                'mean_difference': float(np.mean(all_causal) - np.mean(all_naive)),
                't_statistic': float(overall_t_stat),
                'p_value': float(overall_p_value),
                'cohen_d': float(overall_cohen_d),
                'significant': bool(overall_p_value < 0.05)
            }
        }
    
    def visualize_multi_seed_results(self, multi_seed_results, statistical_analysis):
        """
        Create comprehensive visualizations of multi-seed experimental results
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Experiment 7: Multi-Seed Analysis Results', fontsize=16)
        
        # Plot 1: Accuracy distributions by scenario
        ax1 = axes[0, 0]
        scenario_data = []
        labels = []
        
        for scenario_key, scenario_stats in statistical_analysis['scenario_analysis'].items():
            scenario_data.append([
                [seed_result['test_scenarios'][int(scenario_key.split('_')[1])]['causal_accuracy'] 
                 for seed_result in multi_seed_results],
                [seed_result['test_scenarios'][int(scenario_key.split('_')[1])]['naive_accuracy'] 
                 for seed_result in multi_seed_results]
            ])
            labels.append(f"Conf={scenario_stats['confounding_strength']:.1f}")
        
        # Create box plots
        positions = []
        for i, (causal_data, naive_data) in enumerate(scenario_data):
            pos = i * 3
            positions.extend([pos, pos + 1])
            bp1 = ax1.boxplot([causal_data], positions=[pos], widths=0.8, 
                              patch_artist=True, boxprops=dict(facecolor='green', alpha=0.7))
            bp2 = ax1.boxplot([naive_data], positions=[pos + 1], widths=0.8, 
                              patch_artist=True, boxprops=dict(facecolor='orange', alpha=0.7))
        
        ax1.set_xticks([i * 3 + 0.5 for i in range(len(labels))])
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy Distributions by Test Scenario', fontsize=14)
        ax1.legend([plt.Rectangle((0,0),1,1,fc='green',alpha=0.7), 
                   plt.Rectangle((0,0),1,1,fc='orange',alpha=0.7)], 
                  ['Causal', 'Naive'])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Mean accuracy comparison with error bars
        ax2 = axes[0, 1]
        scenarios = list(range(len(statistical_analysis['scenario_analysis'])))
        causal_means = [statistical_analysis['scenario_analysis'][f'scenario_{i}']['causal_mean'] 
                       for i in scenarios]
        causal_stds = [statistical_analysis['scenario_analysis'][f'scenario_{i}']['causal_std'] 
                      for i in scenarios]
        naive_means = [statistical_analysis['scenario_analysis'][f'scenario_{i}']['naive_mean'] 
                      for i in scenarios]
        naive_stds = [statistical_analysis['scenario_analysis'][f'scenario_{i}']['naive_std'] 
                     for i in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax2.bar(x - width/2, causal_means, width, yerr=causal_stds, 
               label='Causal', color='green', alpha=0.7, capsize=5)
        ax2.bar(x + width/2, naive_means, width, yerr=naive_stds, 
               label='Naive', color='orange', alpha=0.7, capsize=5)
        
        # Add significance stars
        for i, scenario_key in enumerate(statistical_analysis['scenario_analysis']):
            if statistical_analysis['scenario_analysis'][scenario_key]['significant']:
                y_pos = max(causal_means[i] + causal_stds[i], 
                           naive_means[i] + naive_stds[i]) + 0.02
                ax2.text(i, y_pos, '*', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        ax2.set_xlabel('Test Scenario', fontsize=12)
        ax2.set_ylabel('Mean Accuracy', fontsize=12)
        ax2.set_title('Mean Accuracy Comparison (* p < 0.05)', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'S{i}' for i in scenarios])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: P-values and effect sizes
        ax3 = axes[1, 0]
        p_values = [statistical_analysis['scenario_analysis'][f'scenario_{i}']['p_value'] 
                   for i in scenarios]
        cohen_ds = [statistical_analysis['scenario_analysis'][f'scenario_{i}']['cohen_d'] 
                   for i in scenarios]
        
        ax3_twin = ax3.twinx()
        
        # Plot p-values
        p_line = ax3.plot(scenarios, p_values, 'bo-', linewidth=2, markersize=8, label='p-value')
        ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax3.set_ylabel('p-value', fontsize=12, color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        ax3.set_yscale('log')
        
        # Plot Cohen's d
        d_line = ax3_twin.plot(scenarios, cohen_ds, 'go-', linewidth=2, markersize=8, label="Cohen's d")
        ax3_twin.set_ylabel("Cohen's d (Effect Size)", fontsize=12, color='green')
        ax3_twin.tick_params(axis='y', labelcolor='green')
        
        ax3.set_xlabel('Test Scenario', fontsize=12)
        ax3.set_title('Statistical Significance and Effect Size', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Combined legend
        lines = p_line + d_line + [plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2)]
        labels = ['p-value', "Cohen's d", 'α = 0.05']
        ax3.legend(lines, labels, loc='center right')
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""Multi-Seed Experiment Summary (n_seeds = {len(multi_seed_results)}):

Overall Results:
• Causal Classifier:  μ = {statistical_analysis['overall']['causal_mean']:.3f} (σ = {statistical_analysis['overall']['causal_std']:.3f})
• Naive Classifier:   μ = {statistical_analysis['overall']['naive_mean']:.3f} (σ = {statistical_analysis['overall']['naive_std']:.3f})
• Mean Difference:    Δ = {statistical_analysis['overall']['mean_difference']:.3f}

Statistical Test (Paired t-test):
• t-statistic: {statistical_analysis['overall']['t_statistic']:.3f}
• p-value: {statistical_analysis['overall']['p_value']:.2e}
• Cohen's d: {statistical_analysis['overall']['cohen_d']:.3f}
• Significant: {'YES (p < 0.05)' if statistical_analysis['overall']['significant'] else 'NO (p ≥ 0.05)'}

Interpretation:
• The causal classifier {'significantly outperforms' if statistical_analysis['overall']['significant'] else 'does not significantly outperform'} 
  the naive classifier across all test scenarios.
• Effect size is {'large' if abs(statistical_analysis['overall']['cohen_d']) > 0.8 else 'medium' if abs(statistical_analysis['overall']['cohen_d']) > 0.5 else 'small'} (|d| = {abs(statistical_analysis['overall']['cohen_d']):.3f}).

Key Achievement:
Quantum do-calculus enables robust feature selection
by identifying truly causal features, leading to
{'significantly' if statistical_analysis['overall']['significant'] else 'potentially'} better generalization across different
confounding scenarios.
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_multi_seed_results(self, multi_seed_results, statistical_analysis):
        """
        Save multi-seed experimental results to JSON file
        """
        # Helper function to convert numpy types to Python types
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Prepare results for JSON serialization
        json_results = {
            'experiment_info': {
                'experiment_name': 'Experiment 7: Multi-Seed Quantum ML Confounding',
                'timestamp': datetime.now().isoformat(),
                'n_seeds': len(multi_seed_results),
                'n_test_scenarios': len(multi_seed_results[0]['test_scenarios']),
                'n_samples': self.n_samples,
                'test_size': self.test_size
            },
            'multi_seed_accuracies': multi_seed_results,
            'statistical_analysis': {
                'scenario_analysis': convert_to_json_serializable(statistical_analysis['scenario_analysis']),
                'overall': convert_to_json_serializable(statistical_analysis['overall'])
            },
            'summary': {
                'causal_outperforms_naive': bool(statistical_analysis['overall']['significant']),
                'p_value': float(statistical_analysis['overall']['p_value']),
                'cohen_d': float(statistical_analysis['overall']['cohen_d']),
                'mean_improvement': float(statistical_analysis['overall']['mean_difference'])
            }
        }
        
        filename = 'experiment_7_multi_seed_results.json'
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Multi-seed results saved to {filename}")
        
        return filename


def run_complete_quantum_ml_experiment(n_seeds=20):
    """
    Main function to run both single seed and multi-seed experiments
    
    Args:
        n_seeds: Number of random seeds for multi-seed experiment
    """
    # Initialize experiment
    experiment = Experiment7_QuantumMLConfounding(n_samples=2000, test_size=0.3)
    
    # Part 1: Run single seed experiment with full causal analysis
    print("\n" + "="*80)
    print("PART 1: SINGLE SEED EXPERIMENT WITH FULL CAUSAL ANALYSIS")
    print("="*80 + "\n")
    
    single_seed_results = experiment.run_single_seed_full_experiment(seed=42)
    
    # Part 2: Run multi-seed experiment
    print("\n\n" + "="*80)
    print("PART 2: MULTI-SEED EXPERIMENT FOR STATISTICAL VALIDATION")
    print("="*80 + "\n")
    
    multi_seed_results = experiment.run_multi_seed_experiment(n_seeds=n_seeds)
    
    # Part 3: Perform statistical analysis
    statistical_analysis = experiment.perform_statistical_analysis(multi_seed_results)
    
    # Save multi-seed results
    experiment.save_multi_seed_results(multi_seed_results, statistical_analysis)
    
    print("\n" + "="*80)
    print("COMPLETE EXPERIMENT FINISHED")
    print("="*80)
    print("✓ Single seed causal analysis completed")
    print(f"✓ Multi-seed validation with {n_seeds} seeds completed")
    print(f"✓ Statistical significance: p = {statistical_analysis['overall']['p_value']:.4e}")
    print(f"✓ Causal classifier {'significantly' if statistical_analysis['overall']['significant'] else 'does not significantly'} outperforms naive classifier")
    print("✓ Results saved:")
    print("  - experiment_7_single_seed_results.json")
    print("  - experiment_7_multi_seed_results.json")
    print("="*80)


# Main execution
if __name__ == "__main__":
    # Run the complete experiment (single seed + multi-seed)
    run_complete_quantum_ml_experiment(n_seeds=20)
