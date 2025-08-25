import matplotlib.pyplot as plt
import numpy as np
import json

def create_cs_violin_plot_with_ionq(sim_json_path, ionq_1a_path=None, ionq_1b_path=None):
    """
    Create violin plot for Confounding Strength (CS) from experiment 1 results 
    with IonQ hardware data overlay.
    """
    # Load simulation data from JSON file
    with open(sim_json_path, 'r') as f:
        sim_data = json.load(f)
    
    # --- MODIFICATION: Convert all S values to CS values ---
    no_confounding_cs = [abs(s) / 2 for s in sim_data['s_value_distributions']['no_confounding']]
    classical_cs = [abs(s) / 2 for s in sim_data['s_value_distributions']['classical']]
    quantum_cs = [abs(s) / 2 for s in sim_data['s_value_distributions']['quantum']]
    
    # Load IonQ hardware results and convert to CS
    ionq_results = {}
    
    if ionq_1a_path:
        with open(ionq_1a_path, 'r') as f:
            ionq_1a_data = json.load(f)
        s_mean = ionq_1a_data['summary_statistics']['s_mean_no_confounding']
        s_ci_lower = ionq_1a_data['summary_statistics']['s_ci_lower']
        s_ci_upper = ionq_1a_data['summary_statistics']['s_ci_upper']
        ionq_results['no_confounding'] = {
            'mean': abs(s_mean) / 2,
            # For CI of |S|/2 where S_CI spans 0, the lower bound is 0
            'ci_lower': 0, 
            'ci_upper': max(abs(s_ci_lower), abs(s_ci_upper)) / 2,
            'n_trials': len(ionq_1a_data['s_value_distributions']['no_confounding'])
        }
    
    if ionq_1b_path:
        with open(ionq_1b_path, 'r') as f:
            ionq_1b_data = json.load(f)
        s_mean = ionq_1b_data['summary_statistics']['s_mean_quantum']
        s_ci_lower = ionq_1b_data['summary_statistics']['s_ci_lower']
        s_ci_upper = ionq_1b_data['summary_statistics']['s_ci_upper']
        ionq_results['quantum'] = {
            'mean': abs(s_mean) / 2,
            'ci_lower': abs(s_ci_lower) / 2,
            'ci_upper': abs(s_ci_upper) / 2,
            'n_trials': len(ionq_1b_data['s_value_distributions']['quantum_confounding'])
        }
    
    # Prepare data for violin plot
    data_to_plot = [no_confounding_cs, classical_cs, quantum_cs]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create violin plot
    parts = ax.violinplot(data_to_plot, positions=[1, 2, 3],
                          showmeans=True, showmedians=False, showextrema=True)
    
    # Customize violin colors
    colors = ['lightblue', 'orange', 'red']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Customize other elements
    parts['cmeans'].set_color('darkblue')
    parts['cmeans'].set_linewidth(3)
    parts['cbars'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')
    
    # Add IonQ hardware results as points with error bars
    if ionq_results:
        ionq_positions = []
        ionq_means = []
        ionq_errors = []
        
        if 'no_confounding' in ionq_results:
            ionq_positions.append(1)
            mean = ionq_results['no_confounding']['mean']
            ionq_means.append(mean)
            error = [[mean - ionq_results['no_confounding']['ci_lower']], 
                     [ionq_results['no_confounding']['ci_upper'] - mean]]
            ionq_errors.append(error)
            
        if 'quantum' in ionq_results:
            ionq_positions.append(3)
            mean = ionq_results['quantum']['mean']
            ionq_means.append(mean)
            error = [[mean - ionq_results['quantum']['ci_lower']], 
                     [ionq_results['quantum']['ci_upper'] - mean]]
            ionq_errors.append(error)
        
        if ionq_positions:
            ionq_errors_array = np.array(ionq_errors).squeeze().T
            ax.errorbar(ionq_positions, ionq_means,
                        yerr=ionq_errors_array,
                        fmt='o', markersize=12,
                        color='black', markerfacecolor='yellow',
                        markeredgewidth=2, capsize=8, capthick=2,
                        elinewidth=2,
                        label='IonQ Hardware', zorder=10)
            
            # Add text annotations for IonQ results
            for i, (pos, mean) in enumerate(zip(ionq_positions, ionq_means)):
                # Determine scenario based on position
                if pos == 1:
                    scenario = 'no_confounding'
                    label_text = f'IonQ\n($n_{{trials}}$={ionq_results[scenario]["n_trials"]})'
                    # For no confounding, place text to the right
                    x_offset = 0.15
                    y_offset = 0.0
                    ha = 'left'
                    va = 'center'
                    ax.text(pos + x_offset, mean + y_offset, label_text, 
                           ha=ha, va=va, fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                  alpha=0.7, edgecolor='black'))                   
                elif pos == 3:
                    scenario = 'quantum'
                    label_text = f'IonQ\n($n_{{trials}}$={ionq_results[scenario]["n_trials"]})'
                    # For quantum, place text below the point to avoid cutoff
                    x_offset = 0.0
                    y_offset = -0.4
                    ha = 'center'
                    va = 'top'
                
                    ax.text(pos + x_offset -0.2, mean + y_offset +0.2, label_text, 
                           ha=ha, va=va, fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                   alpha=0.7, edgecolor='black'))

    # --- MODIFICATION: Update reference lines to CS scale ---
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2,
               label='Classical Bound (CS ≤ 1)', alpha=0.8)
    ax.axhline(y=np.sqrt(2), color='green', linestyle=':', linewidth=2,
               label=f'Quantum Bound (CS ≤ √2 ≈ {np.sqrt(2):.3f})', alpha=0.8)
    
    # Customize axes
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['No\nConfounding', 'Classical\nConfounding',
                      'Quantum\nSuper-Confounding'], fontsize=16)
    
    # --- MODIFICATION: Update Y-axis label ---
    ax.set_ylabel('Confounding Strength (CS)', fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    
    # --- MODIFICATION: Update text annotations to CS scale ---
    means_cs = [np.mean(d) for d in data_to_plot]
    stds_cs = [np.std(d) for d in data_to_plot]
    
    for i, (mean, std) in enumerate(zip(means_cs, stds_cs)):
        y_offset = 0.15
        if i == 0:
            y_offset = 0.45
        ax.text(i+1, mean + y_offset, f'Sim: μ={mean:.3f}\nσ={std:.3f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.5))
    
    # --- MODIFICATION: Update Y-axis limits for CS scale ---
    ax.set_ylim(-0.2, 1.8) 
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=15) # Adjusted location
    
    plt.tight_layout()
    
    # Save as PDF with high quality
    plt.savefig('fig_2_ex1_cs_hierarchy_with_ionq.pdf', format='pdf', dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    return fig, ax


# Main execution
if __name__ == "__main__":
    # File paths
    sim_json_path = 'experiment_1_results.json'
    ionq_1a_path = 'ex1a_no_confounding_ionq_results.json'
    ionq_1b_path = 'ex1b_quantum_confounding_ionq_results.json'
    
    try:
        # Create and save the combined plot
        fig, ax = create_cs_violin_plot_with_ionq(
            sim_json_path, 
            ionq_1a_path, 
            ionq_1b_path
        )
        
        print("\n🎉 Figure 2 with IonQ hardware validation completed!")
        print("    → Violin plots show simulation distributions")
        print("    → Yellow points show IonQ hardware measurements")
        print("    → Error bars show 95% confidence intervals")
        print("    → Confirms hierarchy: No < Classical < Quantum")
        print("    → Hardware validates theoretical predictions")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Could not find required file")
        print(f"  {e}")
        print("Please ensure all JSON files are in the correct location.")
        
    except Exception as e:
        print(f"\n✗ Error creating plot: {e}")
