import matplotlib.pyplot as plt
import numpy as np

def create_do_calculus_figure():
    """
    Creates Figure 4 for the paper, visualizing the results of the
    physical quantum do-calculus implementation (Experiment 4 / ex3new).
    """
    
    # --- 1. Data from ex3new_quantum_do_calculus.py log ---
    # Observational probabilities
    p_b0_given_a0_obs = 1.0000
    p_b0_given_a1_obs = 0.0000
    
    # Interventional probabilities (using the 'Project-Prepare' method as primary)
    p_b0_given_do_a0_int = 0.5013
    p_b0_given_do_a1_int = 0.5008
    
    # --- 2. Create the Plot ---
    labels = ['Conditioned on A=0', 'Conditioned on A=1']
    observational_values = [p_b0_given_a0_obs, p_b0_given_a1_obs]
    interventional_values = [p_b0_given_do_a0_int, p_b0_given_do_a1_int]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create bars for observational and interventional data
    rects1 = ax.bar(x - width/2, observational_values, width, 
                    label='Observational: $P(B=0|A)$', color='blue', alpha=0.7, edgecolor='black')
    rects2 = ax.bar(x + width/2, interventional_values, width, 
                    label='Interventional: $P(B=0|\\mathcal{DO}(A))$', color='orange', alpha=0.7, edgecolor='black')

    # Add a horizontal line for the independence threshold (0.5)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Independence (Prob = 0.5)')

    # --- 3. Customize the Plot ---
    ax.set_ylabel('Probability P(B=0)', fontsize=20, fontweight='bold')
    #ax.set_title('Quantum $\\mathcal{DO}$-Calculus: Observation vs. Intervention', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=16)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    # Attach a text label above each bar, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=16)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Save the figure
    output_filename = 'fig_4_do_calculus_validation.pdf'
    plt.savefig(output_filename, format='pdf', dpi=400, bbox_inches='tight')
    
    print(f"✅ Figure 4 ('{output_filename}') has been created successfully.")
    
    # Display the plot
    plt.show()

# Main execution
if __name__ == '__main__':
    create_do_calculus_figure()
