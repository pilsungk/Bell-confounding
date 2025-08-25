import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def create_qml_causality_figure():
    """
    Creates the complete Figure 5 for the paper, summarizing the results of
    the QML Causal Feature Selection experiment with multi-seed statistics.
    """
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 2.5, 3.5])

    # --- Subplot (a): The Causal Graph (No changes needed) ---
    ax1 = fig.add_subplot(gs[0])
    ax1.text(0.5, -0.2, '(a) Causal Graph of the Experiment', transform=ax1.transAxes, ha='center', fontsize=18)
    ax1.axis('off')
    
    positions = {'C': (0.1, 0.5), 'A': (0.5, 0.5), 'B': (0.9, 0.5)}
    
    # Nodes
    for name, pos in positions.items():
        ax1.add_patch(patches.Circle(pos, radius=0.08, facecolor='lightblue', edgecolor='black', zorder=5))
        ax1.text(pos[0], pos[1], name, ha='center', va='center', fontsize=20, fontweight='bold', zorder=10)

    # Arrows
    ax1.arrow(positions['A'][0] + 0.08, positions['A'][1], 0.24, 0,
              head_width=0.04, head_length=0.04, fc='black', ec='black', length_includes_head=True)
    ax1.text(0.7, 0.55, 'True\nCausation', ha='center', fontsize=16)

    style = "angle3,angleA=30,angleB=-30"
    arrow_conf = patches.FancyArrowPatch(
        (positions['C'][0], positions['C'][1] + 0.08),
        (positions['A'][0], positions['A'][1] + 0.08),
        connectionstyle=style,
        arrowstyle='<->, head_width=8, head_length=8',
        color='red',
        linestyle='--',
        linewidth=2
    )
    ax1.add_patch(arrow_conf)
    ax1.text(0.3, 0.65, 'Entanglement\n(Confounding)', ha='center', fontsize=16, color='red')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # --- Subplot (b): The Effect of Causal Intervention (No changes needed) ---
    ax2 = fig.add_subplot(gs[1])
    ax2.text(0.5, -0.2, '(b) Effect of Causal Intervention', transform=ax2.transAxes, ha='center', fontsize=18)
    
    obs_prob_c1 = 1.0000
    int_prob_c1 = 0.4860
    
    labels = ['Observational\n$\\mathbf{P(B|C)}$', 'Interventional\n$\\mathbf{P(B|\\mathcal{DO}(C))}$']
    values = [obs_prob_c1, int_prob_c1]
    colors = ['blue', 'orange']

    x_positions = np.arange(len(labels))
    bars = ax2.bar(x_positions, values, width=0.6, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(labels, fontsize=14, fontweight='bold')
    
    ax2.set_ylabel('Probability P(B=1)', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess (0.5)')
    ax2.tick_params(axis='both', labelsize=14)
    ax2.legend(loc='upper right', fontsize=13)
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # --- Subplot (c): The Practical Payoff (Robustness) ---
    ax3 = fig.add_subplot(gs[2])
    ax3.text(0.5, -0.2, '(c) The Practical Payoff: Robustness', transform=ax3.transAxes, ha='center', fontsize=18)
    
    # New data from the multi-seed experiment log
    conf_strengths = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    
    # Causal Classifier data (mean and std dev)
    causal_means = np.array([0.945, 0.952, 0.949, 0.948, 0.947])
    causal_stds = np.array([0.038, 0.035, 0.032, 0.033, 0.032])
    
    # Naive Classifier data (mean and std dev)
    naive_means = np.array([0.733, 0.770, 0.839, 0.891, 0.947])
    naive_stds = np.array([0.221, 0.183, 0.118, 0.065, 0.032])
    
    # Plotting the mean accuracy lines
    ax3.plot(conf_strengths, causal_means, 's-', markersize=8, linewidth=3, label='Causal Classifier (A only)', color='green', zorder=10)
    ax3.plot(conf_strengths, naive_means, 'o-', markersize=8, linewidth=3, label='Naive Classifier (A + C)', color='orange', zorder=10)
    
    # Plotting the shaded standard deviation bands
    ax3.fill_between(conf_strengths, causal_means - causal_stds, causal_means + causal_stds, color='green', alpha=0.15)
    ax3.fill_between(conf_strengths, naive_means - naive_stds, naive_means + naive_stds, color='orange', alpha=0.15)
    
    # Highlighting the robustness gain area
    ax3.fill_between(conf_strengths, naive_means, causal_means, where=(causal_means > naive_means),
                     color='green', alpha=0.25, interpolate=True, label='Robustness Gain')

    ax3.set_xlabel('Confounding Control Parameter in Test Data', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Classifier Accuracy', fontsize=16, fontweight='bold')
    ax3.set_ylim(0.4, 1.05)
    ax3.tick_params(axis='both', labelsize=14)
    ax3.legend(loc='lower left', fontsize=14) # Adjusted location for better visibility
    ax3.grid(True, alpha=0.5)
    
    # --- Final Touches ---
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for titles
    plt.savefig('fig_5_qml_causality_revised.pdf', format='pdf', dpi=400, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == '__main__':
    create_qml_causality_figure()
    print("✅ Revised Figure 5 ('fig_5_qml_causality_revised.pdf') has been created successfully.")
