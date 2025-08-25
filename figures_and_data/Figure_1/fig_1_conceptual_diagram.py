import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_intro_figure_final():
    """
    Creates the final, publication-ready version of Figure 1.
    This version accurately represents entanglement physics with 
    professional colors and proper causal interpretation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Panel (a): The Conventional View ---
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis('off')

    # Entangled Source (larger circle for better text fit)
    ax1.add_patch(patches.Circle((5, 3), radius=0.6, facecolor='steelblue', edgecolor='darkblue', linewidth=2))
    ax1.text(5, 3, r'$|\Phi^+\rangle$' + '\nState', ha='center', va='center', fontsize=14, color='white', fontweight='bold')

    # Variables A and B
    ax1.add_patch(patches.Circle((1.75, 3), radius=0.5, facecolor='dimgray', edgecolor='black', linewidth=2))
    ax1.text(1.75, 3, 'A', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    ax1.add_patch(patches.Circle((8.25, 3), radius=0.5, facecolor='dimgray', edgecolor='black', linewidth=2))
    ax1.text(8.25, 3, 'B', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Particle paths (professional colors)
    #ax1.arrow(4.2, 3, -1.7, 0, head_width=0.15, head_length=0.2, fc='darkslategray', ec='black', linewidth=2)
    #ax1.arrow(5.8, 3, 1.7, 0, head_width=0.15, head_length=0.2, fc='darkslategray', ec='black', linewidth=2)
    
    # "Non-local Correlation" link with mystery annotation
    style = "angle3,angleA=20,angleB=-20"
    correlation_arrow = patches.FancyArrowPatch(
        (2.25, 3), (7.75, 3),
        connectionstyle=style,
        arrowstyle='<|-|>',
        mutation_scale=20,
        color='darkred',
        linestyle='--',
        linewidth=2,
        alpha=0.7
    )
    ax1.add_patch(correlation_arrow)
    ax1.text(5, 4.3, 'Non-Local Correlation', 
             ha='center', va='center', fontsize=14, color='darkred', 
             fontweight='bold')
    ax1.text(5, 4, '(Mechanism Unknown)', 
             ha='center', va='center', fontsize=12, color='darkred', 
             style='italic', alpha=0.8)
    
    # Subplot caption at bottom
    ax1.text(0.5, 0.15, '(a) The Conventional View', transform=ax1.transAxes, 
             ha='center', fontsize=14)

    # --- Panel (b): The Causal Framework ---
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')

    # Confounder Node (pre-existing quantum resource)
    ax2.add_patch(patches.Circle((5, 4), radius=0.6, facecolor='darkslategray', edgecolor='black', linewidth=2))
    ax2.text(5, 4, r'$|\Phi^+\rangle$' + '\nState', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    ax2.text(5, 5.0, 'Pre-existing\nQuantum Confounder', ha='center', va='center', 
             fontsize=14, color='darkslategray', fontweight='bold')

    # Outcome Nodes A and B
    ax2.add_patch(patches.Circle((2.5, 2), radius=0.5, facecolor='dimgray', edgecolor='black', linewidth=2))
    ax2.text(2.5, 2, 'A', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    ax2.add_patch(patches.Circle((7.5, 2), radius=0.5, facecolor='dimgray', edgecolor='black', linewidth=2))
    ax2.text(7.5, 2, 'B', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Shared quantum resource arrows with triangular heads
    arrow1 = patches.FancyArrowPatch(
        (4.5, 3.6), (2.9, 2.4),
        arrowstyle='-|>',
        color='darkslategray',
        linestyle='-',
        linewidth=1.5,
        alpha=0.8,
        mutation_scale=15
    )
    ax2.add_patch(arrow1)
    
    arrow2 = patches.FancyArrowPatch(
        (5.5, 3.6), (7.1, 2.4),
        arrowstyle='-|>',
        color='darkslategray',
        linestyle='-',
        linewidth=1.5,
        alpha=0.8,
        mutation_scale=15
    )
    ax2.add_patch(arrow2)

    # Add "Shared Resource" annotation
    ax2.text(3.1, 3.2, 'Shared\nResource', ha='center', va='center', 
             fontsize=12, color='darkslategray', style='italic', alpha=0.8)
    ax2.text(6.9, 3.2, 'Shared\nResource', ha='center', va='center', 
             fontsize=12, color='darkslategray', style='italic', alpha=0.8)

    # "No Direct Causation" annotation (enhanced)
    ax2.plot([3, 7], [2, 2], color='darkgray', linestyle=(0, (5, 5)), linewidth=3, alpha=0.8)
    ax2.text(5, 2.3, 'No Direct Causation\n(No-Signaling)', ha='center', va='center', 
             fontsize=11, color='darkgray', fontweight='bold')

    # Subplot caption at bottom
    ax2.text(0.5, 0.15, '(b) The Causal Framework', transform=ax2.transAxes, 
             ha='center', fontsize=14)

    # Final Touches
    plt.tight_layout()
    plt.savefig('fig_1_conceptual_diagram.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

# Main execution
if __name__ == '__main__':
    create_intro_figure_final()
    print("✅ Final Figure 1 ('fig_1_conceptual_diagram.pdf') has been created successfully!")
