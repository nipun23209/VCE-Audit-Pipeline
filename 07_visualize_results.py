"""
Phase 7: Visualizing VCE Distributions
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: Uses pandas and seaborn to generate high-resolution distribution graphs of the VCE scores, split by clinical ground truth.
References: Seaborn documentation, Matplotlib.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations():
    print("Loading VCE Audit Data...")
    df = pd.read_csv("vce_audit_results.csv")
    
    # Calculate core statistics
    mean_vce = df["VCE_Score"].mean()
    max_vce = df["VCE_Score"].max()
    min_vce = df["VCE_Score"].min()
    
    print(f"\n--- VCE Summary Statistics ---")
    print(f"Total X-Rays Analyzed: {len(df)}")
    print(f"Average VCE Score: {mean_vce:.4f}")
    print(f"Maximum VCE Spike: {max_vce:.4f}")
    print(f"Minimum VCE Score: {min_vce:.4f}")
    print("------------------------------")
    
    # Initialize the plotting canvas
    plt.figure(figsize=(10, 6))
    
    # Create a beautiful distribution plot, split by the Ground Truth (NORMAL vs PNEUMONIA)
    sns.histplot(
        data=df, 
        x="VCE_Score", 
        hue="Ground_Truth", 
        kde=True, 
        bins=40, 
        palette="viridis",
        edgecolor='black',
        alpha=0.7
    )
    
    # Add professional labels and titles
    plt.title("Distribution of Vision-Conditioned Entropy (VCE) Scores", fontsize=14, fontweight='bold')
    plt.xlabel("VCE Score (Perturbed Entropy - Base Entropy)", fontsize=12)
    plt.ylabel("Number of X-Rays", fontsize=12)
    
    # Add a critical red threshold line at exactly 0.0
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='0.0 Threshold (Text-Prior Hallucination)')
    plt.legend()
    
    # Save the plot as a high-resolution PNG for the final report
    plt.tight_layout()
    plot_filename = "vce_distribution_plot.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nSuccess! High-resolution graph saved to: {plot_filename}")

if __name__ == "__main__":
    generate_visualizations()