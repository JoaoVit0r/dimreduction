import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
import re
import sys
import os
import argparse
import numpy as np
from itertools import combinations
# from upsetplot import UpSet, from_contents

def get_technique_name(filename):
    """Extract technique name from filename"""
    if re.search(r'.*final.*_data.*', filename):
        return 'DimReduction'
    elif filename.startswith('GRN_'):
        # Extract technique name after GRN_ and before next underscore
        match = re.match(r'GRN_([^_]+)', filename)
        if match:
            return match.group(1)
    return os.path.splitext(filename)[0]  # Fallback to filename without extension

def read_edges(filepath, gold_standard):
    """Read edges from file and return set of edges present in gold standard"""

    try:
        # Fallback to comma separator
        df = pd.read_csv(filepath, sep=',', header=None)
    except:
        print(f"Could not read {filepath}")
        return set()

    # Create edges using vectorized operations
    if len(df.columns) < 2:
        return set()
    
    # Convert to string and create edge tuples
    edges = set(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    
    # Filter to only include edges in gold standard
    return edges.intersection(gold_standard)

def create_venn_diagram(edge_sets, technique_names, output_file):
    """Create Venn diagram for 2 or 3 techniques"""
    plt.figure(figsize=(10, 8))
    
    if len(edge_sets) == 2:
        venn2(edge_sets, set_labels=technique_names)
        plt.title("Comparison of 2 Techniques")
    elif len(edge_sets) == 3:
        venn3(edge_sets, set_labels=technique_names)
        plt.title("Comparison of 3 Techniques")
    else:
        print("Venn diagrams are only supported for 2 or 3 techniques")
        return False

    plt.title("Overlap of Correctly Identified Edges Between Techniques")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return True

# def create_upset_plot(technique_edges, technique_names, output_file):
#     """Create an UpSet plot for visualizing intersections between multiple sets"""
#     # Prepare data for UpSet plot
#     upset_data = from_contents(technique_edges)
    
#     # Create UpSet plot
#     plt.figure(figsize=(16, 10))
#     upset = UpSet(upset_data, subset_size='count', show_counts=True)
#     upset.plot()
#     plt.suptitle("Intersections of Correctly Identified Edges Across Techniques", y=0.95)
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     plt.close()

def create_jaccard_heatmap(technique_edges, technique_names, output_file):
    """Create a heatmap of Jaccard similarities between techniques"""
    n_techniques = len(technique_names)
    jaccard_matrix = np.zeros((n_techniques, n_techniques))
    
    # Calculate Jaccard similarity between each pair of techniques
    for i in range(n_techniques):
        for j in range(n_techniques):
            if i == j:
                jaccard_matrix[i, j] = 1.0
            else:
                set_i = technique_edges[technique_names[i]]
                set_j = technique_edges[technique_names[j]]
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(jaccard_matrix, 
                xticklabels=technique_names, 
                yticklabels=technique_names,
                annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title("Jaccard Similarity Between Techniques")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_overlap_matrix(technique_edges, technique_names, output_file):
    """Create a matrix showing overlap counts between techniques"""
    n_techniques = len(technique_names)
    overlap_matrix = np.zeros((n_techniques, n_techniques))
    
    # Calculate overlap between each pair of techniques
    for i in range(n_techniques):
        for j in range(n_techniques):
            if i == j:
                overlap_matrix[i, j] = len(technique_edges[technique_names[i]])
            else:
                set_i = technique_edges[technique_names[i]]
                set_j = technique_edges[technique_names[j]]
                overlap_matrix[i, j] = len(set_i.intersection(set_j))
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(overlap_matrix, 
                xticklabels=technique_names, 
                yticklabels=technique_names,
                annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title("Overlap of Correctly Identified Edges Between Techniques")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create comparison diagrams for GRN inference techniques')
    parser.add_argument('gold_standard', help='Path to gold standard TSV file')
    parser.add_argument('technique_files', nargs='+', help='Paths to technique result files')
    parser.add_argument('--output', default='technique_comparison',
                       help='Output filename prefix for the diagrams')
    
    args = parser.parse_args()

    # Read gold standard
    gold_standard = set()
    try:
        gold_df = pd.read_csv(args.gold_standard, sep='\t', header=None)
        if len(gold_df.columns) >= 2:
            gold_standard = set(zip(gold_df.iloc[:, 0].astype(str), gold_df.iloc[:, 1].astype(str)))
            print(f"Gold standard contains {len(gold_standard)} edges")
    except Exception as e:
        print(f"Error reading gold standard file: {e}")
        sys.exit(1)

    # Process each technique file
    technique_edges = {}
    technique_names = []
    
    for filepath in args.technique_files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        tech_name = get_technique_name(os.path.basename(filepath))
        edges = read_edges(filepath, gold_standard)
        
        technique_edges[tech_name] = edges
        technique_names.append(tech_name)
        print(f"{tech_name}: Found {len(edges)} correct edges")

    # Create appropriate visualization based on number of techniques
    n_techniques = len(technique_names)
    
    if n_techniques == 0:
        print("No valid technique files provided")
        return
    elif n_techniques == 1:
        print("Only one technique provided - no comparison possible")
        return
    elif n_techniques <= 3:
        # Use Venn diagram for 2-3 techniques
        edge_sets = [technique_edges[name] for name in technique_names]
        create_venn_diagram(edge_sets, technique_names, f"{args.output}_venn.png")
    else:
        # Also create heatmaps for additional insights
        create_jaccard_heatmap(technique_edges, technique_names, f"{args.output}_jaccard.png")
        create_overlap_matrix(technique_edges, technique_names, f"{args.output}_overlap.png")
        
        # Also create pairwise Venn diagrams for combinations if not too many
        if n_techniques <= 12:  # Limit to avoid too many combinations
            for i, j in combinations(range(n_techniques), 2):
                name1, name2 = technique_names[i], technique_names[j]
                edge_sets = [technique_edges[name1], technique_edges[name2]]
                create_venn_diagram(
                    edge_sets, 
                    [name1, name2], 
                    f"{args.output}_venn_{name1}_{name2}.png"
                )

    # Print statistics
    print("\nStatistics:")
    for name in technique_names:
        edges = technique_edges[name]
        print(f"{name}: {len(edges)} correct edges ({len(edges)/len(gold_standard)*100:.1f}% of gold standard)")

if __name__ == "__main__":
    main()