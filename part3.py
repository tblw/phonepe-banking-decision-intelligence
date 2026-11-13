import os
import numpy as np

def run_part3():
    os.makedirs("outputs", exist_ok=True)
    data = np.genfromtxt("data/records.csv", delimiter=",", skip_header=1)
    # Continuous predictors X1 to X5: columns 1 to 5
    X_cont = data[:, 1:6]
    outcomes = data[:, 8]
    values = data[:, 9]
    n = len(data)

    # Standardise X1 to X5 over all 800 rows
    mu = np.mean(X_cont, axis=0)
    sd = np.std(X_cont, axis=0, ddof=1)
    X_std = (X_cont - mu) / sd

    # 3.1 K-means from scratch for K = 2..6
    wcss_results = {}
    kmeans_assignments = {}

    for k in range(2, 7):
        start_indices = [i * (n // k) for i in range(k)]
        centroids = X_std[start_indices].copy()
        
        while True:
            # Assign rows to nearest centroid
            dists = np.linalg.norm(X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            assignments = np.argmin(dists, axis=1) # ties go to lower index automatically
            
            # Update centroids
            new_centroids = centroids.copy()
            for c_idx in range(k):
                members = np.where(assignments == c_idx)[0]
                if len(members) > 0:
                    new_centroids[c_idx] = np.mean(X_std[members], axis=0)
            
            if np.all(assignments == locals().get('prev_assignments', None)):
                break
            prev_assignments = assignments
            centroids = new_centroids

        # Calculate WCSS
        wcss = np.sum((X_std - centroids[assignments]) ** 2)
        wcss_results[k] = wcss
        kmeans_assignments[k] = assignments

    wcss_k2 = wcss_results[2]
    wcss_k3 = wcss_results[3]
    wcss_k4 = wcss_results[4]
    wcss_k5 = wcss_results[5]
    wcss_k6 = wcss_results[6]

    # 3.2 Choose K with elbow
    drop_23 = wcss_k2 - wcss_k3
    drop_34 = wcss_k3 - wcss_k4
    drop_45 = wcss_k4 - wcss_k5
    
    elbow_k = 3
    for cand in [3, 4, 5]:
        if locals().get(f"wcss_k{cand}") - locals().get(f"wcss_k{cand+1}") < 0.5 * (locals().get(f"wcss_k{cand-1}") - locals().get(f"wcss_k{cand}")):
            elbow_k = cand
            break

    # Use K = 3 for profiling
    assign_3 = kmeans_assignments[3]
    
    # Order clusters by ascending mean of X3 (column index 2 in X_cont)
    cluster_means_x3 = []
    for c in range(3):
        members = np.where(assign_3 == c)[0]
        cluster_means_x3.append((np.mean(X_cont[members, 2]), c))
    
    cluster_means_x3.sort(key=lambda x: x[0])
    cluster_mapping = {cluster_means_x3[0][1]: 'A', cluster_means_x3[1][1]: 'B', cluster_means_x3[2][1]: 'C'}
    
    segment_rows = {'A': [], 'B': [], 'C': []}
    for idx, c in enumerate(assign_3):
        seg_name = cluster_mapping[c]
        segment_rows[seg_name].append(idx)

    seg_stats = {}
    segment_csv_lines = []
    for seg in ['A', 'B', 'C']:
        indices = segment_rows[seg]
        size = len(indices)
        mean_X1 = np.mean(X_cont[indices, 0])
        mean_X2 = np.mean(X_cont[indices, 1])
        mean_X3 = np.mean(X_cont[indices, 2])
        mean_X4 = np.mean(X_cont[indices, 3])
        mean_X5 = np.mean(X_cont[indices, 4])
        outcome_rate = np.mean(outcomes[indices])
        mean_value = np.mean(values[indices])
        seg_stats[seg] = {
            'size': size, 'mean_X1': mean_X1, 'mean_X2': mean_X2, 'mean_X3': mean_X3,
            'mean_X4': mean_X4, 'mean_X5': mean_X5, 'outcome_rate': outcome_rate, 'mean_value': mean_value
        }
        segment_csv_lines.append(f"{seg},{size},{mean_X1:.4f},{mean_X2:.4f},{mean_X3:.4f},{mean_X4:.4f},{mean_X5:.4f},{outcome_rate:.4f},{mean_value:.4f}")

    size_segment_A = seg_stats['A']['size']
    size_segment_B = seg_stats['B']['size']
    size_segment_C = seg_stats['C']['size']

    with open("outputs/part3_segments.csv", "w") as f:
        f.write("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value\n" + "\n".join(segment_csv_lines) + "\n")

    # 3.4 Hierarchical clustering on supplied table
    diss_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ])
    cohort_names = ["E1", "E2", "E3", "E4", "E5", "E6"]

    def run_hierarchical(linkage_type):
        clusters = {i: [name] for i, name in enumerate(cohort_names)}
        d_mat = diss_matrix.copy()
        merges = []
        
        for step in range(1, 6):
            min_d = float('inf')
            pair = (None, None)
            keys = list(clusters.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    c1, c2 = keys[i], keys[j]
                    members1 = clusters[c1]
                    members2 = clusters[c2]
                    
                    if linkage_type == 'single':
                        d = min(diss_matrix[cohort_names.index(m1), cohort_names.index(m2)] for m1 in members1 for m2 in members2)
                    elif linkage_type == 'complete':
                        d = max(diss_matrix[cohort_names.index(m1), cohort_names.index(m2)] for m1 in members1 for m2 in members2)
                    else: # average
                        d = sum(diss_matrix[cohort_names.index(m1), cohort_names.index(m2)] for m1 in members1 for m2 in members2) / (len(members1) * len(members2))
                    
                    if d < min_d:
                        min_d = d
                        pair = (c1, c2)
            
            c1, c2 = pair
            m1, m2 = clusters[c1], clusters[c2]
            # cluster_a has smaller member number
            sorted_m1 = sorted(m1, key=lambda x: int(x[1]))
            sorted_m2 = sorted(m2, key=lambda x: int(x[1]))
            
            # format name
            name1 = "+".join(sorted_m1)
            name2 = "+".join(sorted_m2)
            if int(sorted_m1[0][1]) < int(sorted_m2[0][1]):
                ca, cb = name1, name2
            else:
                ca, cb = name2, name1

            merges.append((step, ca, cb, min_d))
            
            # Merge clusters
            new_members = m1 + m2
            del clusters[c1]
            del clusters[c2]
            new_key = max(clusters.keys()) + 1 if clusters else 0
            clusters[new_key] = new_members
            
        return merges

    hier_rows = []
    cuts = {}
    for l_type in ['single', 'complete', 'average']:
        merges = run_hierarchical(l_type)
        heights = [m[3] for m in merges]
        gaps = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
        max_gap_idx = np.argmax(gaps)
        # number of clusters = 6 - number of merges below cut
        # merges below cut are indices 0 to max_gap_idx
        clusters_at_cut = 6 - (max_gap_idx + 1)
        cuts[l_type] = clusters_at_cut
        for step, ca, cb, h in merges:
            hier_rows.append(f"{l_type},{step},{ca},{cb},{h:.4f}")

    with open("outputs/part3_hierarchical.csv", "w") as f:
        f.write("linkage,merge_no,cluster_a,cluster_b,height\n" + "\n".join(hier_rows) + "\n")

    single_clusters_at_cut = cuts['single']
    complete_clusters_at_cut = cuts['complete']
    average_clusters_at_cut = cuts['average']

    # 3.5 PCA by seven steps
    # Covariance matrix S = (1 / (n - 1)) * X_std.T @ X_std
    S = (1.0 / (n - 1)) * (X_std.T @ X_std)
    eigenvals, eigenvecs = np.linalg.eigh(S)
    
    # Sort descending
    sort_idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sort_idx]
    eigenvecs = eigenvecs[:, sort_idx]

    # Sign convention: flip eigenvector if largest absolute entry is negative
    for i in range(eigenvecs.shape[1]):
        col = eigenvecs[:, i]
        max_abs_idx = np.argmax(np.abs(col))
        if col[max_abs_idx] < 0:
            eigenvecs[:, i] = -col

    eigenvalue_sum = np.sum(eigenvals)
    explained_ratios = eigenvals / eigenvalue_sum
    cum_ratios = np.cumsum(explained_ratios)

    components_for_80pct = int(np.searchsorted(cum_ratios, 0.8) + 1)

    # Projections Z = X_std @ W_top2
    W_top2 = eigenvecs[:, :2]
    Z = X_std @ W_top2

    row1_pc1, row1_pc2 = Z[0, 0], Z[0, 1]
    row2_pc1, row2_pc2 = Z[1, 0], Z[1, 1]
    row3_pc1, row3_pc2 = Z[2, 0], Z[2, 1]

    mean_pc1_segment_A = np.mean(Z[segment_rows['A'], 0])
    mean_pc1_segment_B = np.mean(Z[segment_rows['B'], 0])
    mean_pc1_segment_C = np.mean(Z[segment_rows['C'], 0])

    # Write outputs/part3_results.txt
    results = [
        f"wcss_k2 = {wcss_k2:.4f}",
        f"wcss_k3 = {wcss_k3:.4f}",
        f"wcss_k4 = {wcss_k4:.4f}",
        f"wcss_k5 = {wcss_k5:.4f}",
        f"wcss_k6 = {wcss_k6:.4f}",
        f"elbow_k = {elbow_k}",
        f"size_segment_A = {size_segment_A}",
        f"size_segment_B = {size_segment_B}",
        f"size_segment_C = {size_segment_C}",
        f"single_clusters_at_cut = {single_clusters_at_cut}",
        f"complete_clusters_at_cut = {complete_clusters_at_cut}",
        f"average_clusters_at_cut = {average_clusters_at_cut}",
        f"eigenvalue_sum = {eigenvalue_sum:.4f}",
        f"eigenvalue_1 = {eigenvals[0]:.4f}",
        f"eigenvalue_2 = {eigenvals[1]:.4f}",
        f"eigenvalue_3 = {eigenvals[2]:.4f}",
        f"eigenvalue_4 = {eigenvals[3]:.4f}",
        f"eigenvalue_5 = {eigenvals[4]:.4f}",
        f"explained_ratio_1 = {explained_ratios[0]:.4f}",
        f"explained_ratio_2 = {explained_ratios[1]:.4f}",
        f"explained_ratio_3 = {explained_ratios[2]:.4f}",
        f"explained_ratio_4 = {explained_ratios[3]:.4f}",
        f"explained_ratio_5 = {explained_ratios[4]:.4f}",
        f"cumulative_ratio_1 = {cum_ratios[0]:.4f}",
        f"cumulative_ratio_2 = {cum_ratios[1]:.4f}",
        f"cumulative_ratio_3 = {cum_ratios[2]:.4f}",
        f"cumulative_ratio_4 = {cum_ratios[3]:.4f}",
        f"cumulative_ratio_5 = {cum_ratios[4]:.4f}",
        f"components_for_80pct = {components_for_80pct}",
        f"pc1_loading_X1 = {eigenvecs[0, 0]:.4f}",
        f"pc1_loading_X2 = {eigenvecs[1, 0]:.4f}",
        f"pc1_loading_X3 = {eigenvecs[2, 0]:.4f}",
        f"pc1_loading_X4 = {eigenvecs[3, 0]:.4f}",
        f"pc1_loading_X5 = {eigenvecs[4, 0]:.4f}",
        f"pc2_loading_X1 = {eigenvecs[0, 1]:.4f}",
        f"pc2_loading_X2 = {eigenvecs[1, 1]:.4f}",
        f"pc2_loading_X3 = {eigenvecs[2, 1]:.4f}",
        f"pc2_loading_X4 = {eigenvecs[3, 1]:.4f}",
        f"pc2_loading_X5 = {eigenvecs[4, 1]:.4f}",
        f"row1_pc1 = {row1_pc1:.4f}",
        f"row1_pc2 = {row1_pc2:.4f}",
        f"row2_pc1 = {row2_pc1:.4f}",
        f"row2_pc2 = {row2_pc2:.4f}",
        f"row3_pc1 = {row3_pc1:.4f}",
        f"row3_pc2 = {row3_pc2:.4f}",
        f"mean_pc1_segment_A = {mean_pc1_segment_A:.4f}",
        f"mean_pc1_segment_B = {mean_pc1_segment_B:.4f}",
        f"mean_pc1_segment_C = {mean_pc1_segment_C:.4f}"
    ]

    with open("outputs/part3_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")

if __name__ == "__main__":
    run_part3()