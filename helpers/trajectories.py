from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
import os


def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

def find_best_genes(genes):
    square_size = 100
    best_index = -1
    best_value = -1
    for i in range(genes.shape[0] - square_size):
        current_value = genes.iloc[i:i + square_size,i:i + square_size].sum().sum()
        if best_value == -1 or best_value > current_value:
            best_value = current_value
            best_index = i

    best_genes = genes.index[best_index:best_index + square_size]
    return best_genes


def main():
    root_folder = "correlations_matrices"
    output_folder = "output_corr"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    perts = ["geldanamycin", "raloxifene", "trichostatin-a", "vorinostat", "wortmannin"]
    for pert in perts:
        pert_folder = os.path.join(root_folder, pert)
        out_pert_folder = os.path.join(output_folder, pert)
        files = os.listdir(pert_folder)
        results_dict = {}
        if not os.path.isdir(out_pert_folder):
            os.mkdir(out_pert_folder)
        for file_name in files:
            file_path = os.path.join(pert_folder, file_name)
            genes = pd.read_csv(file_path)
            genes.set_index('rid', inplace=True)
            dist_mat = 1 - genes.abs()
            method = "average"
            dist_mat_np = dist_mat.values
            ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat_np,method)
            ordered_genes = genes.iloc[res_order]
            tumor = file_name.split('_')[1][:-4]
            current_folder = os.path.join(out_pert_folder, tumor)
            if not os.path.isdir(current_folder):
                os.mkdir(current_folder)
            for i in range(2):
                best_genes = find_best_genes(ordered_genes)
                ordered_genes.drop(best_genes, axis=0, inplace=True)
                if i == 0:
                    results_dict[tumor] = set(best_genes.values)
                with open(os.path.join(current_folder, "best_genes_{}.txt".format(i)), 'w') as f:
                    f.write('\n'.join(best_genes.map(str)))
            plt.pcolormesh(1 - ordered_dist_mat)
            plt.colorbar()
            plt.xlim([0, ordered_dist_mat.shape[0]])
            plt.ylim([0, ordered_dist_mat.shape[0]])
            plt.savefig(os.path.join(current_folder, "genes_map.png"))
            plt.cla()
            plt.close("all")
        tumors = results_dict.keys()
        results_df = pd.DataFrame(index=tumors, columns=tumors)
        for t1 in tumors:
            t1_genes = results_dict[t1]
            for t2 in tumors:
                t2_genes = results_dict[t2]
                results_df.loc[t1, t2] = len(t1_genes.intersection(t2_genes))
        results_df.to_csv(os.path.join(out_pert_folder, "cross_tumors_intersection.csv"))


if __name__ == '__main__':
    main()