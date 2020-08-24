import numpy as np
import pandas as pd
import sys
import urllib.request as urllib
import pickle

GENE_LIST_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\4500_most_frequent_mutated_genes.tsv"
GENE_NAMES_CONVERTION_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\ensmbl_to_gene_id_convertion.xlsx"
GENE_TO_PROTEIN_CONVERTION = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\ensembl_gene_to_protein.xlsx"
INTERACTIONS_LIST_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\string_interactions.tsv"
FULL_INTERACTIONS_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\9606.protein.links.v10.5.txt.gz"
OUTPUT_PARTIAL_GRAPH_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\partial_interactions_graph"
OUTPUT_FULL_GRAPH_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\full_interactions_graph"

def build_empty_graph(k):
    genes_list = pd.read_csv(GENE_LIST_PATH)['id'].head(k)
    graph = np.zeros((genes_list.shape[0], genes_list.shape[0]))
    return graph

def query_genes_db(string_names):
    string_api_url = "https://string-db.org/api"
    output_format = "tsv-no-header"
    method = "network"

    genes = string_names.tolist()[:800]
    species = "9606"
    my_app = "www.tau.org"

    request_url = string_api_url + "/" + output_format + "/" + method + "?"
    request_url += "identifiers=%s" % "%0d".join(genes)
    request_url += "&" + "species=" + species
    request_url += "&" + "caller_identity=" + my_app
    print(request_url)
    try:
        response = urllib.urlopen(request_url)
    except urllib.HTTPError as err:
        error_message = err.read()
        print(error_message)
        sys.exit()
    return response


def build_graph_from_response(df, graph, ensmbl_to_index, string_to_ensmbl_dict, protein_to_gene_dict):
    print(df.columns)
    interactions = list(zip(df['protein1'], df['protein2'], df['combined_score']))
    count = 0
    for i in interactions:
        #print(i)
        if i[0][5:] in protein_to_gene_dict and i[1][5:] in protein_to_gene_dict:
            g1 = protein_to_gene_dict[i[0][5:]]
            g2 = protein_to_gene_dict[i[1][5:]]
            #print(g1, g2)
            if g1 in ensmbl_to_index and g2 in ensmbl_to_index:
                graph[ensmbl_to_index[g1], ensmbl_to_index[g2]] = i[2]
                count += 1
    print("total of interactions added:", count)
    return graph


def build_gene_interactions_graph(k):
    df = pd.read_excel(GENE_NAMES_CONVERTION_PATH, header=None).head(k)
    ensmbl_names, string_names = df[0], df[1]
    string_to_ensmbl_dict = dict(zip(string_names, ensmbl_names))
    ensmbl_to_index = dict(map(lambda t: (t[1], t[0]), enumerate(ensmbl_names)))
    empty_graph = build_empty_graph(k)
    #response = query_genes_db(string_names)
    #pickle.dump(response.readlines(), open("response", "wb"))
    #partial_interactions = pd.read_csv(INTERACTIONS_LIST_PATH, sep='\t')
    gene_to_protein = pd.read_excel(GENE_TO_PROTEIN_CONVERTION)
    protein_to_gene_dict = dict(zip(gene_to_protein['Protein stable ID'], gene_to_protein['Gene stable ID']))
    full_interactions = pd.read_csv(FULL_INTERACTIONS_PATH, compression='gzip', sep=' ')
    final_graph = build_graph_from_response(full_interactions, empty_graph, ensmbl_to_index, string_to_ensmbl_dict, protein_to_gene_dict)
    return final_graph

k = 2000
graph = build_gene_interactions_graph(k)
print(graph)
pickle.dump(graph, open(OUTPUT_PARTIAL_GRAPH_PATH, "wb"))