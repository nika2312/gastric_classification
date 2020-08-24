import pandas as pd
import glob
import numpy as np

GENE_LIST_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\4500_most_frequent_mutated_genes.tsv"
PATIENTS_LABELS_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\patients_and_classes.xlsx"
FILES_BY_PATIENT_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\filtered_FPKM_UQ_cases_and_files.csv"
DATA_FILES_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\gdc_download_20180422_180157\**\*"
OUTPUT_PATH = r"C:\Users\veronica\Desktop\study\Deep Learning\Project\gene_expression_by_patient.csv"

def build_gene_expression_matrix():
    genes_list = pd.read_csv(GENE_LIST_PATH)['id'].tolist()
    patients_list = pd.read_excel(PATIENTS_LABELS_PATH)['TCGA barcode']
    matrix = pd.DataFrame(columns=["patient"] + genes_list)
    matrix["patient"] = patients_list
    matrix.index = matrix["patient"]
    files = pd.read_csv(FILES_BY_PATIENT_PATH)
    filenames = np.array(files['File Name'])
    cases = np.array(files['Case ID'])
    file_to_case_dict = dict(zip(filenames, cases))
    for file in glob.glob(DATA_FILES_PATH, recursive=True):
        #print(file)
        short = file.split("\\")[-1]
        if short in filenames:
            case = file_to_case_dict[short]
            df = pd.read_csv(file, compression='gzip', header=None, sep='\t')
            df.columns = ['id', 'value']
            df['id'] = df['id'].apply(lambda x: x.split(".")[0])
            df = df[df['id'].isin(genes_list)]
            df.index = df['id']
            df = df.drop(['id'], axis=1).transpose()
            #print("df is: ",df.head())
            matrix.ix[case] = df.ix['value']
            matrix["patient"].ix[case] = case
            #print(matrix.ix[case])
            #print(case)
            #break
    return matrix

def rename_class(x):
    if x=="MSI":
        return 0
    if x== "EBV":
        return 1
    if x== "CIN":
        return 2
    if x=="GS":
        return 3

def add_labels():
    data = pd.read_csv(OUTPUT_PATH)
    data = data.rename(columns={"patient": "barcode"})
    data = data.drop(["patient.1"], axis=1)
    labels = pd.read_excel(r"C:\Users\veronica\Desktop\study\Deep Learning\Project\patients_and_classes.xlsx")
    labels = labels.drop([x for x in labels.columns if ("TCGA" not in x and "Molecular Subtype" not in x)], axis=1)
    labels["Molecular Subtype"] = labels["Molecular Subtype"].apply(lambda x: rename_class(x))
    labels = labels.rename(columns={"TCGA barcode": "barcode", "Molecular Subtype": "label"})
    all_data = data.merge(labels, on=["barcode"], how="inner")
    all_data.to_csv("all_data.csv")


df = build_gene_expression_matrix()
df = df.dropna()
df.to_csv(OUTPUT_PATH)
add_labels()