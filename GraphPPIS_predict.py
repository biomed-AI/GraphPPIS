import os, pickle, datetime, argparse, string
import numpy as np
import pandas as pd
from Model.model import *

# Set these paths to your own paths
# Note that in UR90, 'uniref90' (last of the variable) is the prefix of the built database files in the database folder. Same does HHDB.
UR90 = "/bigdat1/pub/yuanqm/uniref90_2018_06/uniref90"
HHDB = "/bigdat1/pub/uniclust30/uniclust30_2017_10/uniclust30_2017_10"

Software_path = "/data2/users/yuanqm/PPI/Software/"
PSIBLAST = Software_path + "ncbi-blast-2.10.1+/bin/psiblast"
HHBLITS = Software_path + "hhsuite-3.0.3/bin/hhblits"
DSSP = Software_path + "dssp-3.1.4/mkdssp"

aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
aa_abbr = [x for x in "ACDEFGHIKLMNPQRSTVWY"]
aa_dict = dict(zip(aa, aa_abbr))

# BLOSUM62
Max_blosum = np.array([4, 5, 6, 6, 9, 5, 5, 6, 8, 4, 4, 5, 5, 6, 7, 4, 5, 11, 7, 4])
Min_blosum = np.array([-3, -3, -4, -4, -4, -3, -4, -4, -3, -4, -4, -3, -3, -4, -4, -3, -2, -4, -3, -3])

# These values are observed in the training sets in our paper
Max_pssm = np.array([8, 9, 9, 9, 12, 9, 8, 8, 12, 9, 7, 9, 11, 10, 9, 8, 8, 13, 10, 8])
Min_pssm = np.array([-11,-12,-13,-13,-12,-13,-13,-12,-13,-13,-13,-13,-12,-12,-13,-12,-12,-13,-13,-12])
Max_hhm = np.array([10655,12141,12162,11354,11802,11835,11457,11686,11806,11262,11571,11979,12234,11884,11732,11508,11207,11388,12201,11743])
Min_hhm = np.zeros(20)

error_code_dic = {"PDB not exist": 1, "chain not exist": 2, "PDB_seq & dismap_seq mismatch": 3, "DSSP too long": 4, "Fail to pad DSSP": 5}


def get_PDB(PDBID, pdb_file, chain, data_path):
    ID = PDBID + chain
    if PDBID == "user" and pdb_file != "": # User custom PDB file
        os.system("mv {} {}".format(pdb_file, os.path.dirname(pdb_file) + "/user.pdb"))
        os.system("perl getchain_pdb.pl {} {}".format(os.path.dirname(pdb_file), ID))
    else:
        os.system("wget -P {} http://www.rcsb.org/pdb/files/{}.pdb.gz".format(data_path, PDBID))
        if os.path.exists(data_path + "{}.pdb.gz".format(PDBID)) == False:
            return "", error_code_dic["PDB not exist"]
        os.system("perl getchain_pdb.pl {} {}".format(data_path, ID))

    os.system("mv {} {}".format(ID, data_path)) # the output of getchain_pdb.pl is in current directory

    seq = ""
    current_pos = -1
    with open(data_path + ID, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos:
            aa_type = line[17:20].strip()
            seq += aa_dict[aa_type]
            current_pos = int(line[22:26].strip())

    if seq == "":
        return "", error_code_dic["chain not exist"]
    else:
        return seq, 0


def process_distance_map(distance_map_file, cutoff = 14):
    with open(distance_map_file, "r") as f:
        lines = f.readlines()

    seq = lines[0].strip()
    length = len(seq)
    distance_map = np.zeros((length, length))

    if lines[1][0] == "#": # missed residues
        missed_idx = [int(x) for x in lines[1].split(":")[1].strip().split()] # 0-based
        lines = lines[2:]
    else:
        missed_idx = []
        lines = lines[1:]

    for i in range(0, len(lines)):
        record = lines[i].strip().split()
        for j in range(0, len(record)):
            if float(record[j]) == -1:
                distance_map[i + 1][j] = 0
            elif float(record[j]) <= cutoff:
                distance_map[i + 1][j] = 1
            else:
                distance_map[i + 1][j] = 0

    for idx in missed_idx:
        if idx > 0:
            distance_map[idx][idx - 1] = 1
        if idx > 1:
            distance_map[idx][idx - 2] = 1
        if idx < length - 1:
            distance_map[idx + 1][idx] = 1
        if idx < length - 2:
            distance_map[idx + 2][idx] = 1

    distance_map = distance_map + distance_map.T + np.eye(length)
    return seq, distance_map


def get_distance_map(ID, PDB_seq, data_path):
    os.system("./caldis_CA {} > {}.map".format(data_path + ID, data_path + "dismap/" + ID))
    dis_map_seq, dis_map = process_distance_map(data_path + "dismap/" + ID + ".map")
    if PDB_seq != dis_map_seq:
        return error_code_dic["PDB_seq & dismap_seq mismatch"]
    else:
        np.save(data_path + "dismap/" + ID, dis_map)
        return 0


def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(9) # The last dim represents "Unknown" for missing residues
        SS_vec[SS_type.find(SS)] = 1
        PHI = float(lines[i][103:109].strip())
        PSI = float(lines[i][109:115].strip())
        ACC = float(lines[i][34:38].strip())
        ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100
        dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))

    return seq, np.array(dssp_feature)


def pad_dssp(seq, feature, ref_seq): # ref_seq is longer
    padded_feature = []
    SS_vec = np.zeros(9) # The last dim represent "Unknown" for missing residues
    SS_vec[-1] = 1
    padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))

    p_ref = 0
    for i in range(len(seq)):
        while p_ref < len(ref_seq) and seq[i] != ref_seq[p_ref]:
            padded_feature.append(padded_item)
            p_ref += 1
        if p_ref < len(ref_seq): # aa matched
            padded_feature.append(feature[i])
            p_ref += 1
        else: # miss match!
            return np.array([])

    if len(padded_feature) != len(ref_seq):
        for i in range(len(ref_seq) - len(padded_feature)):
            padded_feature.append(padded_item)

    return np.array(padded_feature)


def transform_dssp(dssp_feature):
    angle = dssp_feature[:,0:2]
    ASA_SS = dssp_feature[:,2:]

    radian = angle * (np.pi / 180)
    dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis = 1)

    return dssp_feature


def get_dssp(ID, PDB_seq, data_path):
    os.system("{} -i {} -o {}.dssp".format(DSSP, data_path + ID, data_path + "dssp/" + ID))
    dssp_seq, dssp_matrix = process_dssp(data_path + "dssp/" + ID + ".dssp")
    if len(dssp_seq) > len(PDB_seq):
        return error_code_dic["DSSP too long"]
    elif len(dssp_seq) < len(PDB_seq):
        padded_dssp_matrix = pad_dssp(dssp_seq, dssp_matrix, PDB_seq)
        if len(padded_dssp_matrix) == 0:
            return error_code_dic["Fail to pad DSSP"]
        else:
            np.save(data_path + "dssp/" + ID, transform_dssp(padded_dssp_matrix))
    else:
        np.save(data_path + "dssp/" + ID, transform_dssp(dssp_matrix))
    return 0


def process_pssm(pssm_file):
    with open(pssm_file, "r") as f:
        lines = f.readlines()
    pssm_feature = []
    for line in lines:
        if line == "\n":
            continue
        record = line.strip().split()
        if record[0].isdigit():
            pssm_feature.append([int(x) for x in record[2:22]])
    pssm_feature = (np.array(pssm_feature) - Min_pssm) / (Max_pssm - Min_pssm)

    return pssm_feature


def process_hhm(hhm_file):
    with open(hhm_file, "r") as f:
        lines = f.readlines()
    hhm_feature = []
    p = 0
    while lines[p][0] != "#":
        p += 1
    p += 5
    for i in range(p, len(lines), 3):
        if lines[i] == "//\n":
            continue
        feature = []
        record = lines[i].strip().split()[2:-1]
        for x in record:
            if x == "*":
                feature.append(9999)
            else:
                feature.append(int(x))
        hhm_feature.append(feature)
    hhm_feature = (np.array(hhm_feature) - Min_hhm) / (Max_hhm - Min_hhm)

    return hhm_feature


def BLOSUM_embedding(ID, seq, data_path):
    seq_embedding = []
    with open("blosum_dict.pkl", "rb") as f:
        blosum_dict = pickle.load(f)
    for aa in seq:
        seq_embedding.append(blosum_dict[aa])
    seq_embedding = (np.array(seq_embedding) - Min_blosum) / (Max_blosum - Min_blosum)
    np.save(data_path + "blosum/" + ID, seq_embedding)


def MSA(ID, data_path):
    os.system("{0} -db {1} -num_iterations 3 -num_alignments 1 -num_threads 2 -query {3}{2}.fa -out {3}{2}.bla -out_ascii_pssm {3}pssm/{2}.pssm".format(PSIBLAST, UR90, ID, data_path))
    os.system("{0} -i {2}{1}.fa -ohhm {2}hhm/{1}.hhm -oa3m {2}{1}.a3m -d {3} -v 0 -maxres 40000 -cpu 6 -Z 0 -o {2}{1}.hhr".format(HHBLITS, ID, data_path, HHDB))
    pssm_matrix = process_pssm(data_path + "pssm/" + ID + ".pssm")
    np.save(data_path + "pssm/" + ID, pssm_matrix)
    hhm_matrix = process_hhm(data_path + "hhm/" + ID + ".hhm")
    np.save(data_path + "hhm/" + ID, hhm_matrix)


def feature_extraction(PDBID, pdb_file, chain, mode, data_path):
    ID = PDBID + chain

    PDB_seq, error_code = get_PDB(PDBID, pdb_file, chain, data_path)
    if error_code != 0:
        return error_code

    with open(data_path + ID + ".fa", "w") as f:
        f.write(">" + ID + "\n" + PDB_seq)

    if mode == "fast":
        BLOSUM_embedding(ID, PDB_seq, data_path)
    else:
        MSA(ID, data_path)

    error_code = get_dssp(ID, PDB_seq, data_path)
    if error_code != 0:
        return error_code

    error_code = get_distance_map(ID, PDB_seq, data_path)
    if error_code != 0:
        return error_code

    return 0


def predict(ID, data_path, mode):
    with open(data_path + ID + ".fa", "r") as f:
        seq = f.readlines()[1].strip()

    test_dataframe = pd.DataFrame({"ID": [ID]})
    pred_scores = [round(score, 4) for score in test(test_dataframe, data_path, mode)]

    GraphPPIS_threshold = (0.24 if mode == "fast" else 0.18)
    binary_preds = [1 if score >= GraphPPIS_threshold else 0 for score in pred_scores]

    with open(data_path + ID + "_pred_results.txt", "w") as f:
        f.write("The threshold of the predictive score to determine PPI sites is set to {}.\n".format(GraphPPIS_threshold))
        f.write("AA\tProb\tPred\n")
        for i in range(len(seq)):
            f.write(seq[i] + "\t" + str(pred_scores[i]) + "\t" + str(binary_preds[i]) + "\n")


def main(PDBID, pdb_file, chain, mode):
    PDBID = PDBID.lower()
    chain = chain.upper()
    ID = PDBID + chain

    jobID = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    data_path = "./data_{}/".format(jobID)
    if mode == "fast":
        dir_list = ["", "blosum/", "dssp/", "dismap/"]
    else:
        dir_list = ["", "pssm/", "hhm/", "dssp/", "dismap/"]
    for dir_name in dir_list:
        os.makedirs(data_path + dir_name)

    print("\nFeature extraction begins at {}.\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))
    error_code = feature_extraction(PDBID, pdb_file, chain, mode, data_path)
    if error_code == 1:
        print("\nError! The query protein dosen't exist!")
    elif error_code == 2:
        print("\nError! The query chain dosen't exist in this protein!")
    elif error_code != 0:
        print("Error! Error code {}. Please contact the authors of GraphPPIS.".format(error_code))
    else:
        print("\nFeature Extraction is done at {}.\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))
        print("Predicting...\n")
        predict(ID, data_path, mode)
        print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--protein", type = str, help = "PDBID (e.g. 3mcb)")
    parser.add_argument("-f", "--file", type = str, help = "PDB file (.pdb only)")
    parser.add_argument("-c", "--chain", type = str, help = "chain identifier")
    parser.add_argument("-m", "--mode", type = str, default = "fast", help = "fast (use BLOSUM62 + DSSP) or slow (use PSSM + HMM + DSSP)")
    args = parser.parse_args()

    if args.chain == None:
        print("Chain identifier is not provided!")
    elif args.chain not in string.ascii_letters:
        print("Invalid chain identifier!")
    elif args.mode not in ["fast", "slow"]:
        print("Invalid mode!")
    elif args.file: # input by file
        if args.file.endswith(".pdb") == False:
            print("only .pdb file is supported!")
        else:
            main("user", args.file, args.chain, args.mode)
    else: # input by PDBID
        if args.protein == None or len(args.protein) != 4:
            print("Invalid PDB ID!")
        else:
            main(args.protein, "", args.chain, args.mode)
