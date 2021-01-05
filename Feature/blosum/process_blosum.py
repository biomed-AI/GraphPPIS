import pickle
import numpy as np

with open("BLOSUM62.txt", "r") as f:
    lines = f.readlines()[1:]

blosum_matrix = []
blosum_dict = {}

for i in range(0, 20):
    record = lines[i].strip().split()
    aa = record[0]
    embedding = list(map(int, record[1:21]))
    blosum_dict[aa] = embedding
    blosum_matrix.append(embedding)

with open("blosum_dict.pkl", "wb") as f:
    pickle.dump(blosum_dict, f)

blosum_matrix = np.array(blosum_matrix)

print("Max:", np.max(blosum_matrix, axis = 0))
print("Min:", np.min(blosum_matrix, axis = 0))
