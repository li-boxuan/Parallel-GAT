import numpy as np


def main():
    np.random.seed(0)
    num_nodes = 5000
    connected_pctg = 0.01
    num_potential_edges = num_nodes * (num_nodes - 1) // 2
    mask = np.random.binomial(n=1, p=connected_pctg, size=num_potential_edges)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    cursor = 0
    for idx in range(num_nodes):
        cursor_start = cursor
        adj_matrix[idx][idx] = 1
        # 5 nodes, 5 5 5 5 4 4 4 3 3 2, num_nodes - 1 + .. +
        while cursor < (2 * num_nodes - idx - 2) * (idx + 1) // 2:
            if mask[cursor]:
                adj_matrix[idx][cursor - cursor_start + idx + 1] = 1
                adj_matrix[cursor - cursor_start + idx + 1][idx] = 1
            cursor += 1
    print(f"num_edges: {mask.sum()}, num_nonzero_entries: {adj_matrix.sum() - num_nodes}")
    elem_cnt = 0
    col_idx = []
    delim = [0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j]:
                col_idx.append(j)
                elem_cnt += 1
        delim.append(elem_cnt)

    with open("data/generated/5000.1e-2.adj.txt", "w") as fin:
        fin.write(f"{num_nodes} {elem_cnt}\n")
        for i, item in enumerate(col_idx):
            if i != elem_cnt - 1:
                fin.write(f"{item} ")
            else:
                fin.write(f"{item}\n")
        for i, item in enumerate(delim):
            if i != len(delim) - 1:
                fin.write(f"{item} ")
            else:
                fin.write(f"{item}\n")


if __name__ == "__main__":
    main()
