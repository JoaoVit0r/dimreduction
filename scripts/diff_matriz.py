import sys
import os

def read_matrix(filepath):
    with open(filepath, 'r') as file:
        return [[float(value) for value in line.strip().split()] for line in file]

def write_matrix(filepath, matrix):
    with open(filepath, 'w') as file:
        for row in matrix:
            file.write('\t'.join(map(str, row)) + '\n')

def compute_difference_matrix(matrix1, matrix2):
    rows = max(len(matrix1), len(matrix2))
    cols = max(len(matrix1[0]), len(matrix2[0]))
    difference_matrix = []
    non_zero_count = 0
    for i in range(rows):
        row = []
        for j in range(cols):
            v1 = matrix1[i][j] if i < len(matrix1) and j < len(matrix1[0]) else None
            v2 = matrix2[i][j] if i < len(matrix2) and j < len(matrix2[0]) else None
            if v1 is None or v2 is None:
                diff = None
            else:
                diff = v1 - v2
            row.append(diff)
            if diff != 0:
                non_zero_count += 1
        difference_matrix.append(row)
    return difference_matrix, non_zero_count

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file1> <file2>")
        sys.exit(1)
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    matrix1 = read_matrix(file1)
    matrix2 = read_matrix(file2)
    difference_matrix, non_zero_count = compute_difference_matrix(matrix1, matrix2)
    output_folder = os.path.dirname(file2)
    output_filename = f"diff_{os.path.basename(file1)}___{os.path.basename(file2)}___{non_zero_count}.txt"
    output_filepath = os.path.join(output_folder, output_filename)
    write_matrix(output_filepath, difference_matrix)
    print(f"[DIFF-COUNT] {file1} vs {file2}: {non_zero_count} differing elements")
    print(f"Output written to: {output_filepath}")

if __name__ == "__main__":
    main()