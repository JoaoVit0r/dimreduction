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
    rows = min(len(matrix1), len(matrix2))
    cols = min(len(matrix1[0]), len(matrix2[0]))
    difference_matrix = [[0] * cols for _ in range(rows)]
    non_zero_count = 0

    for i in range(rows):
        for j in range(cols):
            difference_matrix[i][j] = matrix1[i][j] - matrix2[i][j]
            if difference_matrix[i][j] != 0:
                non_zero_count += 1

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