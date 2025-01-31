import os

def get_latest_file(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    latest_file = sorted(files)[-1]
    return os.path.join(folder, latest_file)

def read_matrix(filepath):
    with open(filepath, 'r') as file:
        return [[float(value) for value in line.strip().split('\t')] for line in file]

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
    # folder1 = 'results/original_temporal_data'
    # folder2 = '/home/jvski/Documents/UTFPR/dupla_diplomacao/classes/thesis/virt_machine/java-dimreduction/results/original_temporal_data'
    
    # folder1 = 'results/quantized_temporal_data'
    # folder2 = '/home/jvski/Documents/UTFPR/dupla_diplomacao/classes/thesis/virt_machine/java-dimreduction/results/quantized_temporal_data'
    
    # folder1 = 'results/recovered_temporal_data'
    # folder2 = '/home/jvski/Documents/UTFPR/dupla_diplomacao/classes/thesis/virt_machine/java-dimreduction/results/recovered_temporal_data'

    # folder1 = 'results/quantized_data'
    # folder2 = '/home/jvski/Documents/UTFPR/dupla_diplomacao/classes/thesis/virt_machine/java-dimreduction/results/quantized_data'
    
    folder1 = 'results'
    folder2 = '/home/jvski/Documents/UTFPR/dupla_diplomacao/classes/thesis/virt_machine/java-dimreduction/results/'
    output_folder = 'results/diffs'

    latest_file1 = get_latest_file(folder1)
    latest_file2 = get_latest_file(folder2)

    matrix1 = read_matrix(latest_file1)
    matrix2 = read_matrix(latest_file2)

    difference_matrix, non_zero_count = compute_difference_matrix(matrix1, matrix2)

    output_filename = f"{os.path.basename(latest_file1)}___{os.path.basename(latest_file2)}___{non_zero_count}.txt"
    output_filepath = os.path.join(output_folder, output_filename)

    write_matrix(output_filepath, difference_matrix)

if __name__ == "__main__":
    main()