import csv
import sys
import os

def main():
    if len(sys.argv) != 3:
        print("Usage: python crop_matrix.py <input_file> <n>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    n = int(sys.argv[2])
    
    # Create output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_{n}x{n}.csv"
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Check if n is valid (considering headers)
    max_n = min(len(rows)-1, len(rows[0])-1)
    if n > max_n:
        print(f"Error: n cannot exceed {max_n}")
        sys.exit(1)
    
    # Crop to (n+1) x (n+1) to include headers
    cropped = []
    # Add header row
    cropped.append(rows[0][:n+1])
    # Add data rows with their headers
    for i in range(1, n+1):
        cropped.append(rows[i][:n+1])
    
    # Write to output file with quotes only on headers
    with open(output_file, 'w', newline='') as f:
        # Write header row with quotes
        header = [f'"{cell}"' for cell in cropped[0]]
        f.write(','.join(header) + '\n')
        
        # Write data rows with quotes only on first element (row header)
        for row in cropped[1:]:
            row_data = [f'"{row[0]}"']  # Quote the row header
            row_data.extend(row[1:])    # Add numerical values without quotes
            f.write(','.join(row_data) + '\n')
    
    print(f"Cropped matrix saved to: {output_file}")

if __name__ == "__main__":
    main()