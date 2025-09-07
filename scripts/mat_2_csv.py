import scipy.io
import pandas as pd

# Load the .mat file
mat_data = scipy.io.loadmat('/home/jvski/Documents/UTFPR/dupla_diplomacao/classes/thesis/test_external_code/try_minet/input_data/geneci/DREAM4/EVAL/pdf_size100_1.mat')  # Replace with your file path

# Inspect keys (variables) in the .mat file
print(mat_data.keys())

# Extract the variable you want to convert (e.g., 'X')
data = mat_data['auroc_h']  # Replace 'X' with your variable name

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
print(df)
df.to_csv('pdf_size100_1_converted.csv', index=False, header=False)  # Adjust header/index as neededpyto