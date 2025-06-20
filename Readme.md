# DimReduction: Bioinformatics Tool for Network Inference and Feature Selection

## Overview

DimReduction is a comprehensive bioinformatics tool designed for gene network inference and feature selection analysis. It implements advanced methods for analyzing gene expression data, inferring regulatory relationships, and performing dimensionality reduction. The tool supports both time-series and steady-state expression data analysis using information-theoretic approaches.

The software features a robust command-line interface that processes biological data matrices, performs quantization and feature selection, and ultimately generates gene regulatory network models using entropy-based metrics. This allows researchers to identify significant regulatory relationships within complex biological datasets.

## Key Features

### Network Inference

- **Gene Regulatory Network Reconstruction**: Infer relationships between genes based on their expression patterns
- **Multi-approach Support**: Analyze both time-series data (temporal dependencies) and steady-state data (condition-specific relationships)
- **Information Theory Metrics**: Utilize entropy-based measures to evaluate non-linear regulatory interactions

### Data Processing

- **Quantization**: Discretize continuous expression data using adaptive thresholding
  - Column-based (feature/sample-wise) quantization
  - Row-based (gene/variable-wise) quantization
- **Matrix Operations**: Handle transposition, normalization, and preprocessing

### Feature Selection Algorithms

- **Sequential Forward Selection (SFS)**: Incrementally build feature sets by adding features that maximize information gain
<!-- - **Sequential Floating Forward Selection (SFFS)**: Extension of SFS with backtracking capability
- **Exhaustive Search**: Complete search of all possible feature subsets (with computational constraints) -->

### Performance Optimization

- **Multithreading Support**: Parallel processing for computationally intensive tasks
- **Configurable Thread Distribution**: Options for sequential, demand-based, or spaced distribution

## Dependencies

The tool depends are already included in the repository and used on the `run.sh` script.

On development, the Java version used was OpenJDK 21. Ensure you have Java installed on your system.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/dimreduction.git
   cd dimreduction
   ```

2. Create a configuration file:

   ```bash
   cp dimreduction-java/.env.example dimreduction-java/.env
   ```

3. Edit the `.env` file to match your requirements (see Configuration section below)

4. Run the tool:

   ```bash
   dimreduction-java/run.sh
   ```

## Configuration

The tool is configured via a `.env` file containing key parameters. The `.env.example` an example with explanations.

### Key Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `OUTPUT_FOLDER` | Directory where results will be saved |
| `INPUT_FILE_PATH` | Path to the input data file |
| `QUANTIZATION_VALUE` | Number of discrete levels for quantization |
| `APPLY_QUANTIZATION_TYPE` | How to apply quantization (0: none, 1: columns, 2: rows) |
| `TARGET_INDEXES` | Specific gene indices to analyze (comma-separated) |
| `TIME_SERIES_DATA` | Whether data represents time series (true) or steady-state (false) |
| `THREAD_DISTRIBUTION` | Strategy for distributing computation across threads |

## Usage

To run the tool with the configuration specified in your `.env` file:

```bash
dimreduction-java/run.sh
```

The program will:

1. Read the input data file
2. Apply quantization if configured
3. Perform network inference
4. Save results to the specified output folder

## Input Format

The tool accepts delimited text files (space, tab, or semicolon separated) with the following structure:

- Optional row of column headers (gene names or time points)
- Optional column of row headers (sample IDs or gene names)
- Numeric data matrix representing gene expression values

Example:

```plain
G1	G2	G3
7.0	9.3	9.5
7.2	9.2	9.4
```

## Output Format

The tool generates several outputs:

1. **Quantized Data**: Discretized version of the input data (if quantization is enabled)

   ```plain
   0.0	0.0	0.0
   1.0	1.0	1.0
   1.0	1.0	1.0
   ```

2. **Network Inference Results**: Adjacency matrix representing gene regulatory relationships

   ```plain
   0	0	0
   0	0	1
   1	0	0
   ```

   Where a value of 1 at position [i,j] indicates that gene i regulates gene j.

## Application Scenarios

### Gene Regulatory Network Inference

```bash
# Configure for network inference from steady-state data
TIME_SERIES_DATA=false
APPLY_QUANTIZATION_TYPE=1
QUANTIZATION_VALUE=2
THRESHOLD=0.3
```

### Time Series Data Analysis

```bash
# Configure for analyzing time series expression data
TIME_SERIES_DATA=true
IS_IT_PERIODIC=false
APPLY_QUANTIZATION_TYPE=1
QUANTIZATION_VALUE=3
```

### High-Performance Analysis of Large Datasets

The `THREAD_DISTRIBUTION` parameter allows you choice of how to distribute computation across available threads, optimizing performance for large datasets.

The options are:

- `sequential`: Slice the target list into N contiguous groups, each thread processes its group sequentially.
- `demand`: Each thread grabs the next available target from a shared index, allowing dynamic load balancing.
- `spaced`: Each thread processes every N-th target, where N is the number of threads, allowing for spaced processing.

```bash
# Configure for parallel processing
NUMBER_OF_THREADS=8
THREAD_DISTRIBUTION=demand
```

## Contributors and Acknowledgements

This project was modified by:

- João Vítor Fuzetti da Cunha (joaocunha@alunos.utfpr.edu.br)
- Fabricio Martins Lopes (fabricio@utfpr.edu.br)
- José Rufino (rufino@ipb.pt)

This tool was originally developed by:

- Fabricio Martins Lopes (fabriciolopes@vision.ime.usp.br)
- Roberto Marcondes Cesar Junior (cesar@vision.ime.usp.br)
- Luciano da Fontoura Costa (luciano@ifsc.usp.br)

## References

- Lopes, F.M., Martins, D.C. & Cesar, R.M. Feature selection environment for genomic applications. BMC Bioinformatics 9, 451 (2008). https://doi.org/10.1186/1471-2105-9-451

## License

This project is licensed under the terms of the included LICENSE file.