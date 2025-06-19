package fs;

import java.io.IOException;
import java.util.Vector;

import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JTable;

import java.time.Instant;

import org.jfree.data.category.DefaultCategoryDataset;

import agn.AGN;
import agn.AGNRoutines;
import agn.CNMeasurements;
import io.github.cdimascio.dotenv.Dotenv;
import utilities.IOFile;
import utilities.MathRoutines;
import utilities.Timer;

/**
 * MainCLI is the entry point for the CLI-based feature selection and network inference tool.
 * <p>
 * This class handles configuration initialization, data processing, feature selection,
 * and network inference. It is designed for command-line usage and replaces GUI-specific
 * functionality with logging and exceptions.
 * <p>
 * The application workflow consists of the following steps:
 * <ol>
 *   <li>Loading configuration from .env file</li>
 *   <li>Reading input data</li>
 *   <li>Applying quantization (optional)</li>
 *   <li>Looking for cycles (optional)</li>
 *   <li>Executing feature selection (optional)</li>
 *   <li>Performing network inference</li>
 * </ol>
 * 
 * @see agn.AGN
 * @see utilities.IOFile
 * @see utilities.Timer
 */
public class MainCLI {

    // Configuration parameters
    /** Output folder path where all results will be saved. */
    private static String outputFolder;
    /** Prefix used for naming output files. */
    private static String prefix;
    /** Path to the input data file containing the dataset to be processed. */
    private static String inputFilePath;
    /** Flag indicating if the input file has column descriptions in the first row. */
    private static boolean hasColumnDescription;
    /** Flag indicating if the input file has data titles in the first column. */
    private static boolean hasDataTitlesColumns;
    /** Flag indicating if the input matrix should be transposed before processing. */
    private static boolean hasTransposeMatrix;

    // Configuration parameters for quantization
    /** 
     * The quantization level used by the quantization process.
     * Represents the number of discrete values to use in quantization.
     */
    private static int quantizationValue;
    /** 
     * Flag indicating whether to search for cycles in the data.
     * When true, columns are transformed into string patterns to find repetitions.
     */
    private static boolean isToLookForCycles;
    /** 
     * Type of quantization to apply: 
     * 0 = no quantization, 1 = by columns (features/samples), 2 = by rows (variables).
     */
    private static int quantizationType;
    /** 
     * Flag indicating whether to save the quantized data to a file.
     */
    private static boolean isToSaveQuantizedData;

    // Configuration parameters for feature selection
    /** 
     * Flag indicating whether to execute the feature selection process.
     */
    private static boolean isToExecuteFeatureSelection;
    /** 
     * Criteria function to use during feature selection.
     * Different values represent different evaluation methods.
     */
    private static int criteriaFunctionFeatureSelection;
    /** 
     * q-Entropy parameter for feature selection.
     * Used when Tsallis entropy is selected as criteria function.
     */
    private static double qEntropyFeatureSelection;
    /** 
     * Penalization method for feature selection.
     * Controls how feature subsets are penalized during evaluation.
     */
    private static int penalizationMethodFeatureSelection;
    /** 
     * Alpha parameter for feature selection.
     * Used in penalization calculation.
     */
    private static double alphaFeatureSelection;
    /** 
     * Beta parameter for feature selection.
     * Used in penalization calculation.
     */
    private static float betaFeatureSelection;
    /** 
     * Path to the test set file for feature selection.
     * Used when evaluating feature subsets.
     */
    private static String inputTestSetFeatureSelection;
    /** 
     * Search method for feature selection.
     * Different values represent different search strategies:
     * 1 = SFS (Sequential Forward Selection),
     * 2 = Exhaustive Search,
     * 3 = SFFS (Sequential Floating Forward Selection).
     */
    private static int searchMethodFeatureSelection;
    /** 
     * Maximum set size for feature selection.
     * Limits the size of feature subsets considered.
     */
    private static int maximumSetSizeFeatureSelection;
    /** 
     * Maximum result list size for feature selection.
     * Limits the number of feature subsets returned.
     */
    private static int maximumResultListSizeFeatureSelection;

    // Configuration parameters for network inference
    /** 
     * Target indexes for network inference.
     * Comma-separated list of indexes for target genes/variables.
     */
    private static String targetIndexes;
    /** 
     * Flag indicating if targets should also be used as predictors.
     * When true, target variables can also be used to predict other targets.
     */
    private static boolean isTargetsAsPredictors;
    /** 
     * Flag indicating if the data represents a time series.
     * Affects how the network inference algorithm processes the data.
     */
    private static boolean isTimeSeriesData;
    /** 
     * Flag indicating if the time series data is periodic.
     * Used when isTimeSeriesData is true.
     */
    private static boolean isItPeriodic;
    /** 
     * Threshold value for determining edges in the inferred network.
     * Higher values produce sparser networks.
     */
    private static double threshold;
    /** 
     * Flag indicating if final inferred network data should be saved.
     */
    private static boolean isToSaveFinalData;
    // Add thread config fields
    /** 
     * Thread distribution strategy.
     * Controls how processing is split across multiple threads.
     */
    private static String threadDistribution;
    /** 
     * Number of threads to use for parallel processing.
     */
    private static int numberOfThreads;

    // Variables
    /** Timer for measuring execution time of different steps. */
    public static Timer timer = new Timer();
    
    /** 
     * Delimiter characters used for parsing input data files.
     * Includes space, tab, newline, carriage return, form feed, and semicolon.
     */
    public static String delimiter = String.valueOf(' ') + String.valueOf('\t') + String.valueOf('\n') + String.valueOf('\r') + String.valueOf('\f') + String.valueOf(';');
    
    /** Original matrix of data loaded from input file. */
    public static float[][] Mo = null;
    
    /** Working matrix used for processing data. */
    public static float[][] Md = null;
    
    /** Number of variables (rows) in the data matrices. */
    public static int lines = 0;
    
    /** Number of features or samples (columns) in the data matrices. */
    public static int columns = 0;
    
    /** Titles/names for data variables (rows). */
    public static Vector datatitles = null;
    
    /** Titles/names for features or samples (columns). */
    public static Vector featurestitles = null;
    
    /** Training set data for classifier evaluation. */
    private static float[][] trainingset = null;
    
    /** Test set data for classifier evaluation. */
    private static float[][] testset = null;
    
    /** Flag indicating if quantization has been applied to the data. */
    private static boolean flag_quantization = false;
    
    //atributo para armazenar as regras geradas automaticamente.
    //public static AGN agn = null;
    
    /** 
     * Network recovered/inferred by the algorithm.
     * Stores the result of the network inference process.
     */
    public static AGN recoverednetwork = null;
    
    // private static DefaultCategoryDataset dataset = null;
    
    /** 
     * Flag indicating if the last column contains class labels.
     * 0 = no labels, 1 = has labels.
     */
    private static int has_labels = 0;
    
    /** Help display window. */
    private static JFrame help = null;
    
    /** Flag to enable manual garbage collection. */
    private static boolean enableManualGC;

    /**
     * Main entry point for the CLI application.
     * <p>
     * Creates a new MainCLI instance which handles the entire pipeline.
     * 
     * @param args Command line arguments (not used).
     * @throws IOException If there is an error reading/writing files.
     */
    public static void main(String[] args) throws IOException {
        new MainCLI();
    }

    /**
     * Constructor that initializes and runs the data processing pipeline.
     * <p>
     * This constructor performs the following steps:
     * <ol>
     *   <li>Initializes configuration from .env file</li>
     *   <li>Reads data from the input file</li>
     *   <li>Applies quantization if configured</li>
     *   <li>Looks for cycles if configured</li>
     *   <li>Executes feature selection if configured</li>
     *   <li>Performs network inference</li>
     * </ol>
     * 
     * Each step is timed for performance analysis.
     */
    public MainCLI() {

        // Get the configuration parameters
        initConfig();

        timer.start("read_data");
        readDataActionPerformed();
        timer.end("read_data");

        // if (isToSaveReadData) {
        //     // Save the read data into a text file
        //     IOFile.SaveTable(Mo);
        // }

        if (quantizationType > 0 && quantizationType <= 2) {
            timer.start("apply_quantization");
            applyQuantizationAction(quantizationValue, quantizationType);
            
            if (isToSaveQuantizedData) {
                // Save the quantized data into a text file
                IOFile.WriteMatrix(outputFolder + "/quantized_data/" + prefix + "-quantized_data.txt", Md, "\t");
            }
            timer.end("apply_quantization");
        }

        if (isToLookForCycles) {
            timer.start("find_cycle");
            lookForCyclesActionPerformed();
            timer.end("find_cycle");
        }

        if (isToExecuteFeatureSelection) {
            timer.start("execute_feature_selection");
            executeFeatureSelectionActionPerformed();
            timer.end("execute_feature_selection");
        }

        timer.start("network_inference");
        IOFile.PrintlnAndLog(Instant.now() + "; start network inference - start", "timing/timers.log", 0);
        networkInferenceActionPerformed();
        IOFile.PrintlnAndLog(Instant.now() + "; start network inference - end", "timing/timers.log", 0);
        timer.end("network_inference");
    }

    /**
     * Checks if manual garbage collection is enabled in the configuration.
     * <p>
     * This method returns the value of the enableManualGC flag, which controls
     * whether explicit garbage collection (System.gc()) is called in
     * performance-critical areas like RadixSort. Manual garbage collection can
     * improve performance in some scenarios by reducing memory pressure.
     *
     * @return true if manual garbage collection is enabled, false otherwise.
     * @see utilities.RadixSort
     */
    public static boolean isManualGCEnabled() {
        return enableManualGC;
    }

    /**
     * Initializes the configuration parameters from the .env file.
     * <p>
     * This method loads the .env file, reads the configuration parameters,
     * and initializes the corresponding variables. It also creates the output
     * folder if it doesn't exist.
     * <p>
     * Configuration parameters include:
     * <ul>
     *   <li>Input/output file paths and options</li>
     *   <li>Quantization settings</li>
     *   <li>Feature selection parameters</li>
     *   <li>Network inference settings</li>
     *   <li>Threading and performance options</li>
     * </ul>
     * <p>
     * All configuration values are read with appropriate defaults when not specified.
     * Invalid configuration values will trigger exceptions with descriptive error messages.
     * 
     * @throws IllegalArgumentException If any of the configuration parameters have invalid values
     * @see io.github.cdimascio.dotenv.Dotenv
     *
     */
    private void initConfig() {
        // Load the .env file
        Dotenv dotenv = Dotenv.load();

        // - (Text Field) Output Folder: Fill this text box with the path of the output folder, where the results will be saved.
        outputFolder = dotenv.get("OUTPUT_FOLDER");
        // check if the output folder exists, if not, create it
        IOFile.Makedirs(outputFolder);

        int verbosityLevel = Integer.parseInt(dotenv.get("VERBOSITY_LEVEL", String.valueOf(IOFile.VERBOSE_DEBUG)));
        IOFile.setVerbosity(verbosityLevel);

        // - (Text Field) Input File: Store the path of the input data file.
        inputFilePath = dotenv.get("INPUT_FILE_PATH");

        // - (Check Box) Are column description in first row?: If input data file has sample/feature descriptions or titles in first row, mark this check box.
        hasColumnDescription = Boolean.parseBoolean(dotenv.get("ARE_COLUMNS_DESCRIPTIVE"));

        // - (Check Box) Are titles in first column?: If input data file has variables descriptions or titles in first column, mark this check box.
        hasDataTitlesColumns = Boolean.parseBoolean(dotenv.get("ARE_TITLES_ON_FIRST_COLUMN"));

        // - (Check Box) Transpose the matrix?: The variables must be at rows and features/samples or time series must be at columns. Mark this check box if needed.
        hasTransposeMatrix = Boolean.parseBoolean(dotenv.get("TRANSPOSE_MATRIX"));

        
        // # Quantization =======================================

        // - (Text Field) Quantity of Values: Fill this text box with an integer value, which represents the quantization level used by the quantization process.
        quantizationValue = Integer.parseInt(dotenv.get("QUANTIZATION_VALUE"));
        if (quantizationValue <= 0) {
            throw new IllegalArgumentException("The quantization value must be greater than zero.");
        }

        // - (Check Box) The last column stores the labels of the classes: If input data file has labels of the classes, these labels must be at last column and you must mark this check box. This column is used by the classifier at single or cross validation execution.
        if (Boolean.parseBoolean(dotenv.get("HAS_LABELS_CLASSES_ON_LAST_COLUMN"))) {
            has_labels = 1;
        }

        // - (Button) Cycle?: The columns are transformed in a string representation, and then a sequential search is performed to find columns with same string pattern.
        isToLookForCycles = Boolean.parseBoolean(dotenv.get("LOOK_FOR_CYCLES"));

        // - (Button) Apply Quantization (columns): After selecting a discretization value, click here to perform the quantization considering feature/sample values (columns). The quantization process is applied to each feature separately, firstly taking the normal transform that attributes (value - average)/(standard deviation) to each value. The average and standard deviation are obtained considering all values of the feature (column). This values are separated in positives and negatives sets. Using the extreme values (positives and negatives) and quantization level, the quantized data is obtained by thresholds (positives and negatives) that separate the space in equal parts.
        // - (Button) Apply Quantization (rows): Do the same process as done by pressing the previous button, but for rows (variables) instead of columns (features/samples).
        quantizationType = Integer.parseInt(dotenv.get("APPLY_QUANTIZATION_TYPE"));
        if (quantizationType < 0 || quantizationType > 2) {
            throw new IllegalArgumentException("The quantization type must be 0 (no quantization), 1 (quantization by columns) or 2 (quantization by rows).");
        }

        // - (Button) Save Quantized Data: Click to save quantized data into a text file.
        isToSaveQuantizedData = Boolean.parseBoolean(dotenv.get("SAVE_QUANTIZED_DATA"));

                
        // # FeatureSelection =======================================
        isToExecuteFeatureSelection = Boolean.parseBoolean(dotenv.get("EXECUTE_FEATURE_SELECTION"));

        // - (Combo Box) Criterion Function: Select the criterion function based on classifier information (mean conditional entropy) or based on classifier error (CoD - Coefficient of Determination).
        criteriaFunctionFeatureSelection = Integer.parseInt(dotenv.get("CRITERION_FUNCTION_FEATURE_SELECTION", "0"));
        // 0: mean conditional entropy, 1: coefficient of determination
        if (criteriaFunctionFeatureSelection < 0 || criteriaFunctionFeatureSelection > 1) {
            throw new IllegalArgumentException("The criterion function must be 0 (mean conditional entropy) or 1 (coefficient of determination).");
        }
        
        // - (Text Field) q-entropy: The value defined in this text field represents the type of entropy considered in criterion function. Use 1 to apply Shannon Entropy. Use a value different from 1 to apply Tsallis Entropy (smaller than 1 for subextensive entropy or larger than 1 for superextensive entropy). This parameter is used only by the criterion function based on entropy.
        qEntropyFeatureSelection = Double.parseDouble(dotenv.get("Q_ENTROPY_FEATURE_SELECTION", "1.0")); // need to be 0 , if criterion function is based on CoD
        
        // - (Combo Box) Penalization Method:
        // - no_obs: apply penalty for non-observed instances.
        // - poor_obs: apply penalty for poorly observed instances (observed only once).
        penalizationMethodFeatureSelection = Integer.parseInt(dotenv.get("PENALIZATION_METHOD_FEATURE_SELECTION", "0"));
        // 0: no_obs, 1: poor_obs
        if (penalizationMethodFeatureSelection < 0 || penalizationMethodFeatureSelection > 1) {
            throw new IllegalArgumentException("The penalization method must be 0 (no_obs) or 1 (poor_obs).");
        }
        
        // - (Text Field) Alpha (value for penalty): Only used by no_obs penalization method. Alpha value represents the probability mass for the non-observed instances. This parameter is added to the relative frequency (number of occurrences) of all possible instances.
        alphaFeatureSelection = Double.parseDouble(dotenv.get("ALPHA_FEATURE_SELECTION", "1.0"));
        
        // - (Slider Bar) Beta (value of confidence): Only used by poor_obs penalization method. Beta value is attributed to the conditional probability of the observed class, given the instance. The value (1-Beta) is equally distributed over non-observed classes of the considered instance.
        betaFeatureSelection = Float.parseFloat(dotenv.get("BETA_FEATURE_SELECTION", "80"));
        
        // - (Text Field) Input Test Set (optional): Click on File button or fill this text box with the path of a data file, which is used to perform test of the classifier. If this text field has no information, the input data file is used to perform training and test of the classifier (resubstitution error method).
        inputTestSetFeatureSelection = dotenv.get("INPUT_TEST_SET_FEATURE_SELECTION", null);
        
        // - (Option Group) Search Method:
        // - SFS: Apply SFS algorithm to perform feature selection.
        // - Exhaustive Search: Apply Exhaustive algorithm for feature selection.
        // - SFFS: Apply SFFS algorithm to perform feature selection.
        searchMethodFeatureSelection = Integer.parseInt(dotenv.get("SEARCH_METHOD_FEATURE_SELECTION", "1"));
        if (searchMethodFeatureSelection < 1 || searchMethodFeatureSelection > 3) {
            throw new IllegalArgumentException("The search method must be 1 (SFS), 2 (Exhaustive Search) or 3 (SFFS).");
        }

        // - (Text Field) Maximum Set Size: Select the maximum cardinality of a feature set to perform the search. Use this option only for SFFS algorithm.
        maximumSetSizeFeatureSelection = Integer.parseInt(dotenv.get("MAXIMUM_SET_SIZE_FEATURE_SELECTION", "3"));
        // - (Text Field) Size of the Result List: This parameter allows to choose the size of the result list with the first ranked features sets.
        maximumResultListSizeFeatureSelection = Integer.parseInt(dotenv.get("SIZE_OF_THE_RESULT_LIST_FEATURE_SELECTION", "3"));
        

        // # NetworkInference =======================================
        threadDistribution = dotenv.get("THREAD_DISTRIBUTION", "sequential");
        String envThreads = dotenv.get("NUMBER_OF_THREADS", "1");
        try {
            numberOfThreads = Integer.parseInt(envThreads);
        } catch (NumberFormatException e) {
            numberOfThreads = 1;
        }

        // - (Text Field) Target's indexes: Fill this text box with predictors or targets indexes to find others variables (genes) related with them, and then click in the Network Inference button. If this text box is empty, all features are considered to graph generation.
        targetIndexes = dotenv.get("TARGET_INDEXES", null);
        
        // - (Check Box) Targets as Predictors?: Select this option to generate graph from targets (not selected) or predictors (selected).
        isTargetsAsPredictors = Boolean.parseBoolean(dotenv.get("TARGETS_AS_PREDICTORS"));
        
        // - (Option Group):
        // - time-series data: Select this option if your data represents a time-series gene expressions, i.e., it is considered by the method time-dependent relationship among the variables/genes (rows) and its observations/samples (columns). The predictors are observed at time t and the targets are observed at time t+1.
        // - steady-state data: Select this option if your data represents an independent gene expressions, i.e., it is considered by the method relationships among variables/genes (rows) only within each experiment (columns).
        isTimeSeriesData = Boolean.parseBoolean(dotenv.get("TIME_SERIES_DATA"));
        
        // - (Check Box) Is it periodic?: Mark this option to assume that time series is periodic, i. e. the last instant of time is connected to the first instant of time.
        isItPeriodic = Boolean.parseBoolean(dotenv.get("IS_IT_PERIODIC"));
        
        // - (Text Field) Threshold: Fill this text box with a real value to visualize all graph edges (near 1) or just most representative ones (near 0).
        threshold = Double.parseDouble(dotenv.get("THRESHOLD", "0.3"));

        isToSaveFinalData = Boolean.parseBoolean(dotenv.get("SAVE_FINAL_DATA"));
        
        // Config Outputs
        prefix = String.valueOf(Instant.now().getEpochSecond());
        isToSaveQuantizedData = outputFolder != null && !outputFolder.equals("");

        // - (Boolean) Enable Manual Garbage Collection: Controls whether System.gc() is called explicitly in performance-critical areas like RadixSort
        enableManualGC = Boolean.parseBoolean(dotenv.get("ENABLE_MANUAL_GC", "false"));
    }
    
    /**
     * Reads the input data file and initializes the data matrices.
     * 
     * <p>
     * This method prepares the environment for data loading by resetting data titles and 
     * quantization flags, then calls {@link #ReadInputData(String)} to load the data from 
     * the configured input file path. In the CLI version of the application, this method 
     * serves as a wrapper that ensures proper initialization of the data structures before 
     * loading data.
     * </p>
     * 
     * <p>
     * The method clears any existing data titles and resets the quantization flag before
     * loading new data. This prevents issues that could arise from remnants of previous
     * data loading operations affecting the current one.
     * </p>
     * 
     * @see #ReadInputData(String) The core method that performs the actual data loading
     * @see #initConfig() Where the inputFilePath configuration is initially set
     */
    private void readDataActionPerformed() {
        // Reset data structures before loading new data
        datatitles = null;
        featurestitles = null;
        flag_quantization = false;
        
        // Load data from the configured input file path
        ReadInputData(inputFilePath);
    }

    /**
     * Reads the input data from the specified file path and stores it in the data matrices.
     * 
     * <p>
     * This method is responsible for loading and preparing the data matrices that will be used
     * by the gene network analysis pipeline. It intelligently processes input files based on
     * configuration parameters to handle various file formats and data arrangements.
     * </p>
     * 
     * <p>
     * Data loading occurs in several sequential steps:
     * <ol>
     *   <li>File format detection (currently limited to delimited text files; AGN format not supported)</li>
     *   <li>Reading column descriptions/titles from the first row if {@link #hasColumnDescription} is true</li>
     *   <li>Reading row descriptions/titles from the first column if {@link #hasDataTitlesColumns} is true</li> 
     *   <li>Reading the numeric data into a matrix, skipping the header row and/or column as needed</li>
     *   <li>Transposing the matrix if {@link #hasTransposeMatrix} is true to ensure variables are rows and features are columns</li>
     *   <li>Updating dimensions and creating a working copy of the data for subsequent processing</li>
     * </ol>
     * </p>
     * 
     * <p>
     * When transposing the matrix, the method also handles swapping the first elements of row and column
     * titles to maintain consistency between titles and the data they represent. This ensures that
     * labels correctly map to the transposed data.
     * </p>
     * 
     * <p>
     * This method updates several important class fields:
     * <ul>
     *   <li>{@link #Mo} - Original data matrix, containing the raw input values</li>
     *   <li>{@link #Md} - Working data matrix (initialized as a copy of Mo), which will be manipulated by later operations</li>
     *   <li>{@link #lines} - Number of rows (variables/genes) in the matrices</li>
     *   <li>{@link #columns} - Number of columns (samples/features/experiments) in the matrices</li>
     *   <li>{@link #datatitles} - Variable/gene titles or identifiers (if available)</li>
     *   <li>{@link #featurestitles} - Sample/feature titles or identifiers (if available)</li>
     * </ul>
     * </p>
     *
     * @param path The absolute path to the input data file to read.
     * @throws FSException If the file reading fails, if the file format is not supported,
     *                     or if the resulting matrix is empty.
     * @see utilities.IOFile#ReadDataFirstRow(String, int, int, String) For reading column titles/headers
     * @see utilities.IOFile#ReadDataFirstCollum(String, int, String) For reading row titles/identifiers
     * @see utilities.IOFile#ReadMatrix(String, int, int, String) For reading the main data matrix
     * @see utilities.MathRoutines#TransposeMatrix(float[][]) For transposing the matrix when needed
     * @see #initConfig() Where the configuration parameters are initially set
     */
    public synchronized void ReadInputData(String path) {
        int startrow = 0;
        int startcolumn = 0;
        try {
            if (path.endsWith("agn")) {
                throw new FSException("AGN file format not supported.", false);
            } else {
                if (hasColumnDescription) {
                    featurestitles = IOFile.ReadDataFirstRow(path, 0, 0, delimiter);
                    startrow = 1;
                }
                if (hasDataTitlesColumns) {
                    datatitles = IOFile.ReadDataFirstCollum(path, startrow, delimiter);
                    startcolumn = 1;
                }
                Mo = IOFile.ReadMatrix(path, startrow, startcolumn, delimiter);
            }

            if (Mo == null) {
                throw new FSException("Error when reading input file. The matrix is empty.", false);
            }

            if (hasTransposeMatrix) {
                Mo = MathRoutines.TransposeMatrix(Mo);
                String zz = null;
                if (featurestitles != null) {
                    zz = (String) featurestitles.remove(0);
                    if (datatitles != null) {
                        datatitles.add(0, zz);
                    }
                }
            }
        } catch (IOException error) {
            throw new FSException("Error when reading input file. " + error,
                    false);
        } catch (NumberFormatException error) {
            throw new FSException("Error when reading input file. " + error,
                    false);
        }

        lines = Mo.length;
        columns = Mo[0].length;
        Md = Preprocessing.copyMatrix(Mo);//copy of matrix
    }

    /**
     * Applies quantization to the working data matrix (Md) using the specified parameters.
     * <p>
     * This method copies the original data matrix (Mo) to the working matrix (Md),
     * then applies quantization either by columns or by rows, depending on the 'type' parameter.
     * Quantization is performed using the Preprocessing class methods.
     * <ul>
     *   <li>If type == 1: Quantization is applied by columns (features/samples).</li>
     *   <li>If type != 1: Quantization is applied by rows (variables).</li>
     * </ul>
     * The quantized data is stored in Md. If Mo is null, a GUI error dialog is shown.
     *
     * @param qtvalues The quantization degree (number of discrete values to use in quantization).
     * @param type The quantization type: 
     *             1 = by columns (features/samples) - applies normalization by column,
     *             2 = by rows (variables) - applies normalization by row.
     * @throws FSException If the operation fails due to an invalid input matrix.
     * @see Preprocessing#quantizecolumns(float[][], int, boolean, int)
     * @see Preprocessing#quantizerows(float[][], int, boolean, int) 
     */
    public void applyQuantizationAction(int qtvalues, int type) {
        if (Mo != null) {
            Md = Preprocessing.copyMatrix(Mo);//copy of matrix

            if (type == 1) //traditional quantization, apply normalization and
            //creates threshold values to positive and negative values.
            {
                Preprocessing.quantizecolumns(Md, qtvalues, true, has_labels);
            } else {
                Preprocessing.quantizerows(Md, qtvalues, true, has_labels);
                //Preprocessing.normalize(Md, qtvalues, has_labels); // has commented
            }
            
            flag_quantization = true;
            
        } else {
            IOFile.PrintlnAndLog("Execution Error: Select and read input file first.", IOFile.VERBOSE_ERROR);
        }
    }

    /**
     * Looks for cycles in the quantized data matrix (Md).
     * <p>
     * This method searches for repeating patterns in the data, which can indicate
     * cyclic behavior in gene expression or other temporal patterns. It transforms
     * the columns into string representations and then performs a sequential search
     * to find columns with the same string pattern.
     * <p>
     * The method acts on the working data matrix (Md) and requires that quantization
     * has been applied before calling this method for meaningful results.
     * 
     * @throws FSException If the working data matrix (Md) is null or invalid.
     * @see agn.CNMeasurements#FindCycle(float[][])
     */
    private void lookForCyclesActionPerformed() {
        if (Md != null) {
            CNMeasurements.FindCycle(Md);
        } else {
            IOFile.PrintlnAndLog("Execution Error: Select and read input file first.", IOFile.VERBOSE_ERROR);
        }
    }

    /**
     * Executes the feature selection process in a separate thread.
     * <p>
     * This method validates the parameters, starts a dedicated thread for feature selection,
     * and waits for the thread to complete execution. Running feature selection in a 
     * separate thread allows the application to remain responsive during this
     * computationally intensive task.
     * <p>
     * The feature selection process uses the configured parameters including:
     * <ul>
     *   <li>q-entropy - determines the type of entropy (Shannon vs. Tsallis)</li>
     *   <li>alpha - parameter for no_obs penalization method</li>
     *   <li>search method - SFS, SFFS, or Exhaustive Search</li>
     * </ul>
     * 
     * @throws FSException if the thread is interrupted, parameter validation fails,
     *                    or the feature selection process encounters an error.
     * @see #executeFeatureSelection(int)
     */
    private void executeFeatureSelectionActionPerformed() {
        double alpha = alphaFeatureSelection;
        double q_entropy = qEntropyFeatureSelection;

        if (q_entropy < 0 || alpha < 0) {
            IOFile.PrintlnAndLog("Error on parameter value: The values of q-entropy and Alpha must be positives.", IOFile.VERBOSE_ERROR);
            return;
        }
        class Thread1 extends Thread {

            @Override
            public void run() {
                timer.start("feature_selection_inside_thread");
                try {
                    if (searchMethodFeatureSelection > 0 && searchMethodFeatureSelection <= 3) {
                        executeFeatureSelection(searchMethodFeatureSelection);
                    } else {
                        IOFile.PrintlnAndLog("Error on parameter value: The search method must be selected.", IOFile.VERBOSE_ERROR);
                    }

                } catch (IOException error) {
                    timer.end("feature_selection_inside_thread");
                    throw new FSException("Error on Execution of the Search" +
                            " Method." + error, false);
                }
                timer.end("feature_selection_inside_thread");
            }
        }
        Thread thread = new Thread1();
        thread.setPriority(Thread.NORM_PRIORITY);
        thread.setName("SE");
        thread.start();

        try {
            thread.join();
        } catch (InterruptedException error) {
            throw new FSException("Interrupted Thread." + error, false);
        }
    }

    /**
     * Executes the feature selection algorithm based on the selected method.
     * <p>
     * Handles SFS, SFFS, and Exhaustive Search, logs results, and runs a classifier.
     *
     * @param selector The feature selection method:
     *                 1 = SFS (Sequential Forward Selection),
     *                 2 = Exhaustive Search,
     *                 3 = SFFS (Sequential Floating Forward Selection).
     * @param selector The feature selection method to execute:
     *                1 = SFS (Sequential Forward Selection),
     *                2 = Exhaustive Search,
     *                3 = SFFS (Sequential Floating Forward Selection).
     * @throws IOException If an error occurs during file operations or result saving.
     * @throws FSException If parameter validation fails or feature selection encounters errors.
     * @see fs.FS#runSFS(boolean, int)
     * @see fs.FS#runSFFS(int, int, Vector)
     * @see fs.FS#runExhaustive(int, int, Vector)
     */
    public void executeFeatureSelection(int selector) throws IOException {
        // String penalization_type = (String) jCB_PenalizationSE.getSelectedItem();
        // float alpha = ((Double) jS_AlphaSE.getValue()).floatValue();
        // float q_entropy = ((Double) jS_QEntropySE.getValue()).floatValue();
        // float beta = ((float) jSliderBetaSE.getValue() / 100);
        String penalization_type = penalizationMethodFeatureSelection == 0 ? "no_obs" : "poor_obs";
        float alpha = (float) alphaFeatureSelection;
        float q_entropy = (float) qEntropyFeatureSelection;
        float beta = betaFeatureSelection / 100;


        //if selected criterion function is CoD, q_entropy = 0
        // if (jCB_CriterionFunctionSE.getSelectedIndex() == 1) {
        if (criteriaFunctionFeatureSelection == 1) {
            q_entropy = 0;
        }//CoD

        if (q_entropy < 0 || alpha < 0) {
            //entrada de dados invalida.
            IOFile.PrintlnAndLog("Error on parameter value: The values of q-entropy and Alpha must be positives.", IOFile.VERBOSE_ERROR);
            return;
        }

        // jTA_SaidaSE.setText("");
        // jTA_SelectedFeaturesSE.setText("");

        // n = quantidade de valores assumidos pelas caracteristicas.
        int n = Main.maximumValue(Md, 0, lines - 1, 0, columns - 2) + 1;

        //c = numero de rotulos possiveis para as classes.
        int c = Main.maximumValue(Md, 0, lines - 1, columns - 1, columns - 1) + 1;

        // jProgressBarSE.setValue(5);
        Thread.yield();

        char[][] strainingset = MathRoutines.float2char(Md);
        char[][] stestset = null;

        // if (!jTF_InputTestSE.getText().equals("")) {
        //     stestset = IOFile.ReadMatrix(jTF_InputTestSE.getText(), delimiter);
        if (inputTestSetFeatureSelection != null && !inputTestSetFeatureSelection.equals("")) {
            stestset = IOFile.ReadMatrix(inputTestSetFeatureSelection, delimiter);
        } else {
            stestset = MathRoutines.float2char(Md);
        }

        int resultsetsize = 1;
        try {
            //vetor com os resultados da selecao de caracteristica.
            // resultsetsize = (Integer) jS_MaxResultListSE.getValue();
            resultsetsize = maximumResultListSizeFeatureSelection;
            if (resultsetsize < 1) {
                Thread.yield();
                IOFile.PrintlnAndLog("Error on parameter value: The Size of the Result List must be a integer value greater or equal to 1.");
                return;
            }
        } catch (NumberFormatException error) {
            Thread.yield();
            IOFile.PrintlnAndLog("Error on parameter value: The Size of the Result List must be a integer value greater or equal to 1.");
            return;
        }

        /* SELETOR DE CARACTERISTICAS PARA O TREINAMENTO. */
        FS fs = new FS(strainingset, n, c, penalization_type, alpha, beta, q_entropy,
                resultsetsize);

        // jProgressBarSE.setValue(10);
        Thread.yield();

        // int maxfeatures = (Integer) jS_MaxSetSizeSE.getValue();
        int maxfeatures = maximumSetSizeFeatureSelection;
        if (maxfeatures <= 0) {
            IOFile.PrintlnAndLog("Error on parameter value: The Maximum Set Size be a integer value greater or equal to 1.");
            return;
        }

        if (selector == 1) {
            fs.runSFS(false, maxfeatures);
            // jProgressBarSE.setValue(90);
            Thread.yield();
        } else if (selector == 3) {
            fs.runSFFS(maxfeatures, -1, null);
            // jProgressBarSE.setValue(90);
            Thread.yield();
        } else if (selector == 2) {
            fs.runSFS(true, maxfeatures); /* a call to SFS is made in order to get the
            //ideal dimension to run the exhaustive search;*/
            int itmax = fs.itmax;
            if (itmax < maxfeatures) {
                itmax = maxfeatures;
            }
            /*calculating the estimated time to be completed in rea
            computer of 2 GHz*/
            int combinations = 0;
            for (int i = 1; i <= itmax; i++) {
                combinations += MathRoutines.numberCombinations(columns - 1, i);
            }
            double estimatedTime = (0.0062 + 3.2334e-7 * strainingset.length) *
                    combinations * Math.log(combinations) / Math.log(2);
            // int answer = JOptionPane.showConfirmDialog(null, "Estimated " +
            //         "time to finish: " + estimatedTime + " s.\n Do you want to" +
            //         " continue?", "Exhaustive Search",
            //         JOptionPane.YES_NO_OPTION);
            // if (answer == 1) {
            //     // jProgressBarSE.setValue(0);
            //     return;
            // }
            IOFile.PrintlnAndLog("Estimated time to finish: "+estimatedTime+" s");
            float new_itmax = itmax;
            FS fsPrev = new FS(strainingset, n, c, penalization_type, alpha, beta,
                    q_entropy, resultsetsize);
            for (int i = 1; i <= itmax; i++) {
                IOFile.PrintlnAndLog("Iteration " + i);
                fs = new FS(strainingset, n, c, penalization_type, alpha, beta,
                        q_entropy, resultsetsize);
                fs.itmax = i;
                fs.runExhaustive(0, 0, fs.I);
                //if (fs.hGlobal == 0) {
                //    break;
                //}
                if (fs.hGlobal < fsPrev.hGlobal) {
                    fsPrev = fs;
                } else {
                    fs = fsPrev;
                    break;
                }
                float new_i = i;
                float pb = (new_i / new_itmax);
                pb = (pb * 80.0f);
                // jProgressBarSE.setValue(10 + (int) pb);
                Thread.yield();
            }
        }

        // /*
        // jTA_SelectedFeaturesSE.setText("1st Global Criterion Function Value: " + fs.hGlobal);
        // jTA_SelectedFeaturesSE.append("\nSelected Features: ");
        // for (int i = 0; i < fs.I.size(); i++) {
        // jTA_SelectedFeaturesSE.append(fs.I.elementAt(i) + " ");
        // }
        //  * substituido pelo codigo abaixo para exibir uma lista de resultados
        //  * de tamanho escolhido pelo usuario.
        //  */
        for (int i = 0; i < fs.resultlist.size(); i++) {
            float fsvalue = ((Float) fs.resultlist.get(i).get(0));
            if (i == 0) {
                // jTA_SelectedFeaturesSE.setText((i + 1) + "st Global Criterion" +
                //         " Function Value: " + fsvalue);
                IOFile.PrintlnAndLog((i + 1) + "st Global Criterion Function Value: " + fsvalue);
            } else if (i == 1) {
                // jTA_SelectedFeaturesSE.append("\n\n" + (i + 1) + "nd Global" +
                //         " Criterion Function Value: " + fsvalue);
                IOFile.PrintlnAndLog((i + 1) + "nd Global Criterion Function Value: " + fsvalue);
            } else if (i == 2) {
                // jTA_SelectedFeaturesSE.append("\n\n" + (i + 1) + "rd Global" +
                //         " Criterion Function Value: " + fsvalue);
                IOFile.PrintlnAndLog((i + 1) + "rd Global Criterion Function Value: " + fsvalue);
            } else {
                // jTA_SelectedFeaturesSE.append("\n\n" + (i + 1) + "th Global" +
                //         " Criterion Function Value: " + fsvalue);
                IOFile.PrintlnAndLog((i + 1) + "th Global Criterion Function Value: " + fsvalue);
            }
            // jTA_SelectedFeaturesSE.append("\nSelected Features: ");
            IOFile.PrintlnAndLog("Selected Features: ");
            Vector features = (Vector) fs.resultlist.get(i).get(1);
            for (int j = 0; j < features.size(); j++) {
                // jTA_SelectedFeaturesSE.append((Integer) features.get(j) + " ");
                IOFile.PrintlnAndLog((Integer) features.get(j) + " ");
            }
        }

        // CLASSIFICADOR.
        Classifier clas = new Classifier();
        clas.classifierTable(strainingset, fs.I, n, c);

        for (int i = 0; i < clas.table.size(); i++) {
            double[] tableLine = (double[]) clas.table.elementAt(i);
            double instance = (Double) clas.instances.elementAt(i);
            IOFile.PrintAndLog(instance + " ");
            for (int j = 0; j < c; j++) {
                IOFile.PrintAndLog((int) tableLine[j] + " ");
            }
            IOFile.PrintlnAndLog();
        }

        // jProgressBarSE.setValue(95);
        Thread.yield();

        double[] instances = clas.classifyTestSamples(stestset, fs.I, n, c);

        // jTA_SaidaSE.setText("Correct Labels  -  Classified Labels - " +
        //         "Classification Instances\n(Considering the first selected" +
        //         " features)\n");
        IOFile.PrintlnAndLog("Correct Labels  -  Classified Labels - Classification Instances\n(Considering the first selected features)");
        double hits = 0;
        for (int i = 0; i < clas.labels.length; i++) {
            int correct_label = (int) stestset[i][columns - 1];
            int classified_label = (int) clas.labels[i];
            // jTA_SaidaSE.append("\n" + correct_label + "  -  " +
            //         classified_label + "  -  " + instances[i]);
            IOFile.PrintlnAndLog(correct_label + "  -  " + classified_label + "  -  " + instances[i]);
            if (correct_label == classified_label) {
                hits++;
            }

            Thread.yield();
        }
        double hit_rate = hits / clas.labels.length;
        // jTA_SaidaSE.append("\nrate of hits = " + hit_rate);
        IOFile.PrintlnAndLog("rate of hits = " + hit_rate);
        // jProgressBarSE.setValue(100);
        Thread.yield();
    }


    /**
     * Executes network inference on the data to reconstruct the gene regulatory network.
     * <p>
     * This method performs network inference using the configured parameters and
     * the previously loaded and potentially quantized data. It creates a gene network 
     * representation by inferring relationships between genes or variables based on 
     * their expression patterns, using feature selection algorithms to identify the 
     * most informative predictors for each target gene.
     * </p>
     * 
     * <p>
     * The method supports two distinct types of data analysis approaches:
     * <ul>
     *   <li><strong>Time-series data</strong>: Relationships between variables are time-dependent,
     *       with predictors at time t and targets at time t+1. This is appropriate
     *       when data represents sequential measurements over time, where gene
     *       expression at one time point may influence expression at the next.</li>
     *   <li><strong>Steady-state data</strong>: Relationships between variables are determined
     *       within each experiment/column. This is appropriate for data from independent
     *       experiments or conditions without temporal dependence.</li>
     * </ul>
     * </p>
     * 
     * <p>
     * The method uses advanced information theory metrics (entropy-based measures) to 
     * evaluate relationships between genes, allowing for complex non-linear regulatory 
     * interactions to be discovered. The process involves:
     * <ol>
     *   <li>Parameter validation and initialization</li>
     *   <li>Target gene identification (specified by index or all genes if not specified)</li>
     *   <li>Optional column inversion for target-as-predictor analysis</li>
     *   <li>Network reconstruction through feature selection for each target gene</li>
     *   <li>Storage and optional saving of the resulting network</li>
     * </ol>
     * </p>
     * 
     * <p>
     * The inferred network is stored in the {@link #recoverednetwork} object and
     * can be saved to a file if {@link #isToSaveFinalData} is true. The saved network
     * is represented as an adjacency matrix where each entry [i,j] indicates whether
     * gene i regulates gene j.
     * </p>
     * 
     * @throws FSException If parameter validation fails or network inference encounters errors
     * @see agn.AGNRoutines#RecoverNetworkfromTemporalExpression
     * @see agn.AGNRoutines#createAdjacencyMatrix(AGN)
     * @see fs.Preprocessing#InvertColumns(float[][])
     */
    private void networkInferenceActionPerformed() {//GEN-FIRST:event_jButton5ActionPerformed
        // float threshold_entropy = ((Double) jS_ThresholdEntropy.getValue()).floatValue();
        // String type_entropy = (String) jCB_PenalizationSE.getSelectedItem();
        // float alpha = ((Double) jS_AlphaSE.getValue()).floatValue();
        // float q_entropy = ((Double) jS_QEntropySE.getValue()).floatValue();
        // float beta = ((float) jSliderBetaSE.getValue() / 100);
        // int search_alg = 0;
        // //jTA_SelectedFeaturesSE.setText("");
        // String path = null;//IOFile.SaveFile();
        // StringBuffer txt = null;
        // int maxf = (Integer) jS_MaxSetSizeSE.getValue();
        float threshold_entropy = (float) threshold;
        String type_entropy = penalizationMethodFeatureSelection == 0 ? "no_obs" : "poor_obs";
        float alpha = (float) alphaFeatureSelection;
        float q_entropy = (float) qEntropyFeatureSelection;
        float beta = betaFeatureSelection / 100;
        int search_alg = 0;
        String path = null;//IOFile.SaveFile();
        StringBuffer txt = null;
        int maxf = maximumSetSizeFeatureSelection;

        if (q_entropy < 0 || alpha < 0) {
            //entrada de dados invalida.
            IOFile.PrintlnAndLog("Error on parameter value: The values of q-entropy and Alpha must be positives.", IOFile.VERBOSE_ERROR);
            return;
        }
        // if (jRB_SFSSE.isSelected()) {
        //     search_alg = 1;
        // } //SFS
        // else if (jRB_ESSE.isSelected()) {
        //     search_alg = 2;
        // }//Exhaustive
        // else if (jRB_SFFSSE.isSelected()) {
        //     search_alg = 3;
        // }//SFFS
        search_alg = searchMethodFeatureSelection;

        Vector targets = null;
        // if (!jTF_Target.getText().equalsIgnoreCase("")) {
        if (targetIndexes != null && !targetIndexes.equalsIgnoreCase("")) {
            targets = new Vector();
            // String strtargets = jTF_Target.getText();
            String strtargets = targetIndexes;
            String str = "";
            for (int i = 0; i < strtargets.length(); i++) {
                if (strtargets.charAt(i) == ' ') {
                    if (!str.equalsIgnoreCase("")) {
                        targets.add(str);
                        str = "";
                    }
                } else {
                    str += strtargets.charAt(i);
                }
            }
            //para o ultimo target da string
            if (!str.equalsIgnoreCase("")) {
                targets.add(str);
                str = "";
            }
        }

        //inverte os tempos de expressao, de forma que os targets passem
        //a ser os preditores, i.e., os preditores passem a considerar o valor
        //do target no instante de tempo posterior.
        //IOFile.PrintMatrix(Md);
        // if (jCB_TargetsAsPredictors.isSelected()) {
        if (isTargetsAsPredictors) {
            Md = Preprocessing.InvertColumns(Md);
            //IOFile.PrintMatrix(Md);
        }

        //vetor com os resultados da selecao de caracteristica.
        //aqui so sera analisada a primeira resposta.
        int resultsetsize = 1;//(Integer)jS_MaxResultListSE.getValue();
        int n = Main.maximumValue(Md, 0, Md.length - 1, 0, Md[0].length - 1) + 1;
        recoverednetwork = new AGN(Md.length, Md[0].length, n);
        recoverednetwork.setTemporalsignal(Mo);
        recoverednetwork.setTemporalsignalquantized(Md);
        if (featurestitles != null){
            recoverednetwork.setLabelstemporalsignal(featurestitles);
        }
        if (datatitles != null){
            AGNRoutines.setGeneNames(recoverednetwork, datatitles);
        }
        //define o parametro do tipo de dados:
        //1 == temporal
        //2 == steady state
        int datatype = 2;
        // if (jRBTimeSeries.isSelected()){
        if (isTimeSeriesData) {
            datatype = 1;
        }
        timer.start("execute_feature_selection");
        txt = AGNRoutines.RecoverNetworkfromTemporalExpression(
                recoverednetwork,
                null,
                datatype,//datatype: 1==temporal, 2==steady-state.
                // jCB_Periodic.isSelected(),
                isItPeriodic,
                threshold_entropy,
                type_entropy,
                alpha,
                beta,
                q_entropy,
                targets,//target indexes
                maxf,
                search_alg,//1==SFS, 2==Exhaustive, 3==SFFS
                // jCB_TargetsAsPredictors.isSelected(),
                isTargetsAsPredictors,
                resultsetsize,
                null,
                threadDistribution,
                numberOfThreads);


        // AGNRoutines.ViewAGN(recoverednetwork);
        if (isToSaveFinalData) {
            IOFile.WriteMatrix(outputFolder + "/final_data/" + prefix + "-final_data.txt", AGNRoutines.createAdjacencyMatrix(recoverednetwork), "\t");
        }
        // AGNRoutines.ViewAGNCLI(recoverednetwork);
        // jTA_SelectedFeaturesSE.setText(txt.toString());
    }//GEN-LAST:event_jButton5ActionPerformed

}
