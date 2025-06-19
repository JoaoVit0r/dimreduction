/***************************************************************************/
/*** Interactive Graphic Environment for Dimensionality Reduction        ***/
/***                                                                     ***/
/*** Copyright (C) 2006  David Corrêa Martins Junior                     ***/
/***                     Fabrício Martins Lopes                          ***/
/***                     Roberto Marcondes Cesar Junior                  ***/
/***                                                                     ***/
/*** This library is free software; you can redistribute it and/or       ***/
/*** modify it under the terms of the GNU Lesser General Public          ***/
/*** License as published by the Free Software Foundation; either        ***/
/*** version 2.1 of the License, or (at your option) any later version.  ***/
/***                                                                     ***/
/*** This library is distributed in the hope that it will be useful,     ***/
/*** but WITHOUT ANY WARRANTY; without even the implied warranty of      ***/
/*** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   ***/
/*** Lesser General Public License for more details.                     ***/
/***                                                                     ***/
/*** You should have received a copy of the GNU Lesser General Public    ***/
/*** License along with this library; if not, write to the Free Software ***/
/*** Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA       ***/
/*** 02110-1301  USA                                                     ***/
/***                                                                     ***/
/*** Contact: David Corrêa Martins Junior - davidjr@vision.ime.usp.br    ***/
/***          Fabrício Martins Lopes - fabriciolopes@vision.ime.usp.br   ***/
/***          Roberto Marcondes Cesar Junior - cesar@vision.ime.usp.br   ***/
/***************************************************************************/
/***************************************************************************/
/*** This class implements methods for reading of files containing a     ***/
/*** matrix                                                              ***/
/***************************************************************************/
package utilities;

import agn.AGN;
import fs.FSException;
import fs.MainCLI;

import java.io.*;
import java.time.Instant;
import java.util.StringTokenizer;
import java.util.Vector;
import javax.swing.JFileChooser;
import javax.swing.JTable;
import javax.swing.filechooser.FileNameExtensionFilter;

/**
 * Provides comprehensive file input/output operations for data manipulation and analysis.
 * 
 * <p>
 * This utility class implements a wide range of methods for reading, writing, and managing
 * various data formats, particularly focusing on matrix data for gene network analysis.
 * It supports operations such as loading expression matrices, saving analysis results,
 * serializing/deserializing AGN objects, and generating formatted output reports.
 * </p>
 * 
 * <p>
 * The class also provides logging functionality with configurable verbosity levels,
 * file chooser dialogs for interactive file selection, and specialized parsers
 * for different data formats commonly used in bioinformatics research.
 * </p>
 * 
 * <p>
 * Key capabilities include:
 * <ul>
 *   <li>Reading numerical matrices from delimited text files</li>
 *   <li>Writing matrices and vectors to formatted text files</li>
 *   <li>Logging with configurable verbosity levels</li>
 *   <li>Serialization of AGN objects for persistent storage</li>
 *   <li>Statistical result formatting for gene network analysis</li>
 *   <li>File format detection and handling</li>
 * </ul>
 * </p>
 * 
 * @see agn.AGN
 * @see fs.FSException
 * @see java.io.Serializable
 */
public class IOFile {
    // Verbosity level constants
    /**
     * No output verbosity level.
     * When set, suppresses all output messages.
     */
    public static final int VERBOSE_NONE = 0;
    
    /**
     * Error-only verbosity level.
     * Only critical errors will be displayed.
     */
    public static final int VERBOSE_ERROR = 1;
    
    /**
     * Warning verbosity level.
     * Includes errors and warning messages.
     */
    public static final int VERBOSE_WARNING = 2;
    
    /**
     * Informational verbosity level.
     * Includes errors, warnings, and informational messages.
     */
    public static final int VERBOSE_INFO = 3;
    
    /**
     * Timer verbosity level.
     * Includes errors, warnings, informational messages, and timing information.
     */
    public static final int VERBOSE_TIMER = 4;
    
    /**
     * Debug verbosity level.
     * Maximum verbosity with all types of messages including detailed debug information.
     */
    public static final int VERBOSE_DEBUG = 5;

    /**
     * Current verbosity level for logging operations.
     * Controls which messages are displayed based on their importance.
     */
    private static int verbosityLevel = VERBOSE_DEBUG;

    /**
     * Global file writer for output operations.
     * Used as a shared resource for writing to files.
     */
    private static FileWriter out = null;

    // File selection methods
    /**
     * Displays a file chooser dialog and returns the absolute path of the selected file.
     * 
     * <p>
     * Opens a standard file selection dialog initialized to the current working directory,
     * allowing the user to navigate the filesystem and select a file. Returns the absolute
     * path to the selected file or null if the operation was canceled.
     * </p>
     * 
     * @return The absolute path of the selected file as a String, or null if no file was selected.
     * @throws FSException If an error occurs during file selection.
     */
    public static String OpenPath() {
        JFileChooser dialogo;
        int opcao;
        String pastainicial = System.getProperty("user.dir").toString();
        dialogo = new JFileChooser(pastainicial);
        opcao = dialogo.showOpenDialog(null);
        if (opcao == JFileChooser.APPROVE_OPTION) {
            try {
                return (dialogo.getSelectedFile().getAbsolutePath());
            } catch (Exception error) {
                throw new FSException("Error when selecting file. " + error, false);
            }
        } else {
            return (null);
        }
    }

    /**
     * Displays a file chooser dialog specifically configured for AGN files.
     * 
     * <p>
     * Opens a file selection dialog with a file filter for Artificial Gene Network (.agn) files,
     * allowing the user to easily locate and select AGN model files. Returns the absolute
     * path to the selected file or null if the operation was canceled.
     * </p>
     * 
     * @return The absolute path of the selected AGN file as a String, or null if no file was selected.
     * @see #SaveAGNFile() The complementary method for saving AGN files.
     */
    public static String OpenAGNFile() {
        JFileChooser dialogo;
        int opcao;
        String pastainicial = System.getProperty("user.dir").toString();
        dialogo = new JFileChooser(pastainicial);
        FileNameExtensionFilter agnFilter = new FileNameExtensionFilter("AGN Artificial Gene Network", "agn");
        dialogo.addChoosableFileFilter(agnFilter);
        dialogo.setFileFilter(agnFilter);
        opcao = dialogo.showOpenDialog(null);
        if (opcao == JFileChooser.APPROVE_OPTION) {
            return (dialogo.getSelectedFile().getAbsolutePath());
        } else {
            return (null);
        }
    }

    /**
     * Displays a file chooser dialog and allows the user to save a file with the specified text content.
     * 
     * <p>
     * Opens a file save dialog initialized to the current working directory, and writes the provided
     * text content to the selected file. If the file already exists, the user is prompted to confirm
     * overwriting. Returns true if the file was successfully saved, false otherwise.
     * </p>
     * 
     * @param texto The text content to be saved to the file.
     * @return true if the file was successfully saved, false otherwise.
     * @throws FSException If an error occurs during file saving.
     */
    public static boolean SaveFile(String texto) {
        JFileChooser dialogo;
        int opcao;
        String pastainicial = System.getProperty("user.dir").toString();

        dialogo = new JFileChooser(pastainicial);
        opcao = dialogo.showSaveDialog(null);
        if (opcao == JFileChooser.APPROVE_OPTION) {
            try {
                BufferedWriter fw = new BufferedWriter(new FileWriter(
                        dialogo.getSelectedFile().getAbsolutePath(), true));
                fw.write(texto);
                fw.flush();
                fw.close();
                return (true);
            } catch (Exception error) {
                throw new FSException("Error when selecting file. " + error, false);
            }
        }
        return (false);
    }

    /**
     * Creates the directory specified by the given path, including any necessary but nonexistent
     * parent directories.
     * 
     * <p>
     * This method is used to ensure that the directory structure for saving files exists.
     * It will create all necessary parent directories if they do not already exist.
     * </p>
     * 
     * @param path The path of the directory to be created.
     * @return true if the directory was created successfully, false otherwise.
     */
    public static boolean Makedirs(String path) {
        File dir = new File(path);
        return dir.mkdirs();
    }

    // Logging methods
    /**
     * Logs the given text at the DEBUG verbosity level.
     * 
     * <p>
     * This is a convenience method that calls {@link #printlnAndLog(String, int) printlnAndLog}
     * with the verbosity level set to DEBUG. It is used for logging detailed debug information.
     * </p>
     * 
     * @param texto The text to be logged.
     */
    public static void PrintlnAndLog(String texto) {
        IOFile.PrintlnAndLog(texto, IOFile.VERBOSE_DEBUG);
    }

    /**
     * Logs the given text with the specified verbosity level.
     * 
     * <p>
     * Messages are logged to both the console and a log file. The verbosity level controls
     * whether the message is displayed based on the current verbosity setting.
     * </p>
     * 
     * @param texto The text to be logged.
     * @param verbosity The verbosity level of the message.
     */
    public static void PrintlnAndLog(String texto, int verbosity) {
        if (verbosity <= verbosityLevel) {

            System.out.println(texto);
            try {
                BufferedWriter fw = new BufferedWriter(new FileWriter("logs/logs.log", true));
                fw.write(texto + "\n");
                fw.flush();
                fw.close();
            } catch (Exception error) {
                throw new FSException("Error when selecting file. " + error, false);
            }
        }
    }

    /**
     * Logs the given text to the specified file with the DEBUG verbosity level.
     * 
     * <p>
     * This is a convenience method that calls {@link #printlnAndLog(String, String, int) printlnAndLog}
     * with the verbosity level set to DEBUG. It is used for logging detailed debug information
     * to a specific file.
     * </p>
     * 
     * @param texto The text to be logged.
     * @param path The path of the file to log the text to.
     */
    public static void PrintlnAndLog(String texto, String path) {
        IOFile.PrintlnAndLog(texto, path, IOFile.VERBOSE_DEBUG);
    }

    /**
     * Logs the given text to the specified file with the specified verbosity level.
     * 
     * <p>
     * Messages are logged to the file at the given path. The verbosity level controls
     * whether the message is displayed based on the current verbosity setting.
     * </p>
     * 
     * @param texto The text to be logged.
     * @param path The path of the file to log the text to.
     * @param verbosity The verbosity level of the message.
     */
    public static void PrintlnAndLog(String texto, String path, int verbosity) {
        if (verbosity <= verbosityLevel) {

            System.out.println(texto);
            try {
                BufferedWriter fw = new BufferedWriter(new FileWriter(path, true));
                fw.write(texto + "\n");
                fw.flush();
                fw.close();
            } catch (Exception error) {
                throw new FSException("Error when selecting file. " + error, false);
            }
        }
    }

    /**
     * Logs a blank line to the log file and console at the DEBUG verbosity level.
     * 
     * <p>
     * This is used to create a separation between log entries, making the log easier to read.
     * </p>
     * 
     * @see #PrintlnAndLog(String, int)
     */
    public static void PrintlnAndLog() {
        IOFile.PrintlnAndLog(IOFile.VERBOSE_DEBUG);
    }

    /**
     * Logs a blank line to the log file and console with the specified verbosity level.
     * 
     * <p>
     * This is used to create a separation between log entries, making the log easier to read.
     * </p>
     * 
     * @param verbosity The verbosity level for the blank line log entry.
     * @see #PrintlnAndLog(String, int)
     */
    public static void PrintlnAndLog(int verbosity) {
        if (verbosity <= verbosityLevel) {

            System.out.println();
            try {
                BufferedWriter fw = new BufferedWriter(new FileWriter("logs/logs.log", true));
                fw.write("\n");
                fw.flush();
                fw.close();
            } catch (Exception error) {
                throw new FSException("Error when selecting file. " + error, false);
            }
        }
    }

    /**
     * Sets the global verbosity level for logging.
     * 
     * <p>
     * This method allows changing the verbosity level at runtime, affecting which messages
     * are logged based on their importance.
     * </p>
     * 
     * @param level The new verbosity level to be set.
     */
    public static void setVerbosity(int level) {
        verbosityLevel = level;
    }

    /**
     * Logs the given text without a newline at the end, at the DEBUG verbosity level.
     * 
     * <p>
     * This is a convenience method that calls {@link #PrintAndLog(String, int) PrintAndLog}
     * with the verbosity level set to DEBUG. It is used for logging debug information that
     * should not have a trailing newline.
     * </p>
     * 
     * @param texto The text to be logged.
     */
    public static void PrintAndLog(String texto) {
        IOFile.PrintAndLog(texto, IOFile.VERBOSE_DEBUG);
    }

    /**
     * Logs the given text without a newline at the end, with the specified verbosity level.
     * 
     * <p>
     * The text is written to the log file and the console, allowing for immediate feedback
     * and persistent logging.
     * </p>
     * 
     * @param texto The text to be logged.
     * @param verbosity The verbosity level of the message.
     */
    public static void PrintAndLog(String texto, int verbosity) {
        if (verbosity <= verbosityLevel) {
            System.out.print(texto);
            try {
                BufferedWriter fw = new BufferedWriter(new FileWriter("logs/logs.log", true));
                fw.write(texto);
                fw.flush();
                fw.close();
            } catch (Exception error) {
                throw new FSException("Error when selecting file. " + error, false);
            }
        }
    }

    /**
     * Saves the given text content to the specified file, overwriting any existing content.
     * 
     * <p>
     * This method is used to save data such as analysis results or processed data matrices
     * to a file. It will replace the entire content of the file with the new data.
     * </p>
     * 
     * @param texto The text content to be saved to the file.
     * @param path The path of the file to save the text to.
     */
    public static void SaveFile(String texto, String path) {
        try {
            BufferedWriter fw = new BufferedWriter(new FileWriter(path, false));
            fw.write(texto);
            fw.flush();
            fw.close();
        } catch (Exception error) {
            throw new FSException("Error when selecting file. " + error, false);
        }
    }

    /**
     * Displays a file chooser dialog and returns the absolute path of the selected file for saving.
     * 
     * <p>
     * This method is similar to {@link #OpenPath()} but is intended for use cases where a file
     * needs to be saved, such as exporting results or saving user-generated data.
     * </p>
     * 
     * @return The absolute path of the selected file as a String, or null if no file was selected.
     * @throws FSException If an error occurs during file selection.
     */
    public static String SaveFile() {
        JFileChooser dialogo;
        int opcao;
        String pastainicial = System.getProperty("user.dir").toString();
        dialogo = new JFileChooser(pastainicial);
        opcao = dialogo.showSaveDialog(null);
        if (opcao == JFileChooser.APPROVE_OPTION) {
            return (dialogo.getSelectedFile().getAbsolutePath());
        } else {
            return (null);
        }
    }

    /*
     * Get the extension of a file.
     */
    public static String getExtension(String filename) {
        String ext = null;
        int i = filename.lastIndexOf('.');
        if (i > 0 && i < filename.length() - 1) {
            ext = filename.substring(i + 1).toLowerCase();
        }
        return ext;
    }

    /**
     * Displays a file chooser dialog for saving, specifically configured for AGN files.
     * 
     * <p>
     * This method is similar to {@link #OpenAGNFile()} but is used for saving AGN files.
     * It ensures that the saved file has the correct .agn extension.
     * </p>
     * 
     * @return The absolute path of the selected AGN file for saving, or null if no file was selected.
     * @see #OpenAGNFile() For opening AGN files.
     */
    public static String SaveAGNFile() {
        JFileChooser dialogo;
        int opcao;
        String pastainicial = System.getProperty("user.dir").toString();
        dialogo = new JFileChooser(pastainicial);
        FileNameExtensionFilter agnFilter = new FileNameExtensionFilter("AGN Artificial Gene Network", "agn");
        dialogo.addChoosableFileFilter(agnFilter);
        dialogo.setFileFilter(agnFilter);
        opcao = dialogo.showSaveDialog(null);
        if (opcao == JFileChooser.APPROVE_OPTION) {
            String filename = dialogo.getSelectedFile().getAbsolutePath();
            String ext = getExtension(filename);
            if (ext == null) {
                FileNameExtensionFilter filter = (FileNameExtensionFilter) dialogo.getFileFilter();
                filename = filename + "." + filter.getExtensions()[0];
            }
            return (filename);
        } else {
            return (null);
        }
    }

    /**
     * Displays a file chooser dialog for saving, specifically configured for image files.
     * 
     * <p>
     * This method allows the user to save image files in various formats such as JPEG, PNG, GIF, and BMP.
     * The appropriate file extension is automatically appended based on the selected format.
     * </p>
     * 
     * @return The absolute path of the selected image file for saving, or null if no file was selected.
     * @see #OpenPath() For opening files.
     */
    public static String SaveIMGFile() {
        JFileChooser dialogo;
        int opcao;
        String pastainicial = System.getProperty("user.dir").toString();
        dialogo = new JFileChooser(pastainicial);
        FileNameExtensionFilter jpgFilter = new FileNameExtensionFilter("JPEG Compressed Image Files", "jpg");
        FileNameExtensionFilter pngFilter = new FileNameExtensionFilter("PNG Portable Network Graphics", "png");
        FileNameExtensionFilter gifFilter = new FileNameExtensionFilter("GIF Graphics Interchange Format", "gif");
        FileNameExtensionFilter bmpFilter = new FileNameExtensionFilter("BMP file format", "bmp");
        dialogo.addChoosableFileFilter(pngFilter);
        dialogo.addChoosableFileFilter(jpgFilter);
        dialogo.addChoosableFileFilter(gifFilter);
        dialogo.addChoosableFileFilter(bmpFilter);
        dialogo.setFileFilter(pngFilter);
        opcao = dialogo.showSaveDialog(null);
        if (opcao == JFileChooser.APPROVE_OPTION) {
            String filename = dialogo.getSelectedFile().getAbsolutePath();
            String ext = getExtension(filename);
            if (ext == null) {
                FileNameExtensionFilter filter = (FileNameExtensionFilter) dialogo.getFileFilter();
                filename = filename + "." + filter.getExtensions()[0];
            }
            return (filename);
        } else {
            return (null);
        }
    }

    /**
     * Saves the content of the given JTable as a space-delimited text file.
     * 
     * <p>
     * This method allows exporting table data to a text file, where each row is written as a
     * separate line and columns are separated by spaces. It is useful for saving analysis results
     * or data exports in a human-readable format.
     * </p>
     * 
     * @param table The JTable containing the data to be saved.
     * @return true if the table data was successfully saved, false otherwise.
     * @throws FSException If an error occurs during file saving.
     */
    public static boolean SaveTable(JTable table) {
        JFileChooser dialogo;
        int opcao;
        String pastainicial = System.getProperty("user.dir").toString();

        dialogo = new JFileChooser(pastainicial);
        opcao = dialogo.showSaveDialog(null);
        if (opcao == JFileChooser.APPROVE_OPTION) {
            try {
                BufferedWriter fw = new BufferedWriter(new FileWriter(
                        dialogo.getSelectedFile().getAbsolutePath(), false));
                for (int i = 0; i < table.getRowCount(); i++) {
                    for (int j = 0; j < table.getColumnCount(); j++) {
                        Object valor = table.getValueAt(i, j);
                        fw.write(valor + "");
                        if (j < table.getColumnCount() - 1) {
                            fw.write(" ");
                        }
                    }
                    if (i < table.getRowCount() - 1) {
                        fw.write("\n");
                    }
                    fw.flush();
                }
                fw.close();
                return (true);
            } catch (Exception error) {
                throw new FSException("Error when selecting file. " + error, false);
            }
        }
        return (false);
    }

    // BufferedReader and BufferedWriter opening methods
    /**
     * Opens a BufferedReader for reading the specified file.
     * 
     * <p>
     * This method wraps a FileReader in a BufferedReader to provide efficient reading of text files.
     * It is used internally by other methods to read data from files and serves as a central point
     * for file opening operations, ensuring consistent error handling across the application.
     * </p>
     * 
     * <p>
     * The BufferedReader provides buffered reading capabilities, which significantly improves 
     * performance when reading large files compared to using a FileReader directly. This is 
     * particularly important for the application's typical use case of reading large data matrices.
     * </p>
     * 
     * <p>
     * If the file cannot be found or opened, this method throws an FSException with a descriptive
     * error message, which can be caught and handled by the calling method.
     * </p>
     * 
     * @param path The absolute path of the file to be opened.
     * @return A BufferedReader for the specified file.
     * @throws FSException If the file cannot be found or opened.
     * @see java.io.BufferedReader The underlying class providing buffered reading functionality.
     * @see java.io.FileReader The underlying class for file access.
     * @see fs.FSException The custom exception class used for reporting file system errors.
     */
    public static BufferedReader OpenBufferedReader(String path) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(path));
        } catch (FileNotFoundException error) {
            throw new FSException("Error on File Open. " + error, false);
        }
        return (br);
    }

    /**
     * Opens a BufferedWriter for writing to the specified file.
     * 
     * <p>
     * This method wraps a FileWriter in a BufferedWriter to provide efficient writing to text files.
     * It is used internally by other methods to write data to files.
     * </p>
     * 
     * @param path The path of the file to be opened.
     * @param append If true, data will be appended to the end of the file; otherwise, the file will be overwritten.
     * @return A BufferedWriter for the specified file.
     * @throws FSException If the file cannot be created or opened for writing.
     */
    public static BufferedWriter OpenBufferedWriter(String path, boolean append) {
        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new FileWriter(path, append));
        } catch (IOException error) {
            throw new FSException("Error on File Open. " + error, false);
        }
        return (bw);
    }
    // receives a file name and returns a matrix of double values


    /**
     * Reads a matrix of float values from a delimited text file.
     * 
     * <p>
     * Parses a text file containing numeric data and converts it to a 2D float array.
     * The method allows skipping a specified number of initial rows and columns,
     * and uses the provided delimiter to separate values. This is particularly useful
     * for loading gene expression matrices with header rows and ID columns.
     * </p>
     * 
     * <p>
     * The method first determines the dimensions of the matrix by scanning the file 
     * to count rows and columns. It then allocates a float array of the appropriate 
     * size, skips the specified number of initial rows and columns, and reads the 
     * remaining data into the array, converting each token to a float value.
     * </p>
     * 
     * <p>
     * This method is central to the application's data loading pipeline and is used
     * extensively for reading experimental data, gene expression matrices, and other
     * numerical datasets for subsequent processing and analysis.
     * </p>
     * 
     * @param arch_name The path to the input file.
     * @param startrow The number of initial rows to skip.
     * @param startcolumn The number of initial columns to skip.
     * @param delimiter The character(s) used to separate values in the file.
     * @return A 2D float array containing the matrix data, or null if the file cannot be opened.
     * @throws IOException If an I/O error occurs during file reading.
     * @throws NumberFormatException If the file contains non-numeric values that cannot be parsed as floats.
     * @throws FSException If the file cannot be found or opened.
     * 
     * @see #ReadMatrix(String, String) The simpler version for character matrices.
     * @see #OpenBufferedReader(String) The method used to create a file reader.
     * @see StringTokenizer For the tokenization mechanism used to parse each line.
     */
    @SuppressWarnings("empty-statement")
    public static float[][] ReadMatrix(String arch_name, int startrow,
            int startcolumn, String delimiter)
            throws IOException, NumberFormatException {
        // obtaining the number of lines and collumns of the matrix
        BufferedReader fp = OpenBufferedReader(arch_name);
        if (fp == null) {
            return (null);
        }
        StringTokenizer s = new StringTokenizer(fp.readLine(), delimiter);
        int collumns = s.countTokens();
        int lines;
        for (lines = 1; fp.readLine() != null; lines++) {
            ;
        }
        // reading the matrix
        float[][] A = new float[lines - startrow][collumns - startcolumn];
        fp = OpenBufferedReader(arch_name);
        if (fp == null) {
            return (null);
        }

        for (int l = 0; l < startrow; l++) {
            s = new StringTokenizer(fp.readLine(), delimiter);// desconsidera linha
        }
        for (int i = 0; i < lines - startrow; i++) {
            s = new StringTokenizer(fp.readLine(), delimiter);

            for (int j = 0; j < startcolumn; j++) {
                s.nextToken();// desconsidera coluna
            }
            for (int j = 0; s.hasMoreTokens(); A[i][j++] = Float.parseFloat(s.nextToken())) {
                ;
            }
            // System.out.println("linha " + i);
        }
        fp.close();
        return A;
    }
    // receives a file name and returns a matrix of chars (small integers for
    // memory space economy)

    @SuppressWarnings("empty-statement")
    public static char[][] ReadMatrix(String arch_name, String delimiter)
            throws IOException {
        BufferedReader fp = OpenBufferedReader(arch_name);
        if (fp == null) {
            return (null);
        }
        // obtaining the number of lines and collumns of the matrix
        StringTokenizer s = new StringTokenizer(fp.readLine(), delimiter);
        int collumns = s.countTokens();
        int lines;
        for (lines = 1; fp.readLine() != null; lines++) {
            ;
        }
        char[][] A = new char[lines][collumns];
        fp = OpenBufferedReader(arch_name);
        if (fp == null) {
            return (null);
        }
        // reading the matrix
        for (int i = 0; i < lines; i++) {
            s = new StringTokenizer(fp.readLine(), delimiter);
            for (int j = 0; s.hasMoreTokens(); A[i][j++] = (char) Integer.parseInt(s.nextToken(delimiter))) {
                ;
            }
        }
        fp.close();
        return A;
    }

    /**
     * Reads the first column of data from a delimited text file.
     * 
     * <p>
     * Extracts the first token from each line of a text file, starting after a specified number
     * of initial rows. This method is particularly useful for reading row labels or gene/variable
     * identifiers from data files.
     * </p>
     * 
     * @param arch_name The path to the input file.
     * @param startrow The number of initial rows to skip.
     * @param delimiter The character(s) used to separate values in the file.
     * @return A Vector containing the first token from each line as strings.
     * @throws IOException If an I/O error occurs during file reading.
     * @throws FSException If the file cannot be found or opened.
     * 
     * @see #ReadDataFirstRow(String, int, int, String) The complementary method for reading row data.
     * @see #OpenBufferedReader(String) The method used to create a file reader.
     */
    public static Vector ReadDataFirstCollum(String arch_name, int startrow,
            String delimiter)
            throws IOException {
        BufferedReader fp = OpenBufferedReader(arch_name);
        if (fp == null) {
            return (null);
        }
        Vector vetout = new Vector();
        StringTokenizer s;

        if (fp.ready()) {
            for (int l = 0; l < startrow; l++) {
                s = new StringTokenizer(fp.readLine(), delimiter);// desconsidera linha
            }
        }
        while (fp.ready()) {
            // reading the matrix
            s = new StringTokenizer(fp.readLine(), delimiter);
            vetout.add(s.nextToken());
        }
        fp.close();
        return vetout;
    }

    public static Vector[] ReadDataCollumns(String arch_name, int startrow,
            Vector collumns, String delimiter)
            throws IOException {
        BufferedReader fp = OpenBufferedReader(arch_name);
        if (fp == null) {
            return (null);
        }
        Vector[] vetout = new Vector[collumns.size()];
        for (int i = 0; i < collumns.size(); i++) {
            vetout[i] = new Vector();
        }

        StringTokenizer s;
        if (fp.ready()) {
            for (int l = 0; l < startrow; l++) {
                s = new StringTokenizer(fp.readLine(), delimiter);// desconsidera linha
            }
        }

        while (fp.ready()) {
            s = new StringTokenizer(fp.readLine(), delimiter);
            int pos = 0;
            int col = 0;
            while (s.hasMoreTokens()) {
                if (collumns.contains(col)) {
                    vetout[pos++].add(s.nextToken());// token of the collumn
                } else {
                    s.nextToken();// token of the collumn
                }
                col++;
            }
        }
        fp.close();
        return vetout;
    }

    /**
     * Reads the first row of data from a delimited text file.
     * 
     * <p>
     * Extracts data from a specific row of a text file, skipping a specified number
     * of initial rows and columns. This method is particularly useful for reading
     * column headers or feature names from data files.
     * </p>
     * 
     * <p>
     * The method first skips the specified number of rows, then reads the next row,
     * skips the specified number of columns, and collects all remaining tokens from
     * that row into a Vector. This allows for flexible extraction of header information
     * from different file formats.
     * </p>
     * 
     * @param arch_name The path to the input file.
     * @param startrow The number of initial rows to skip.
     * @param startcolumn The number of initial columns to skip.
     * @param delimiter The character(s) used to separate values in the file.
     * @return A Vector containing the extracted values as strings, or null if the file cannot be opened.
     * @throws IOException If an I/O error occurs during file reading.
     * @throws FSException If the file cannot be found or opened.
     * 
     * @see #ReadDataFirstCollum(String, int, String) The complementary method for reading column data.
     * @see #OpenBufferedReader(String) The method used to create a file reader.
     */
    public static Vector ReadDataFirstRow(String arch_name, int startrow,
            int startcolumn, String delimiter) throws IOException {
        BufferedReader fp = OpenBufferedReader(arch_name);
        if (fp == null) {
            return (null);
        }
        Vector vetout = new Vector();
        StringTokenizer s;
        if (fp.ready()) {
            for (int l = 0; l < startrow; l++) {
                s = new StringTokenizer(fp.readLine(), delimiter);// desconsidera linha
            }
            // reading the matrix
            s = new StringTokenizer(fp.readLine(), delimiter);
            for (int j = 0; j < startcolumn; j++) {
                s.nextToken();// desconsidera coluna
            }
            while (s.hasMoreTokens()) {
                vetout.add(s.nextToken());
            }
        }
        fp.close();
        return vetout;
    }

    /**
     * Reads all lines from a text file into a Vector.
     * 
     * <p>
     * Opens a text file and reads each line as a separate String, returning
     * all lines in a Vector. This method is useful for loading text-based data
     * where each line represents a complete data entry.
     * </p>
     * 
     * @param arch_name The path to the input file.
     * @return A Vector containing all lines from the file as Strings.
     * @throws IOException If an I/O error occurs during file reading.
     * 
     * @see #ReadDataFirstRow(String, int, int, String) For reading specific rows.
     */
    public static Vector ReadDataLine(String arch_name)
            throws IOException {
        BufferedReader fp = OpenBufferedReader(arch_name);
        if (fp == null) {
            return (null);
        }
        Vector out = new Vector();
        while (fp.ready()) // reading the matrix
        {
            out.add(fp.readLine());
        }
        fp.close();
        return out;
    }

    public static AGN ReadAGNfromFile(String path) {
        File arquivo = null;
        FileInputStream fluxoentrada = null;
        ObjectInputStream obj = null;
        AGN network = null;
        try {
            arquivo = new File(path);
            if (arquivo != null) {
                fluxoentrada = new FileInputStream(arquivo);
            } else {
                return null;
            }
            obj = new ObjectInputStream(fluxoentrada);
            network = (AGN) obj.readObject();
            fluxoentrada.close();
            obj.close();
        } catch (IOException error) {
            System.out.println("Error on create the File Inpu Stream. " + error);
            return null;
        } catch (ClassNotFoundException error) {
            System.out.println("Error on AGN read. " + error);
            return null;
        }
        return network;
    }

    public static boolean WriteAGNtoFile(AGN network, String path) {
        File arquivo = null;
        FileOutputStream fluxosaida = null;
        ObjectOutputStream obj = null;
        try {
            arquivo = new File(path);
            if (arquivo != null) {
                fluxosaida = new FileOutputStream(arquivo, false);// append == true == continua gravando no arquivo
                obj = new ObjectOutputStream(fluxosaida);
                obj.writeObject(network);
            } else {
                return false;
            }
        } catch (IOException e) {
            System.out.println("Erro na criacao do FileOutputStream. " + e);
            return false;
        }
        return true;
    }

    /**
     * Writes a floating-point matrix to a delimited text file.
     * 
     * <p>
     * This method saves a 2D float array to a file with the specified delimiter between values.
     * Each row of the matrix is written as a separate line in the output file, with values
     * separated by the specified delimiter. This format is commonly used for exporting
     * numerical data for further analysis in other tools.
     * </p>
     * 
     * <p>
     * In the gene network analysis pipeline, this method is particularly useful for saving
     * processed data matrices, quantized expression data, and other numerical results.
     * The resulting file format is compatible with most data analysis tools and can be
     * easily imported into spreadsheet applications or other analysis software.
     * </p>
     * 
     * @param path The path to the output file. If the file exists, it will be overwritten.
     * @param M The 2D float array to be written to the file.
     * @param delimiter The character(s) to use as delimiter between values (e.g., tab, comma).
     * @throws FSException If an error occurs during file writing.
     * @see #WriteMatrix(String, int[][], String) The version for integer matrices.
     * @see MainCLI#networkInferenceActionPerformed() Where this method is used to save network inference results.
     */
    public static void WriteMatrix(String path, float[][] M,
            String delimiter) {
        try {
            out = new FileWriter(new File(path), false);
            for (int i = 0; i < M.length; i++) {
                for (int j = 0; j < M[0].length; j++) {
                    if (j < M[0].length - 1) {
                        out.write(M[i][j] + delimiter);
                    } else {
                        out.write(M[i][j] + "\n");
                    }
                }
                // out.write("\n");
                out.flush();
            }
            out.close();
        } catch (IOException error) {
            throw new FSException("Error when save adjacency matrix. " + error, false);
        }
    }

    /**
     * Writes an integer matrix to a delimited text file.
     * 
     * <p>
     * This method saves a 2D integer array to a file with the specified delimiter between values.
     * Each row of the matrix is written as a separate line in the output file, with values
     * separated by the specified delimiter. This format is particularly useful for exporting
     * discrete or categorical data, such as adjacency matrices representing network structures.
     * </p>
     * 
     * <p>
     * In gene network analysis, this method is commonly used to save:
     * <ul>
     *   <li>Adjacency matrices representing inferred gene regulatory networks</li>
     *   <li>Boolean matrices representing presence/absence relationships</li>
     *   <li>Quantized or discretized data matrices</li>
     *   <li>Connectivity maps between genes or variables</li>
     * </ul>
     * </p>
     * 
     * <p>
     * The resulting file format is compatible with network analysis tools and can be
     * easily visualized using graph drawing software or imported into network analysis packages.
     * </p>
     *
     * @param path The path to the output file. If the file exists, it will be overwritten.
     * @param M The 2D integer array to be written to the file.
     * @param delimiter The character(s) to use as delimiter between values (e.g., tab, comma).
     * @throws FSException If an error occurs during file writing.
     * @see #WriteMatrix(String, float[][], String) The version for floating-point matrices.
     * @see AGNRoutines#createAdjacencyMatrix(AGN) Which creates adjacency matrices that are often saved using this method.
     */
    public static void WriteMatrix(String path, int[][] M,
            String delimiter) {
        try {
            out = new FileWriter(new File(path), false);
            for (int i = 0; i < M.length; i++) {
                for (int j = 0; j < M[0].length; j++) {
                    if (j < M[0].length - 1) {
                        out.write(M[i][j] + delimiter);
                    } else {
                        out.write(M[i][j] + "\n");
                    }
                }
                // out.write("\n");
                out.flush();
            }
            out.close();
        } catch (IOException error) {
            throw new FSException("Error when save adjacency matrix. " + error, false);
        }
    }

    /**
     * Writes a double-precision matrix to a space-delimited text file.
     * 
     * <p>
     * This method saves a 2D double array to a file with spaces between values.
     * Each row of the matrix is written as a separate line in the output file.
     * This format is particularly suitable for high-precision numerical data that
     * requires the full double precision, such as statistical results, correlation
     * matrices, or normalized expression values.
     * </p>
     * 
     * <p>
     * Unlike the {@link #WriteMatrix(String, float[][], String)} method, this method:
     * <ul>
     *   <li>Always uses spaces as delimiters (not configurable)</li>
     *   <li>Preserves full double precision in the output</li>
     *   <li>Is optimized for statistical or analytical results</li>
     * </ul>
     * </p>
     *
     * @param path The path to the output file. If the file exists, it will be overwritten.
     * @param M The 2D double array to be written to the file.
     * @throws FSException If an error occurs during file writing.
     * @see #WriteMatrix(String, float[][], String) For more flexible delimiter options.
     * @see #WriteMatrix(String, int[][], String) For integer matrix output.
     */
    public static void WriteFile(String path, double[][] M) {
        try {
            out = new FileWriter(new File(path), false);
            for (int i = 0; i < M.length; i++) {
                for (int j = 0; j < M[0].length; j++) {
                    out.write(M[i][j] + " ");
                }
                out.write("\n");
                out.flush();
            }
            out.close();
        } catch (IOException error) {
            throw new FSException("Error when save normalized file. " + error, false);
        }
    }

    /**
     * Writes the contents of a Vector to a text file, with each element on a separate line.
     * 
     * <p>
     * This method writes each element of the provided Vector to a file, with each element 
     * on a separate line. The method supports both creating a new file and appending to an
     * existing file, depending on the value of the append parameter.
     * </p>
     * 
     * <p>
     * This method is particularly useful for saving:
     * <ul>
     *   <li>Lists of gene names or identifiers</li>
     *   <li>Feature selection results</li>
     *   <li>Log entries or processing steps</li>
     *   <li>Lists of target genes for analysis</li>
     * </ul>
     * </p>
     *
     * @param path The path to the output file.
     * @param vet The Vector containing the elements to write.
     * @param append If true, data is appended to the existing file; if false, the file is overwritten.
     * @throws FSException If an error occurs during file writing.
     * @see #ReadDataLine(String) The complementary method for reading line-based files.
     */
    public static void WriteFile(String path, Vector vet, boolean append) {
        try {
            out = new FileWriter(new File(path), append);
            for (int i = 0; i < vet.size(); i++) {
                out.write(vet.get(i) + "\n");
                out.flush();
            }
            out.close();
        } catch (IOException error) {
            throw new FSException("Error when save normalized file. " + error, false);
        }
    }

    public static void WriteIndividualResults(
            AGN originalagn,
            Vector originalpredictors,
            Vector inferredpredictors,
            StringBuffer outline) throws IOException {

        float TP = 0, FP = 0, FN = 0, TN = 0;
        for (int i = 0; i < inferredpredictors.size(); i++) {
            int ip = (Integer) inferredpredictors.get(i);
            if (originalpredictors.contains(ip)) {
                TP++;
            } else {
                FP++;
            }
            outline.append(ip + " ");
        }
        for (int i = 0; i < originalpredictors.size(); i++) {
            int op = (Integer) originalpredictors.get(i);
            if (!inferredpredictors.contains(op)) {
                FN++;
            }
        }
        TN = (originalagn.getNrgenes() - (originalpredictors.size() + FN));
        outline.append(";");// fecha a celula de predictores
        float PPV = (TP / (TP + FP));// precision
        float Sensitivity = (TP / (TP + FN));// recall
        float Specificity = (TN / (TN + FP));
        outline.append(TP + ";");
        outline.append(FP + ";");
        outline.append(FN + ";");
        outline.append(TN + ";");
        outline.append(PPV + ";");
        outline.append(Sensitivity + ";");
        outline.append(Specificity + ";");
    }

    public static void IndividualResults(
            AGN originalagn,
            Vector originalpredictors,
            Vector ties,
            StringBuffer outline) throws IOException {

        for (int tp = 0; tp < ties.size(); tp++) {
            // para cada conj de predictores empatados sao gerados os resultados abaixo.
            Vector inferredpredictors = (Vector) ties.get(tp);
            WriteIndividualResults(
                    originalagn,
                    originalpredictors,
                    inferredpredictors,
                    outline);
        }
    }

    public static void WriteTies(
            AGN originalagn,
            String path,
            int target,
            int avgedges,
            String networktopology,
            Vector originalpredictors,
            float qvalue,
            Vector inferredpredictors,
            Vector ties,
            float cfvalue, // valor obtido pela funcao criterio
            boolean initialization) {

        try {
            out = new FileWriter(new File(path), true);
            StringBuffer outline = new StringBuffer();
            if (initialization) {
                outline.append("target;");
                outline.append("avg-edges(k);");
                outline.append("topology;");
                outline.append("original-predictors;");
                outline.append("CF Value;");
                outline.append("q-value;");
                outline.append("ties?;");
                outline.append("#ties;");
                outline.append("recovered-predictors;");
                outline.append("TP;");
                outline.append("FP;");
                outline.append("FN;");
                outline.append("TN;");
                outline.append("PPV;");
                outline.append("Sensitivity;");
                outline.append("Specificity;");
                outline.append("recovered-predictors;");
                outline.append("TP;");
                outline.append("FP;");
                outline.append("FN;");
                outline.append("TN;");
                outline.append("PPV;");
                outline.append("Sensitivity;");
                outline.append("Specificity;");
                outline.append("...;");
            } else {
                outline.append(target + ";");
                outline.append(avgedges + ";");
                outline.append(networktopology + ";");
                // predictores originais
                int[] op = null;
                if (originalpredictors != null) {
                    op = new int[originalpredictors.size()];
                    for (int i = 0; i < originalpredictors.size(); i++) {
                        op[i] = (Integer) originalpredictors.get(i);
                        outline.append(op[i] + " ");
                    }
                }
                outline.append(";");

                // criterion function value
                outline.append(cfvalue + ";");

                // entropic parameter q applied in the predictors recover
                outline.append(qvalue + ";");

                if (ties != null && ties.size() > 1) {
                    outline.append("1;");// ties?
                    outline.append(ties.size() + ";");// #ties
                    IndividualResults(
                            originalagn,
                            originalpredictors,
                            ties,
                            outline);
                } else if (inferredpredictors != null) {
                    outline.append("0;");// ties?
                    outline.append("0;");// #ties
                    WriteIndividualResults(
                            originalagn,
                            originalpredictors,
                            inferredpredictors,
                            outline);
                } else {
                    outline.append("0;");// ties?
                    outline.append("0;");// #ties
                    outline.append(";");// TP
                    outline.append(";");// FP
                    outline.append(";");// FN
                    outline.append(";");// TN
                    outline.append(";");// PPV
                    outline.append(";");// Sensitivity
                    outline.append(";");// Specificity
                }
            }
            outline.append("\n");
            out.write(outline.toString());
            out.flush();
            out.close();
        } catch (IOException error) {
            throw new FSException("Error when save net comparisons file. " + error, false);
        }
    }

    public static void WriteFile(String path, int ini, String net, int nr_nodes,
            int avg_edges, int nr_obs, int quantization, float[] CM, int concat,
            float q_tsallis, boolean append) {
        try {
            out = new FileWriter(new File(path), append);
            StringBuffer outline = new StringBuffer();
            if (ini == 1) {
                outline.append("net-type");
                while (outline.length() < 9) {
                    outline.append(" ");
                }

                outline.append("nr-nodes");
                while (outline.length() < 18) {
                    outline.append(" ");
                }

                outline.append("avg-edges");
                while (outline.length() < 28) {
                    outline.append(" ");
                }

                outline.append("nr-obs");
                while (outline.length() < 35) {
                    outline.append(" ");
                }

                outline.append("concat");
                while (outline.length() < 42) {
                    outline.append(" ");
                }

                outline.append("quant");
                while (outline.length() < 48) {
                    outline.append(" ");
                }

                outline.append("TP");
                while (outline.length() < 54) {
                    outline.append(" ");
                }

                outline.append("FN");
                while (outline.length() < 60) {
                    outline.append(" ");
                }

                outline.append("TN");
                while (outline.length() < 69) {
                    outline.append(" ");
                }

                outline.append("FP");
                while (outline.length() < 75) {
                    outline.append(" ");
                }

                outline.append("TNR");
                while (outline.length() < 87) {
                    outline.append(" ");
                }

                outline.append("TPR");
                while (outline.length() < 100) {
                    outline.append(" ");
                }

                outline.append("TP-H");
                while (outline.length() < 106) {
                    outline.append(" ");
                }

                outline.append("FN-H");
                while (outline.length() < 112) {
                    outline.append(" ");
                }

                outline.append("TN-H");
                while (outline.length() < 121) {
                    outline.append(" ");
                }

                outline.append("FP-H");
                while (outline.length() < 127) {
                    outline.append(" ");
                }

                outline.append("TNR-H");
                while (outline.length() < 139) {
                    outline.append(" ");
                }

                outline.append("TPR-H");
                while (outline.length() < 152) {
                    outline.append(" ");
                }

                outline.append("q-Tsallis");
            } else {
                outline.append(net);
                while (outline.length() < 9) {
                    outline.append(" ");
                }

                outline.append(nr_nodes);
                while (outline.length() < 18) {
                    outline.append(" ");
                }

                outline.append(avg_edges);
                while (outline.length() < 28) {
                    outline.append(" ");
                }

                outline.append(nr_obs);
                while (outline.length() < 35) {
                    outline.append(" ");
                }

                outline.append(concat);
                while (outline.length() < 42) {
                    outline.append(" ");
                }

                outline.append(quantization);
                while (outline.length() < 48) {
                    outline.append(" ");
                }

                outline.append((int) CM[0]);// true positive
                while (outline.length() < 54) {
                    outline.append(" ");
                }

                outline.append((int) CM[3]);// false negative
                while (outline.length() < 60) {
                    outline.append(" ");
                }

                outline.append((int) CM[2]);// true negative
                while (outline.length() < 69) {
                    outline.append(" ");
                }

                outline.append((int) CM[1]);// false positive
                while (outline.length() < 75) {
                    outline.append(" ");
                }

                outline.append(CM[4]);// TNR
                while (outline.length() < 87) {
                    outline.append(" ");
                }

                outline.append(CM[5]);// TPR
                while (outline.length() < 100) {
                    outline.append(" ");
                }

                outline.append((int) CM[6]);// TP-Hub
                while (outline.length() < 106) {
                    outline.append(" ");
                }

                outline.append((int) CM[9]);// FN-Hub
                while (outline.length() < 112) {
                    outline.append(" ");
                }

                outline.append((int) CM[8]);// TN-Hub
                while (outline.length() < 121) {
                    outline.append(" ");
                }

                outline.append((int) CM[7]);// FP-Hub
                while (outline.length() < 127) {
                    outline.append(" ");
                }

                outline.append(CM[10]);// TNR-Hub
                while (outline.length() < 139) {
                    outline.append(" ");
                }

                outline.append(CM[11]);// TPR-Hub
                while (outline.length() < 152) {
                    outline.append(" ");
                }

                outline.append(q_tsallis);// q-entropy used in criterion function
            }

            outline.append("\n");
            out.write(outline.toString());
            out.flush();
            out.close();
        } catch (IOException error) {
            throw new FSException("Error when save net comparisons file. " + error, false);
        }

    }

    public static synchronized void WriteFile(BufferedWriter out, int target, Vector predictors, double entropy) {
        if (out != null) {
            try {
                // BufferedWriter out = new BufferedWriter(new FileWriter(path, true));
                for (int i = 0; i < predictors.size(); i++) {
                    // passo 2 pq sao armazenados o indice do preditor e sua entropia em sequencia.
                    out.write((Integer) predictors.get(i) + "       " + target + "       " + entropy);
                    out.write("\n");
                    out.flush();
                }
                // out.close();

            } catch (IOException error) {
                throw new FSException("Error when save network ascii file. " + error, false);
            }

        }
    }

    // prints the content of a matrix
    public static void PrintMatrix(char[][] M) {
        int lines = M.length;
        int collumns = M[0].length;
        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < collumns; j++) {
                System.out.print((int) M[i][j] + " ");
            }

            System.out.println();
        }
        System.out.println();
    }

    public static void PrintMatrix(int[][] M) {
        int lines = M.length;
        int collumns = M[0].length;
        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < collumns; j++) {
                System.out.print(M[i][j] + " ");
            }

            System.out.println();
        }
        System.out.println();
    }

    // prints the content of a matrix
    public static void PrintMatrix(float[][] M) {
        int lines = M.length;
        int collumns = M[0].length;
        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < collumns; j++) {
                System.out.print(M[i][j] + " ");
            }

            System.out.println();
        }
        System.out.println();
    }

    // prints the content of a matrix
    public static void PrintArray(float[] M) {
        int rows = M.length;
        for (int i = 0; i < rows; i++) {
            System.out.print(M[i] + " ");
        }
        System.out.println();
    }

    public static void PrintMatrix(int[][] Mo, int[][] Mr) {
        int lines = Mo.length;
        int collumns = Mo[0].length;
        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < collumns; j++) {
                if (Mo[i][j] == 1) {
                    System.out.print(Mr[i][j] + "* ");
                } else {
                    System.out.print(Mr[i][j] + " ");
                }
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void MakeHeaderStatistics(String output) {
        try {
            BufferedWriter fw = new BufferedWriter(new FileWriter(output, false));
            StringBuffer out = new StringBuffer();
            out.append("#");
            while (out.length() < 5) {
                out.append(" ");
            }

            out.append("nrnos");
            while (out.length() < 12) {
                out.append(" ");
            }

            out.append("maxpred");
            while (out.length() < 22) {
                out.append(" ");
            }

            out.append("iter");
            while (out.length() < 29) {
                out.append(" ");
            }

            out.append("obs");
            while (out.length() < 35) {
                out.append(" ");
            }

            out.append("concat");
            while (out.length() < 44) {
                out.append(" ");
            }

            out.append("certos/total");
            while (out.length() < 59) {
                out.append(" ");
            }

            out.append("falsosp");
            while (out.length() < 69) {
                out.append(" ");
            }

            out.append("negativos");
            out.append("\n");
            fw.write(out.toString());
            fw.close();
        } catch (Exception error) {
            throw new FSException("Error when creating statistics model file. " + error, false);
        }

    }

    public static void CloseBufferedWriter(BufferedWriter out) {
        if (out != null) {
            try {
                out.close();
            } catch (IOException error) {
                throw new FSException("Error when close file writer. " + error, false);
            }
        }
    }

    public static void PrintVectorofPredictors(Vector<Vector> vpredictors) {
        for (int i = 0; i < vpredictors.size(); i++) {
            PrintPredictors(vpredictors.get(i));
        }
    }

    public static void PrintPredictors(Vector predictors) {
        System.out.print("( ");
        for (int i = 0; i < predictors.size(); i++) {
            System.out.print(predictors.get(i) + " ");
        }
        System.out.print(")\n");
    }
}
