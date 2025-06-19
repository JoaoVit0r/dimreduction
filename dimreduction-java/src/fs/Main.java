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
/*** This class implements the Main function of the software.            ***/
/***************************************************************************/
package fs;

import agn.AGNRoutines;
import agn.AGN;
import java.io.IOException;
import java.util.Vector;
import utilities.IOFile;

/**
 * Main entry point for the GUI-based feature selection and network inference tool.
 * <p>
 * This class provides the entry point for the GUI application and contains utility
 * methods used by other classes in the project, such as finding maximum values in 
 * data matrices, processing matrix data, and other common operations.
 * <p>
 * For CLI usage, see {@link MainCLI} which provides similar functionality with
 * command-line interface.
 * 
 * @see MainCLI
 * @see MainWindow
 */
public class Main {

    /**
     * Standard delimiter string used for parsing input files.
     * <p>
     * Contains space, tab, newline, carriage return, form feed, and semicolon.
     */
    public static String delimiter = String.valueOf(' ') + String.valueOf('\t') + String.valueOf('\n') + String.valueOf('\r') + String.valueOf('\f') + String.valueOf(';');

    /**
     * Finds the maximum integer value in a specified range of a float matrix.
     * <p>
     * This method scans a rectangular subregion of the input matrix defined by
     * the start/end line and column parameters, and returns the maximum integer
     * value found. This is commonly used to determine the number of discrete values
     * in quantized data.
     *
     * @param M The float matrix to search.
     * @param startLine The starting row index (inclusive).
     * @param endLine The ending row index (inclusive).
     * @param startCollumn The starting column index (inclusive).
     * @param endCollumn The ending column index (inclusive).
     * @return The maximum integer value found in the specified region.
     */
    public static int maximumValue(float[][] M, int startLine, int endLine, int startCollumn, int endCollumn) {
        int i, j;
        int maximum = 0;
        for (i = startLine; i <= endLine; i++) {
            for (j = startCollumn; j <= endCollumn; j++) {
                if (M[i][j] > maximum) {
                    maximum = (int) M[i][j];
                }
            }
        }
        return maximum;
    }

    /**
     * Finds the maximum integer value in a specified range of a char matrix.
     * <p>
     * This method scans a rectangular subregion of the input matrix defined by
     * the start/end line and column parameters, and returns the maximum integer
     * value found. This is commonly used in feature selection to determine the
     * number of discrete values or class labels.
     *
     * @param M The char matrix to search.
     * @param startLine The starting row index (inclusive).
     * @param endLine The ending row index (inclusive).
     * @param startCollumn The starting column index (inclusive).
     * @param endCollumn The ending column index (inclusive).
     * @return The maximum integer value found in the specified region.
     */
    public static int maximumValue(char[][] M, int startLine, int endLine, int startCollumn, int endCollumn) {
        int i, j;
        int maximum = 0;
        for (i = startLine; i <= endLine; i++) {
            for (j = startCollumn; j <= endCollumn; j++) {
                if (M[i][j] > maximum) {
                    maximum = M[i][j];
                }
            }
        }
        return maximum;
    }

    /**
     * Main entry point for the GUI-based application.
     * <p>
     * This method initializes and displays the main application window.
     * For CLI usage, see {@link MainCLI}.
     * 
     * @param args Command-line arguments (not used in the GUI version).
     * @throws IOException If there is an error reading input files or initializing the application.
     */
    public static void main(String[] args) throws IOException {

        int verbosityLevel = IOFile.VERBOSE_NONE;
        IOFile.setVerbosity(verbosityLevel);

        if (args.length > 0) {
            if (args[0].equalsIgnoreCase("-v")) {
                Vector genenames = null;
                Vector featurestitles = null;

                String inpath = args[1];
                String outpath = args[2];

                float threshold_entropy = 1;
                String type_entropy = "no_obs";
                float alpha = 1;
                float q_entropy = 1;
                float beta = 0.8f;
                int maxfeatures = 3;
                int resultsetsize = 1;

                featurestitles = IOFile.ReadDataFirstRow(inpath, 0, 1, delimiter);
                int startrow = 1;
                genenames = IOFile.ReadDataFirstCollum(inpath, startrow, delimiter);
                int startcolumn = 1;
                float[][] expressiondata = IOFile.ReadMatrix(inpath, startrow, startcolumn, delimiter);

                int nrgenes = expressiondata.length;
                int signalsize = expressiondata[0].length;

                float[] mean = new float[signalsize];
                float[] std = new float[signalsize];
                float[] lowthreshold = new float[signalsize];
                float[] hithreshold = new float[signalsize];
                int[][] quantizeddata = new int[nrgenes][signalsize];
                //data quantization
                float[][] normalizeddata = Preprocessing.quantizecolumnsMAnormal(
                        expressiondata,
                        quantizeddata,
                        2,
                        mean,
                        std,
                        lowthreshold,
                        hithreshold);
                IOFile.WriteMatrix(outpath + "normalized-log2.txt", normalizeddata, ";");
                IOFile.WriteMatrix(outpath + "quantized-log2.txt", quantizeddata, ";");

                AGN recoverednetwork = new AGN(nrgenes, signalsize, 2);
                recoverednetwork.setMean(mean);
                recoverednetwork.setStd(std);
                recoverednetwork.setLowthreshold(lowthreshold);
                recoverednetwork.setHithreshold(hithreshold);
                recoverednetwork.setTemporalsignal(expressiondata);
                recoverednetwork.setTemporalsignalquantized(quantizeddata);
                recoverednetwork.setTemporalsignalnormalized(normalizeddata);
                recoverednetwork.setLabelstemporalsignal(featurestitles);
                AGNRoutines.setGeneNames(recoverednetwork, genenames);

                AGNRoutines.RecoverNetworkfromTemporalExpression(
                        recoverednetwork,
                        null,
                        1, //datatype: 1==temporal, 2==steady-state. 
                        false,
                        threshold_entropy,
                        type_entropy,
                        alpha,
                        beta,
                        q_entropy,
                        null, //targets
                        maxfeatures,
                        3,//SFFS_stack(pilha)//1==SFS, 2==Exhaustive, 3==SFFS, 4==SFFS_stack(expandindo todos os empates encontrados).
                        false,//jCB_TargetsAsPredictors.isSelected()
                        resultsetsize,
                        null,
                        "sequential",
                        1);
                //armazenamento dos resultados
                IOFile.WriteAGNtoFile(recoverednetwork, outpath + "log2-complete.agn");
            } else {
                //execution with graphical interface...
                java.awt.EventQueue.invokeLater(new Runnable() {

                    @Override
                    public void run() {
                        new MainWindow().setVisible(true);
                    }
                });
            }
        } else {
            //execution with graphical interface...
            java.awt.EventQueue.invokeLater(new Runnable() {

                @Override
                public void run() {
                    new MainWindow().setVisible(true);
                }
            });
        }
    }
}
