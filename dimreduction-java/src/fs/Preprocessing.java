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
/*** This class implements preprocessing methods that can be useful such ***/
/*** as normalization, quantization and matrix transposition (if the     ***/
/*** features are disposed by lines instead of collumns)                 ***/
/***************************************************************************/
package fs;

import java.util.Vector;
import utilities.IOFile;

/**
 * Provides data preprocessing operations for feature selection and network inference.
 * <p>
 * This class implements various preprocessing methods that can be applied to data
 * matrices before feature selection or network inference, such as:
 * <ul>
 *   <li>Data normalization</li>
 *   <li>Quantization (by rows or columns)</li>
 *   <li>Matrix transposition</li>
 *   <li>Matrix copying and manipulation</li>
 * </ul>
 * <p>
 * These preprocessing operations help prepare the data for further analysis and
 * ensure consistent results across different datasets.
 * 
 * @see MainCLI#applyQuantizationAction(int, int)
 */
public class Preprocessing {

    /**
     * Special value used to mark data points that should be skipped during processing.
     */
    public static final int skipvalue = -999;
    
    /**
     * Array to store minimum negative values during quantization.
     * Used internally by quantization methods.
     */
    private static float[] minNeg = null;
    
    /**
     * Array to store maximum positive values during quantization.
     * Used internally by quantization methods.
     */
    private static float[] maxPos = null;
    /**
     * Array to store mean values during normalization operations.
     * Used internally by normalization methods.
     */
    private static float[] means = null;
    
    /**
     * Array to store standard deviation values during normalization operations.
     * Used internally by normalization methods.
     */
    private static float[] stds = null;

    /**
     * Finds the maximum and minimum values in a specified column of a matrix.
     * <p>
     * This method scans all elements in the specified column and returns the
     * maximum value in mm[0] and the minimum value in mm[1].
     *
     * @param M The input matrix to analyze
     * @param mm The array to store results (must have at least 2 elements)
     * @param col The column index to analyze
     */
    public static void MaxMinColumn(float[][] M, float[] mm, int col) {
        mm[0] = M[0][col];
        mm[1] = M[0][col];
        for (int i = 0; i < M.length; i++) {
            if (M[i][col] > mm[0]) {
                mm[0] = M[i][col];
            }
            if (M[i][col] < mm[1]) {
                mm[1] = M[i][col];
            }
        }
    }

    /**
     * Finds the maximum and minimum values in a specified row of a matrix.
     * <p>
     * This method scans elements in the specified row (excluding label column if present)
     * and returns the maximum value in mm[0] and the minimum value in mm[1].
     *
     * @param M The input matrix to analyze
     * @param mm The array to store results (must have at least 2 elements)
     * @param row The row index to analyze
     * @param label Whether to exclude the last column (1) or not (0)
     */
    public static void MaxMinRow(float[][] M, float[] mm, int row, int label) {
        mm[0] = M[row][0];
        mm[1] = M[row][0];
        for (int j = 0; j < M[0].length - label; j++) {
            if (M[row][j] > mm[0]) {
                mm[0] = M[row][j];
            }
            if (M[row][j] < mm[1]) {
                mm[1] = M[row][j];
            }
        }
    }

    /**
     * Finds the maximum and minimum values in an entire matrix.
     * <p>
     * This method scans all elements in the matrix and returns the
     * maximum value in mm[0] and the minimum value in mm[1].
     *
     * @param M The input matrix to analyze
     * @param mm The array to store results (must have at least 2 elements)
     */
    public static void MaxMin(float[][] M, float[] mm) {
        mm[0] = M[0][0];
        mm[1] = M[0][0];
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[0].length; j++) {
                if (M[i][j] > mm[0]) {
                    mm[0] = M[i][j];
                }
                if (M[i][j] < mm[1]) {
                    mm[1] = M[i][j];
                }
            }
        }
    }

    /**
     * Finds the maximum and minimum values in a 1D array.
     * <p>
     * This method scans elements in the array (skipping any elements that equal the skipvalue)
     * and returns the maximum value in mm[0] and the minimum value in mm[1].
     *
     * @param M The input array to analyze
     * @param mm The array to store results (must have at least 2 elements)
     */
    public static void MaxMin(float[] M, float[] mm) {
        mm[0] = M[0];
        mm[1] = M[0];
        for (int i = 1; i < M.length; i++) {
            if ((int) M[i] != skipvalue) {
                if (M[i] > mm[0]) {
                    mm[0] = M[i];
                }
                if (M[i] < mm[1]) {
                    mm[1] = M[i];
                }
            }
        }
    }

    /**
     * Filters microarray data by removing rows with missing values (zeros).
     * <p>
     * This method removes rows (genes) from the expression data matrix that contain
     * one or more zeros, which are typically used to represent missing values in
     * microarray data. It records which genes were kept and which were removed
     * in the provided vectors.
     *
     * @param expressiondata The input expression data matrix
     * @param geneids Array of vectors containing gene identifiers
     * @param remaingenes Vector that will be populated with indices of genes that pass the filter
     * @param removedgenes Vector that will be populated with identifiers of removed genes
     * @return A new matrix containing only the filtered data
     */
    public static float[][] FilterMA(float[][] expressiondata, Vector[] geneids,
            Vector remaingenes, Vector removedgenes) {
        //remover todas as linhas que apresentem 1 ou mais zeros...missing values.
        for (int lin = 0; lin < expressiondata.length; lin++) {
            int contz = 0;
            for (int col = 0; col < expressiondata[0].length; col++) {
                if (expressiondata[lin][col] == 0) {
                    contz++;
                }
            }
            if (contz < 1) {
                remaingenes.add(lin);
            } else {
                removedgenes.add(geneids[0].get(lin));
                IOFile.PrintlnAndLog("Gene " + (String) geneids[0].get(lin) + " was removed by filter.");
            }
        }
        IOFile.PrintlnAndLog(removedgenes.size() + " removed genes.");

        //gera a nova matriz de dados, apenas com os dados que passaram pelo filtro.
        float[][] filtereddata = new float[remaingenes.size()][expressiondata[0].length];
        for (int i = 0; i < remaingenes.size(); i++) {
            int lin = (Integer) remaingenes.get(i);
            for (int col = 0; col < expressiondata[0].length; col++) {
                filtereddata[i][col] = expressiondata[lin][col];
            }
        }
        return (filtereddata);
    }

    /**
     * Applies a logarithm base 2 transformation to expression data.
     * <p>
     * This method transforms each value in the expression data matrix using the log2
     * function, which helps to normalize the data by reducing the effect of extreme values
     * while preserving smaller variations. This is commonly used in microarray data analysis.
     * <p>
     * Values that equal the skipvalue are preserved as-is without transformation.
     *
     * @param expressiondata The input expression data matrix
     * @return A new matrix containing the log2-transformed data
     */
    public static float[][] ApplyLog2(float[][] expressiondata) {
        /*
        //calcula a media e desvio padrao de cada coluna (feature, instante de tempo).
        float[] mean = new float[expressiondata[0].length];
        float[] std = new float[expressiondata[0].length];
        boolean [] zerorows = new boolean[expressiondata.length];
        for (int col = 0; col < expressiondata[0].length; col++) {
        mean[col] = 0;
        int contz = 0;
        for (int row = 0; row < expressiondata.length; row++) {
        if (expressiondata[row][col] == 0) {
        zerorows[row] = true;
        contz++;
        }else{
        mean[col] += expressiondata[row][col];
        }
        }
        mean[col] /= (expressiondata.length-contz);
        std[col] = 0;
        for (int row = 0; row < expressiondata.length; row++) {
        if (!zerorows[row]){
        std[col] += (expressiondata[row][col] - mean[col]) * (expressiondata[row][col] - mean[col]);
        }
        }
        std[col] /= (expressiondata.length-contz-1);
        std[col] = ((Double) Math.sqrt(std[col])).floatValue();
        }*/

        //aplica o LOG2 para suavizar os picos, mantendo as pequenas variacoes.
        float[][] filtereddata = new float[expressiondata.length][expressiondata[0].length];
        for (int row = 0; row < expressiondata.length; row++) {
            for (int col = 0; col < expressiondata[0].length; col++) {
                if (expressiondata[row][col] != skipvalue) {
                    filtereddata[row][col] = ((Double) (Math.log(expressiondata[row][col]) / Math.log(2))).floatValue();
                } else {
                    filtereddata[row][col] = expressiondata[row][col];
                }
            }
        }
        return (filtereddata);
    }

    /**
     * Sorts a 2D array using the QuickSort algorithm.
     * <p>
     * This method sorts a matrix based on the values in the first column (index 0).
     * It implements the QuickSort algorithm, which has an average time complexity of O(n log n).
     * The sort is performed in-place, modifying the original matrix.
     * <p>
     * Note: This implementation assumes that each row has at least 2 columns.
     *
     * @param M The matrix to be sorted
     * @param inicio The starting index for the sort (usually 0)
     * @param fim The ending index for the sort (usually M.length-1)
     */
    public static void QuickSort(float[][] M, int inicio, int fim) {
        int i, j;
        float[] aux = new float[2];
        float pivo = M[inicio][0];
        i = inicio;
        j = fim;
        while (i < j) {
            while (M[i][0] <= pivo && i < fim) {
                i++;
            }
            while (M[j][0] >= pivo && j > inicio) {
                j--;
            }
            if (i < j) {
                aux[0] = M[i][0];
                aux[1] = M[i][1];

                M[i][0] = M[j][0];
                M[i][1] = M[j][1];

                M[j][0] = aux[0];
                M[j][1] = aux[1];
            }
        }
        if (inicio != j) {
            aux[0] = M[inicio][0];
            aux[1] = M[inicio][1];

            M[inicio][0] = M[j][0];
            M[inicio][1] = M[j][1];

            M[j][0] = aux[0];
            M[j][1] = aux[1];
        }
        if (inicio < j - 1) {
            QuickSort(M, inicio, j - 1);
        }
        if (fim > j + 1) {
            QuickSort(M, j + 1, fim);
        }
    }

    /**
     * Sorts a 1D array using the QuickSort algorithm.
     * <p>
     * This method implements the QuickSort algorithm for a one-dimensional array
     * of float values. The sort is performed in-place, modifying the original array.
     * This implementation has an average time complexity of O(n log n).
     *
     * @param M The array to be sorted
     * @param inicio The starting index for the sort (usually 0)
     * @param fim The ending index for the sort (usually M.length-1)
     */
    public static void QuickSort(float[] M, int inicio, int fim) {
        int i, j;
        float aux;
        float pivo = M[inicio];
        i = inicio;
        j = fim;
        while (i < j) {
            while (M[i] <= pivo && i < fim) {
                i++;
            }
            while (M[j] >= pivo && j > inicio) {
                j--;
            }
            if (i < j) {
                aux = M[i];
                M[i] = M[j];
                M[j] = aux;
            }
        }
        if (inicio != j) {
            aux = M[inicio];
            M[inicio] = M[j];
            M[j] = aux;
        }
        if (inicio < j - 1) {
            QuickSort(M, inicio, j - 1);
        }
        if (fim > j + 1) {
            QuickSort(M, j + 1, fim);
        }
    }

    /**
     * Sorts a 2D array in ascending order based on multiple index columns.
     * <p>
     * This method implements the QuickSort algorithm for multi-column sorting.
     * The sort is performed based on the values in the columns specified by the index array.
     * The primary sort key is index[1], with index[0] used as a secondary sort key.
     *
     * @param M The matrix to be sorted
     * @param inicio The starting index for the sort (usually 0)
     * @param fim The ending index for the sort (usually M.length-1)
     * @param index Array containing the column indexes to use for sorting
     */
    public static void QuickSortASC(float[][] M, int inicio, int fim, int[] index) {
        int i, j, k;
        float[] aux = new float[M[0].length];
        float[] pivo = new float[index.length];

        for (i = 0; i < index.length; i++) {
            pivo[i] = M[inicio][index[i]];
        }
        i = inicio;
        j = fim;

        while (i < j) {
            while (M[i][index[1]] <= pivo[1] && i < fim) {
                if (M[i][index[1]] == pivo[1]) {
                    if (M[i][index[0]] <= pivo[0]) {
                        i++;
                    } else {
                        break;
                    }
                } else {
                    i++;
                }
            }

            while (M[j][index[1]] >= pivo[1] && j > inicio) {
                if (M[j][index[1]] == pivo[1]) {
                    if (M[j][index[0]] >= pivo[0]) {
                        j--;
                    } else {
                        break;
                    }
                } else {
                    j--;
                }
            }

            if (i < j) {
                //faz a troca
                for (k = 0; k < M[0].length; k++) {
                    aux[k] = M[i][k];
                }
                for (k = 0; k < M[0].length; k++) {
                    M[i][k] = M[j][k];
                }
                for (k = 0; k < M[0].length; k++) {
                    M[j][k] = aux[k];
                    //i++;
                    //j--;
                }
            }
        }

        if (inicio != j) {
            for (k = 0; k < M[0].length; k++) {
                aux[k] = M[inicio][k];
            }
            for (k = 0; k < M[0].length; k++) {
                M[inicio][k] = M[j][k];
            }
            for (k = 0; k < M[0].length; k++) {
                M[j][k] = aux[k];            //aux = M[inicio];
                //M[inicio] = M[j];
                //M[j] = aux;
            }
        }
        if (inicio < j - 1) {
            QuickSortASC(M, inicio, j - 1, index);
        }
        if (fim > j + 1) {
            QuickSortASC(M, j + 1, fim, index);
        }
    }

    /**
     * Sorts a Vector of Vectors in descending order based on multiple index columns.
     * <p>
     * This method implements the QuickSort algorithm for multi-column sorting of a Vector
     * structure. The sort is performed based on values in specified indices of contained arrays.
     * The primary sort key is index[0], with index[1] used as a secondary sort key.
     *
     * @param M The Vector of Vectors to be sorted
     * @param inicio The starting index for the sort (usually 0)
     * @param fim The ending index for the sort (usually M.size()-1)
     * @param index Array containing the indexes to use for sorting
     */
    public static void QuickSortDSC(Vector M, int inicio, int fim, int[] index) {
        int i, j;
        Vector aux;
        double[] pivo = new double[index.length];

        for (i = 0; i < index.length; i++) {
            pivo[i] = ((double[]) ((Vector) M.get(inicio)).get(0))[index[i]];
        }
        i = inicio;
        j = fim;

        while (i < j) {
            while (((double[]) ((Vector) M.get(i)).get(0))[index[0]] <= pivo[0] && i < fim) {
                if (((double[]) ((Vector) M.get(i)).get(0))[index[0]] == pivo[0]) {
                    if (((double[]) ((Vector) M.get(i)).get(0))[index[1]] >= pivo[1]) {
                        i++;
                    } else {
                        break;
                    }
                } else {
                    i++;
                }
            }

            while (((double[]) ((Vector) M.get(j)).get(0))[index[0]] >= pivo[0] && j > inicio) {
                if (((double[]) ((Vector) M.get(j)).get(0))[index[0]] == pivo[0]) {
                    if (((double[]) ((Vector) M.get(j)).get(0))[index[1]] <= pivo[1]) {
                        j--;
                    } else {
                        break;
                    }
                } else {
                    j--;
                }
            }

            if (i < j) {
                aux = (Vector) M.get(i);
                M.set(i, M.get(j));
                M.set(j, aux);
            }
        }

        if (inicio != j) {
            aux = (Vector) M.get(inicio);
            M.set(inicio, M.get(j));
            M.set(j, aux);
        }
        if (inicio < j - 1) {
            QuickSortDSC(M, inicio, j - 1, index);
        }
        if (fim > j + 1) {
            QuickSortDSC(M, j + 1, fim, index);
        }
    }

    /**
     * Sorts a Vector of Vectors using the Selection Sort algorithm.
     * <p>
     * This method sorts the vector based on the first element (index 0) of each contained Vector.
     * Selection Sort is used, which works by finding the minimum element and placing it at the front.
     *
     * @param M The Vector of Vectors to be sorted based on first element values
     */
    public static void SelectionSort(Vector<Vector> M) {
        Vector aux;
        float minvalue = 0;
        int minposition = 0;
        for (int j = 0; j < M.size() - 1; j++) {
            minvalue = (Float) M.get(j).get(0);
            minposition = j;
            for (int i = j + 1; i < M.size(); i++) {
                if ((Float) M.get(i).get(0) < minvalue) {
                    minvalue = (Float) M.get(i).get(0);
                    minposition = i;
                }
            }
            if (minposition != j) {
                aux = M.get(j);
                M.set(j, M.get(minposition));
                M.set(minposition, aux);
            }
        }
    }

    /**
     * Sorts a Vector of Vectors using the Selection Sort algorithm based on a specified index.
     * <p>
     * This method sorts the vector based on the element at the specified index of each contained Vector.
     * Selection Sort is used, which works by finding the minimum element and placing it at the front.
     *
     * @param M The Vector of Vectors to be sorted
     * @param index The index within each contained Vector to use for comparison
     */
    public static void SelectionSort(Vector<Vector> M, int index) {
        Vector aux;
        int minvalue = 0;
        int minposition = 0;
        for (int j = 0; j < M.size() - 1; j++) {
            minvalue = (Integer) M.get(j).get(index);
            minposition = j;
            for (int i = j + 1; i < M.size(); i++) {
                if ((Integer) M.get(i).get(index) < minvalue) {
                    minvalue = (Integer) M.get(i).get(index);
                    minposition = i;
                }
            }
            if (minposition != j) {
                aux = M.get(j);
                M.set(j, M.get(minposition));
                M.set(minposition, aux);
            }
        }
    }

    /**
     * Sorts a Vector of Vectors using the Bubble Sort algorithm.
     * <p>
     * This method sorts the vector based on the first element (index 0) of each contained Vector.
     * Bubble Sort is used, which repeatedly steps through the list, compares adjacent elements
     * and swaps them if they are in the wrong order.
     *
     * @param M The Vector of Vectors to be sorted based on first element values
     */
    public static void BubbleSort(Vector<Vector> M) {
        Vector aux;
        boolean change = true;
        while (change) {
            change = false;
            for (int i = 0; i < M.size() - 1; i++) {
                if ((Double) M.get(i).get(0) > (Double) M.get(i + 1).get(0)) {
                    aux = M.get(i);
                    M.set(i, M.get(i + 1));
                    M.set(i + 1, aux);
                    change = true;
                }
            }
        }
    }

    /**
     * Sorts an array of integers in descending order using Bubble Sort algorithm.
     * <p>
     * This method sorts the values array in-place in descending order and returns
     * an array of indices that maps from the sorted positions to the original positions.
     *
     * @param values The array of integers to sort in descending order (modified in place)
     * @return An array of indices mapping sorted positions to original positions
     */
    public static int[] BubbleSortDEC(int[] values) {
        boolean change = true;
        int aux;
        int[] indexes = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            indexes[i] = i;
        }
        while (change) {
            change = false;
            for (int i = 0; i < values.length - 1; i++) {
                if (values[i] < values[i + 1]) {
                    //change values order
                    aux = values[i];
                    values[i] = values[i + 1];
                    values[i + 1] = aux;

                    //change indexes order
                    aux = indexes[i];
                    indexes[i] = indexes[i + 1];
                    indexes[i + 1] = aux;

                    change = true;
                }
            }
        }
        return (indexes);
    }

    /**
     * Sorts a Vector of Vectors using the QuickSort algorithm.
     * <p>
     * This method sorts the vector based on the first element (index 0) of each contained Vector.
     * QuickSort is used, which is generally faster than other sorting algorithms for large datasets.
     *
     * @param M The Vector of Vectors to be sorted based on first element values
     * @param inicio The starting index for the sort (usually 0)
     * @param fim The ending index for the sort (usually M.size()-1)
     */
    public static void QuickSort(Vector<Vector> M, int inicio, int fim) {
        int i, j;
        Vector aux;
        float pivo = (Float) M.get(inicio).get(0);

        i = inicio;
        j = fim;

        while (i < j) {
            while ((Float) M.get(i).get(0) <= pivo && i < fim) {
                i++;
            }
            while ((Float) M.get(j).get(0) >= pivo && j > inicio) {
                j--;
            }
            if (i < j) {
                aux = M.get(i);
                M.set(i, M.get(j));
                M.set(j, aux);
            }
        }

        if (inicio != j) {
            aux = M.get(inicio);
            M.set(inicio, M.get(j));
            M.set(j, aux);
        }
        if (inicio < j - 1) {
            QuickSort(M, inicio, j - 1);
        }
        if (fim > j + 1) {
            QuickSort(M, j + 1, fim);
        }
    }

    /**
     * Sorts a Vector of Double values in ascending order using the QuickSort algorithm.
     * <p>
     * This method sorts the vector of Double values in ascending order using QuickSort,
     * which is generally faster than other sorting algorithms for large datasets.
     *
     * @param M The Vector of Double values to be sorted
     * @param inicio The starting index for the sort (usually 0)
     * @param fim The ending index for the sort (usually M.size()-1)
     */
    public static void QuickSortASC(Vector M, int inicio, int fim) {
        int i, j;
        double aux;
        double pivo = (Double) M.get(inicio);

        i = inicio;
        j = fim;

        while (i < j) {
            while ((Double) M.get(i) <= pivo && i < fim) {
                i++;
            }
            while ((Double) M.get(j) >= pivo && j > inicio) {
                j--;
            }
            if (i < j) {
                aux = (Double) M.get(i);
                M.set(i, M.get(j));
                M.set(j, aux);
            }
        }

        if (inicio != j) {
            aux = (Double) M.get(inicio);
            M.set(inicio, M.get(j));
            M.set(j, aux);
        }
        if (inicio < j - 1) {
            QuickSortASC(M, inicio, j - 1);
        }
        if (fim > j + 1) {
            QuickSortASC(M, j + 1, fim);
        }
    }

    /**
     * Creates a formatted result list from a matrix and stores structured data in a Vector.
     * <p>
     * This method processes a matrix of results, sorts it using specified columns,
     * groups related entries, and creates a formatted text representation of the results.
     * The method also populates the provided Vector with structured data extracted from
     * the results matrix.
     *
     * @param R The results matrix containing data to process
     * @param ordenado A Vector that will be populated with structured result data
     * @return A StringBuffer containing a formatted text representation of the results
     */
    public static StringBuffer MakeResultList(float[][] R, Vector ordenado) {
        int[] I = new int[2];
        I[0] = 0;
        I[1] = 2;
        //IOFile.PrintMatrix(R);
        QuickSortASC(R, 0, R.length - 1, I); // sorting the samples
        //IOFile.PrintMatrix(R);
        int i = 0;
        int maxc = 0;
        while (i < R.length) {
            Vector preditos = new Vector();
            preditos.add((int) R[i][1]);
            int preditor = (int) R[i][0];
            float entropia = (float) R[i][2];
            int count = 1;
            i++;
            while (i < R.length && R[i][0] == preditor && Math.abs(R[i][2] - entropia) < 0.001) {
                preditos.add((int) R[i][1]);
                count++;
                i++;
            }
            if (preditos.size() > maxc) {
                maxc = preditos.size();
            }
            Vector item = new Vector();
            item.add(new double[]{preditor, entropia, count});
            item.add(preditos);
            ordenado.add(item);
        }

        I[0] = 1;
        I[1] = 2;
        QuickSortDSC(ordenado, 0, ordenado.size() - 1, I); // sorting the samples
        StringBuffer res = new StringBuffer("predictor\tentropy \tfrequency \n");
        for (i = 0; i < ordenado.size(); i++) {
            Vector item = (Vector) ordenado.get(i);
            double[] item1 = (double[]) item.get(0);

            int preditor = (int) item1[0];
            float entropia = (float) item1[1];
            int count = (int) item1[2];
            res.append(preditor + "\t\t" + entropia + "\t\t" + count + "\n");
        }
        return (res);
    }
    /**
     * Creates the transpose of a matrix.
     * <p>
     * This method creates a new matrix that is the transpose of the input matrix.
     * In the transposed matrix, rows become columns and columns become rows.
     * This is useful when the features are arranged by rows instead of columns.
     * 
     * @param M The input matrix to transpose
     * @return A new matrix that is the transpose of the input matrix
     */
    public static double[][] transpose(double[][] M) {
        int lines = M.length;
        int columns = M[0].length;
        double[][] Mtrans = new double[columns][lines];
        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < columns; j++) {
                Mtrans[j][i] = M[i][j];
            }
        }
        return Mtrans;
    }
    //inverte os tempos de expressao, de forma que os targets passem
    //a ser os preditores, i.e. os preditores passem a considerar o valor 
    //do target no instante de tempo posterior.

    /**
     * Inverts the order of columns in a matrix, effectively reversing each row.
     * <p>
     * This method creates a new matrix where the column order is reversed from the original.
     * For example, in the new matrix, the first column will be the last column from the original,
     * the second column will be the second-to-last from the original, and so on.
     * </p>
     * 
     * <p>
     * In gene network analysis, this method is particularly useful when performing
     * target-as-predictor analysis, where the temporal relationship needs to be inverted.
     * By reversing column order, we can analyze how future states (originally targets)
     * might predict past states (originally predictors), effectively inverting the
     * causal relationship being explored.
     * </p>
     * 
     * <p>
     * Note that this is not a matrix transposition (rows becoming columns); it only
     * reverses the order of columns while preserving the row structure.
     * </p>
     *
     * @param M The input matrix whose columns will be inverted.
     * @return A new matrix with the columns in reverse order.
     * @see MainCLI#networkInferenceActionPerformed()
     */
    public static float[][] InvertColumns(float[][] M) {
        int lines = M.length;
        int columns = M[0].length;
        float[][] MInv = new float[lines][columns];
        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < columns; j++) {
                MInv[i][j] = M[i][columns - j - 1];
            }
        }
        return MInv;
    }
    // receives a matrix M and returns a copy of M (copy M)

    /**
     * Creates a deep copy of a 2D floating-point matrix.
     * 
     * <p>
     * This method creates a completely independent copy of the input matrix, 
     * ensuring that subsequent modifications to either the original or the copy
     * do not affect each other. This is essential for preserving the original data
     * while performing potentially destructive operations on the working copy.
     * </p>
     * 
     * <p>
     * In the gene network analysis pipeline, this method is particularly important
     * for creating the working data matrix (Md) from the original matrix (Mo),
     * allowing operations like quantization and normalization to be applied while
     * preserving the original data for reference or alternative processing paths.
     * </p>
     *
     * @param M The input matrix to be copied.
     * @return A new matrix containing the same values as the input matrix.
     * @throws NullPointerException If the input matrix is null or contains null rows.
     * @throws ArrayIndexOutOfBoundsException If the input matrix is not rectangular
     *                                       (all rows must have the same length).
     * @see MainCLI#applyQuantizationAction(int, int)
     */
    public static float[][] copyMatrix(float[][] M) {
        int lines = M.length;
        int columns = M[0].length;
        float[][] cM = new float[lines][columns];
        for (int i = 0; i < lines; i++) {
            for (int j = 0; j < columns; j++) {
                cM[i][j] = M[i][j];
            }
        }
        return cM;
    }
    /**
     * Applies a normal transformation to each column of the matrix M.
     * <p>
     * Each value is transformed to (value - mean) / stddev for its column.
     *
     * @param M the matrix to normalize (modified in place)
     * @param extreme_values whether to compute mean/stddev for each column
     * @param label indicates if the last column is a label (1) or not (0)
     */
    public static void normalTransformcolumns(float[][] M, boolean extreme_values,
            int label) {
        int lines = M.length;
        int columns = M[0].length;

        if (extreme_values) {
            means = new float[columns - label];//CLASS LABEL AT THE FINAL COLUMN?
            stds = new float[columns - label];//CLASS LABEL AT THE FINAL COLUMN?
        } else if (minNeg == null) {
            new fs.FSException("Error on applying normal transform.", false);
        }

        for (int j = 0; j < columns - label; j++) // for each feature
        {
            if (M[0][j] != skipvalue) {
                if (extreme_values) {
                    // calculating the mean of the feature values
                    float sum = 0;
                    for (int i = 0; i < lines; i++) {
                        sum += M[i][j];
                    }
                    means[j] = sum / lines;

                    // calculating the standard deviation of the feature values
                    stds[j] = 0f;
                    for (int i = 0; i < lines; i++) {
                        stds[j] += (M[i][j] - means[j]) * (M[i][j] - means[j]);
                    }
                    stds[j] /= (lines - 1);
                    stds[j] = ((Double) Math.sqrt(stds[j])).floatValue();
                }
                // are the values of the reffered feature the same for all samples?
                if (stds[j] > 0) {
                    // each feature value is subtracted by the mean and is divided
                    // by the standard deviation
                    for (int i = 0; i < lines; i++) {
                        M[i][j] -= means[j];
                        M[i][j] /= stds[j];
                    }
                } else {
                    for (int i = 0; i < lines; i++) {
                        M[i][j] = 0;
                    }
                }
            }
        }
    }
    // applies a normal transformation to the matrix M
    //foi adicionado a variavel label, para indicar se os rotulos est�o na 
    //ultima columa (label = 1) ou se nao existe rotulo (label=0).

    /**
     * Applies a normal transformation to each row of the matrix M.
     * <p>
     * Each value in a row is transformed to (value - mean) / stddev for its row.
     * This method is useful for normalizing data where each row represents
     * an entity to be compared with others.
     *
     * @param M The matrix to normalize (modified in place)
     * @param extreme_values Whether to compute mean/stddev for each row
     * @param label Indicates if the last column is a label (1) or not (0)
     */
    public static void normalTransformlines(float[][] M, boolean extreme_values,
            int label) {
        int lines = M.length;
        int columns = M[0].length;

        if (extreme_values) {
            means = new float[lines];
            stds = new float[lines];
        } else if (minNeg == null) {
            new fs.FSException("Error on applying normal transform.", false);
        }

        for (int j = 0; j < lines; j++) // for each sample
        {
            if (extreme_values) {
                // calculating the mean of the sample values
                double sum = 0;
                for (int i = 0; i < columns - label; i++) {
                    sum += M[j][i];
                }
                means[j] = ((Double) (sum / (columns - label))).floatValue();

                // calculating the standard deviation of the feature values
                stds[j] = 0;
                for (int i = 0; i < columns - label; i++) {
                    stds[j] += (M[j][i] - means[j]) * (M[j][i] - means[j]);
                }
                stds[j] /= (columns - label - 1);
                stds[j] = ((Double) Math.sqrt(((Float) stds[j]).doubleValue())).floatValue();
            }
            // are the values of the reffered feature the same for all samples?
            if (stds[j] > 0) {
                // each feature value is subtracted by the mean and is divided
                // by the standard deviation
                for (int i = 0; i < columns - label; i++) {
                    M[j][i] -= means[j];
                    M[j][i] /= stds[j];
                }
            } else {
                for (int i = 0; i < columns - label; i++) {
                    M[j][i] = 0;
                }
            }
        }
    }

    /**
     * Scales each column of a matrix to a range from 0 to maxvalue-1.
     * <p>
     * This method normalizes each column independently based on its minimum and maximum values.
     * For each value v in column c, the scaled value is: (maxvalue-1) * (v - min_c) / (max_c - min_c).
     * 
     * @param M The matrix to scale (modified in place)
     * @param maxvalue The maximum value for the scaled data (output range is 0 to maxvalue-1)
     * @param label Indicates if the last column is a label (1) or not (0)
     */
    public static void ScaleColumn(float[][] M, int maxvalue, int label) {
        float[] maxmin = new float[2];
        float normalizedvalue = 0;
        for (int col = 0; col < M[0].length - label; col++) {
            //encontra os valores extremos maximo no indice 0 e minimo no indice 1.
            MaxMinColumn(M, maxmin, col);
            for (int lin = 0; lin < M.length; lin++) {
                normalizedvalue = (maxvalue - 1) * (M[lin][col] - maxmin[1])
                        / (maxmin[0] - maxmin[1]);
                M[lin][col] = normalizedvalue;
            }
        }
    }

    /**
     * Scales each row of a matrix to a range from 0 to maxvalue-1.
     * <p>
     * This method normalizes each row independently based on its minimum and maximum values.
     * For each value v in row r, the scaled value is: (maxvalue-1) * (v - min_r) / (max_r - min_r).
     * 
     * @param M The matrix to scale (modified in place)
     * @param maxvalue The maximum value for the scaled data (output range is 0 to maxvalue-1)
     * @param label Indicates if the last column is a label (1) or not (0)
     */
    public static void ScaleRow(float[][] M, int maxvalue, int label) {
        float[] maxmin = new float[2];
        float normalizedvalue = 0;
        for (int row = 0; row < M.length; row++) {
            //encontra os valores extremos maximo no indice 0 e minimo no indice 1.
            MaxMinRow(M, maxmin, row, label);
            for (int col = 0; col < M[0].length - label; col++) {
                normalizedvalue = (maxvalue - 1) * (M[row][col] - maxmin[1])
                        / (maxmin[0] - maxmin[1]);
                M[row][col] = normalizedvalue;
            }
        }
    }

    /**
     * Normalizes and quantizes a matrix to integer values from 0 to qd-1.
     * <p>
     * This method scales the values in each row of the matrix based on the
     * row's min and max values, then assigns discrete values from 0 to qd-1
     * using thresholds.
     * 
     * @param M The matrix to normalize and quantize (modified in place)
     * @param qd The quantization degree (number of discrete values)
     * @param label Indicates if the last column is a label (1) or not (0)
     */
    public static void normalize(float[][] M, int qd, int label) {
        float[] threshold = new float[qd - 1];
        float[] maxmin = new float[2];
        float normalizedvalue = 0;

        float increment = (qd - 1) / ((float) qd);

        for (int k = 0; k < (qd - 1); k++) {
            threshold[k] = increment;
            increment += increment;
        }

        for (int lin = 0; lin < M.length; lin++) {
            //encontra os valores extremos maximo no indice 0 e minimo no indice 1.
            MaxMin(M[lin], maxmin);
            for (int col = 0; col < M[lin].length - label; col++) {
                normalizedvalue = (qd - 1) * (M[lin][col] - maxmin[1])
                        / (maxmin[0] - maxmin[1]);

                int k = 0;
                for (k = 0; k < (qd - 1); k++) {
                    if (threshold[k] >= normalizedvalue) {
                        break;
                    }
                }
                M[lin][col] = k;
                /*
                // obtaining the thresholds for quantization
                int indThreshold = 0;
                double increment = - maxmin[1] / ((double) qd / 2);
                double [] threshold = new double [qd - 1];
                for (double i = maxmin[1] + increment; i < 0; i += increment, indThreshold++)
                threshold[indThreshold] = i;
                increment = maxmin[0] / ((double) qd / 2);
                indThreshold = qd - 2;
                for (double i = maxmin[0] - increment; i > 0 ; i -= increment, indThreshold--)
                threshold[indThreshold] = i;

                // quantizing the feature values
                int k = 0;
                for (k = 0; k < qd; k++)
                if (threshold[k] >= M[lin][col])
                break;
                M[lin][col] = k;
                 */
            }
        }
    }
    /**
     * Quantizes data by rows (variables).
     * <p>
     * This method applies quantization to each row of the input matrix. It first
     * normalizes the data using a normal transformation by rows, then applies quantization
     * based on the specified quantization degree (number of discrete values).
     * <p>
     * The method creates threshold values for positive and negative values separately,
     * using the extreme values found in each row.
     *
     * @param M The matrix to quantize (modified in place).
     * @param qd The quantization degree (number of discrete values).
     * @param extreme_values Whether to compute extreme values for each row.
     * @param label Indicates if the last column is a label (1) or not (0).
     * @throws FSException If there is an error during quantization.
     * @see #normalTransformlines(float[][], boolean, int)
     * @see MainCLI#applyQuantizationAction(int, int)
     */
    public static void quantizerows(float[][] M, int qd, boolean extreme_values, int label) {
        int lines = M.length;
        int columns = M[0].length;
        normalTransformlines(M, extreme_values, label); // applying a normal transformation to the matrix M

        /*
        if (extreme_values) {
        minNeg = new float[lines];
        maxPos = new float[lines];
        } else if (minNeg == null) {
        new fs.FSException("Error on data quantization.", false);
        }
         *
         */

        //foi adicionado a variavel label, para indicar se os rotulos est�o 
        //na ultima columa (label = 1) ou se nao existe rotulo (label=0).
        for (int j = 0; j < lines; j++) // for each feature
        {
            if (M[0][j] != skipvalue) {
                // retrieving the negative and positive values of the considered
                // feature
                Vector negatives = new Vector();
                Vector positives = new Vector();
                float meanneg = 0;
                float meanpos = 0;
                for (int i = 0; i < columns - label; i++) {
                    if (M[j][i] < 0) {
                        negatives.add(M[j][i]);
                        meanneg += M[j][i];
                    } else {
                        positives.add(M[j][i]);
                        meanpos += M[j][i];
                    }
                }
                meanneg /= negatives.size();
                meanpos /= positives.size();

                /*
                if (extreme_values) {
                // are the values of the reffered feature the same for all
                // samples?
                if (stds[j] == 0) {
                continue;
                // retrieving the smallest negative value
                }

                minNeg[j] = (Float) negatives.elementAt(0);
                for (int i = 1; i < negatives.size(); i++) {
                if (minNeg[j] > (Float) negatives.elementAt(i)) {
                minNeg[j] = (Float) negatives.elementAt(i);                    // retrieving the largest positive value
                }
                }
                maxPos[j] = (Float) positives.elementAt(0);
                for (int i = 1; i < positives.size(); i++) {
                if (maxPos[j] < (Float) positives.elementAt(i)) {
                maxPos[j] = (Float) positives.elementAt(i);
                }
                }
                }
                 *
                 */
                // obtaining the thresholds for quantization
                int indThreshold = 0;
                double increment = -meanneg / ((double) qd / 2);
                double[] threshold = new double[qd - 1];
                for (double i = meanneg + increment; i < 0; i += increment, indThreshold++) {
                    threshold[indThreshold] = i;
                }
                increment = meanpos / ((double) qd / 2);
                indThreshold = qd - 2;
                for (double i = meanpos - increment; i > 0; i -= increment, indThreshold--) {
                    threshold[indThreshold] = i;
                    // quantizing the feature values
                }
                for (int i = 0; i < columns - label; i++) {
                    int k;
                    for (k = 0; k < qd - 1; k++) {
                        if (threshold[k] >= M[j][i]) {
                            break;
                        }
                    }
                    M[j][i] = k;
                }
            }
        }
    }
    /**
     * Quantizes data by columns (features/samples).
     * <p>
     * This method applies quantization to each column of the input matrix. It first
     * normalizes the data using a normal transformation, then applies quantization
     * based on the specified quantization degree (number of discrete values).
     * <p>
     * The method creates threshold values for positive and negative values separately,
     * using the extreme values found in each column.
     *
     * @param M The matrix to quantize (modified in place).
     * @param qd The quantization degree (number of discrete values).
     * @param extreme_values Whether to compute extreme values for each column.
     * @param label Indicates if the last column is a label (1) or not (0).
     * @throws FSException If there is an error during quantization.
     * @see #normalTransformcolumns(float[][], boolean, int)
     * @see MainCLI#applyQuantizationAction(int, int)
     */
    public static void quantizecolumns(float[][] M, int qd, boolean extreme_values, int label) {
        int lines = M.length;
        int columns = M[0].length;
        normalTransformcolumns(M, extreme_values, label); // applying a normal transformation to the matrix M

        /*
        if (extreme_values) {
        minNeg = new float[columns - label];
        maxPos = new float[columns - label];
        } else if (minNeg == null) {
        new fs.FSException("Error on data quantization.", false);
        }
         *
         */

        //foi adicionado a variavel label, para indicar se os rotulos est�o
        //na ultima columa (label = 1) ou se nao existe rotulo (label=0).
        for (int j = 0; j < columns - label; j++) // for each feature
        {
            if (M[0][j] != skipvalue) {
                // retrieving the negative and positive values of the considered
                // feature
                Vector negatives = new Vector();
                Vector positives = new Vector();
                double meanneg = 0;
                double meanpos = 0;
                for (int i = 0; i < lines; i++) {
                    if (M[i][j] < 0) {
                        negatives.add(M[i][j]);
                        meanneg += M[i][j];
                    } else {
                        positives.add(M[i][j]);
                        meanpos += M[i][j];
                    }
                }
                meanneg /= negatives.size();
                meanpos /= positives.size();

                /*
                if (extreme_values) {
                // are the values of the reffered feature the same for all
                // samples?
                if (stds[j] == 0) {
                continue;
                // retrieving the smallest negative value
                }
                if (negatives.isEmpty() || positives.isEmpty()) {
                continue;
                }
                /* REMOVED. NOW CONSIDERING THE AVERAGES.
                minNeg[j] = (Float) negatives.elementAt(0);
                for (int i = 1; i < negatives.size(); i++) {
                if (minNeg[j] > (Float) negatives.elementAt(i)) {
                minNeg[j] = (Float) negatives.elementAt(i);                    // retrieving the largest positive value
                }
                }
                maxPos[j] = (Float) positives.elementAt(0);
                for (int i = 1; i < positives.size(); i++) {
                if (maxPos[j] < (Float) positives.elementAt(i)) {
                maxPos[j] = (Float) positives.elementAt(i);
                }
                }
                }*/

                // obtaining the thresholds for quantization
                int indThreshold = 0;
                double increment = -meanneg / ((double) qd / 2);
                double[] threshold = new double[qd - 1];
                for (double i = meanneg + increment; i < 0; i += increment, indThreshold++) {
                    threshold[indThreshold] = i;
                }
                increment = meanpos / ((double) qd / 2);
                indThreshold = qd - 2;
                for (double i = meanpos - increment; i > 0; i -= increment, indThreshold--) {
                    threshold[indThreshold] = i;
                    // quantizing the feature values
                }
                for (int i = 0; i < lines; i++) {
                    int k;
                    for (k = 0; k < qd - 1; k++) {
                        if (threshold[k] >= M[i][j]) {
                            break;
                        }
                    }
                    M[i][j] = k;
                }
            }
        }
    }

    /**
     * Performs specialized quantization of matrix columns for microarray data.
     * <p>
     * This method normalizes and quantizes the data in each column of the input matrix.
     * It's designed specifically for microarray data processing with special handling
     * for 2-state and 3-state quantization. The method computes statistics for each column
     * and stores quantized values in the provided quantizeddata array.
     *
     * @param M The input matrix to process
     * @param quantizeddata Array to store the quantized integer values
     * @param qd The quantization degree (2 or 3)
     * @param mean Array to store the mean value for each column
     * @param std Array to store the standard deviation for each column
     * @param lowthreshold Array to store the lower threshold for each column
     * @param hithreshold Array to store the upper threshold for each column
     * @return A normalized copy of the input matrix
     */
    public static float[][] quantizecolumnsMAnormal(
            float[][] M,
            int[][] quantizeddata,
            int qd,
            float[] mean,
            float[] std,
            float[] lowthreshold,
            float[] hithreshold) {

        int totalrows = M.length;
        int totalcols = M[0].length;
        //int[][] quantizeddata = new int[totalrows][totalcols];
        float[][] auxM = Preprocessing.copyMatrix(M);//copy of matrix
        normalTransformcolumns(auxM, true, 0); // applying a normal transformation to the matrix M
        minNeg = new float[totalcols];
        maxPos = new float[totalcols];

        //foi adicionado a variavel label, para indicar se os rotulos est�o
        //na ultima columa (label = 1) ou se nao existe rotulo (label=0).
        for (int col = 0; col < totalcols; col++) // for each feature
        {
            if (M[0][col] != skipvalue) {
                // calculating the mean of the feature values
                float sum = 0;
                float[] colvalues = new float[totalrows];
                for (int row = 0; row < totalrows; row++) {
                    sum += M[row][col];
                    colvalues[row] = auxM[row][col];
                }
                mean[col] = sum / totalrows;
                std[col] = 0;
                for (int row = 0; row < totalrows; row++) {
                    std[col] += (M[row][col] - mean[col]) * (M[row][col] - mean[col]);
                }
                std[col] /= (totalrows - 1);
                std[col] = ((Double) Math.sqrt(std[col])).floatValue();

                IOFile.PrintArray(colvalues);
                QuickSort(colvalues, 0, totalrows - 1);
                IOFile.PrintArray(colvalues);

                // retrieving the negative and positive values of the considered feature
                Vector negatives = new Vector();
                Vector positives = new Vector();
                float meanpos = 0;
                float meanneg = 0;
                float meantotal = 0;
                for (int i = 0; i < totalrows; i++) {
                    if (auxM[i][col] < 0) {
                        negatives.add(auxM[i][col]);
                        meanneg += auxM[i][col];
                    } else {
                        positives.add(auxM[i][col]);
                        meanpos += auxM[i][col];
                    }
                    meantotal += auxM[i][col];
                }
                // are the values of the refered feature the same for all samples?
                if (std[col] == 0) {
                    continue;
                    // retrieving the smallest negative value
                }
                if (negatives.isEmpty() || positives.isEmpty()) {
                    continue;
                }
                meanneg /= negatives.size();
                meanpos /= positives.size();
                meantotal /= totalrows;
                /*
                minNeg[col] = (Float) negatives.elementAt(0);
                for (int i = 1; i < negatives.size(); i++) {
                if (minNeg[col] > (Float) negatives.elementAt(i)) {
                minNeg[col] = (Float) negatives.elementAt(i);// retrieving the largest positive value
                }
                }
                maxPos[col] = (Float) positives.elementAt(0);
                for (int i = 1; i < positives.size(); i++) {
                if (maxPos[col] < (Float) positives.elementAt(i)) {
                maxPos[col] = (Float) positives.elementAt(i);
                }
                }
                 */
                // obtaining the thresholds for quantization
                //int indThreshold = 0;
                //double increment = -minNeg[col] / ((double) qd / 2);
                //double increment = -meanneg / ((double) qd / 2);
                double[] threshold = new double[qd - 1];
                //for (double i = minNeg[col] + increment; i < 0; i += increment, indThreshold++) {
                //    threshold[indThreshold] = i;
                //}
                //increment = maxPos[col] / ((double) qd / 2);
                //increment = meanpos / ((double) qd / 2);
                //indThreshold = qd - 2;
                //for (double i = maxPos[col] - increment; i > 0; i -= increment, indThreshold--) {
                //    threshold[indThreshold] = i;
                // quantizing the feature values
                //}

                //int index1stq = Math.round(0.25f*(totalrows+1));
                //int index3rdq = Math.round(0.75f*(totalrows+1));
                //threshold[0] = colvalues[index1stq];
                //threshold[1] = colvalues[index3rdq];

                if (qd == 2) {

                    //threshold[0] = (meanneg+meanpos)/2;

                    int index3rdq = Math.round(0.75f * (totalrows + 1));
                    threshold[0] = colvalues[index3rdq];

                    lowthreshold[col] = (float) threshold[0];
                    hithreshold[col] = (float) threshold[0];
                } else if (qd == 3) {
                    threshold[0] = meanneg;
                    threshold[1] = meanpos;
                    lowthreshold[col] = (float) threshold[0];
                    hithreshold[col] = (float) threshold[1];
                } else {
                    return null;
                }

                int count0 = 0;
                int count1 = 0;
                for (int i = 0; i < totalrows; i++) {
                    int k;
                    for (k = 0; k < qd - 1; k++) {
                        if (auxM[i][col] <= threshold[k]) {
                            break;
                        }
                    }
                    if (qd == 3) {
                        //dados Marie-Anne
                        if (k == 2 || k == 0) {
                            k = 1;
                        } else if (k == 1) {
                            k = 0;
                        }
                    }
                    if (k == 0) {
                        count0++;
                    } else {
                        count1++;
                    }
                    quantizeddata[i][col] = k;
                }
                IOFile.PrintlnAndLog("Totais quantizados na coluna " + col + ": " + (count0 + count1));
                IOFile.PrintlnAndLog("zeros = " + count0);
                IOFile.PrintlnAndLog("ums = " + count1);
                IOFile.PrintlnAndLog();
            } else {
                for (int i = 0; i < totalrows; i++) {
                    quantizeddata[i][col] = (int) M[i][col];
                }
            }
        }
        return (auxM);
    }

    /*  public static void main (String [] args) throws IOException{
    double [][] M = ReadFile.readMatrixDouble(args[0]);
    int qd = Integer.parseInt(args[1]);
    int lines = M.length;
    int collumns = M[0].length;
    quantize(M,qd);
    for (int i = 0; i < lines; i++)
    {
    for (int j = 0; j < collumns; j++)
    System.out.print(M[i][j]+" ");
    IOFile.PrintlnAndLog();
    }
    } */
}
