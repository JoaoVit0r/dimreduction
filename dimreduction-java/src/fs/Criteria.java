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
/*** Contact: David Corr�a Martins Junior - davidjr@vision.ime.usp.br    ***/
/***          Fabr�cio Martins Lopes - fabriciolopes@vision.ime.usp.br   ***/
/***          Roberto Marcondes Cesar Junior - cesar@vision.ime.usp.br   ***/
/***************************************************************************/
/***************************************************************************/
/*** This class implements the feature selection criteria based on mean  ***/
/*** conditional entropy and coefficient of determination (COD). There   ***/
/*** are two types of penalization of non-observed instances: no_obs     ***/
/*** (penalty for non-observed instances) and poor_obs                   ***/
/*** (penalty for poorly observed instances). The Tsallis entropy        ***/
/*** (q-entropy) is adopted here, and q is a parameter. The traditional  ***/
/*** Shannon entropy is obtained for q = 1.                              ***/
/***************************************************************************/
package fs;

import java.util.Vector;
import utilities.RadixSort;

/**
 * Implements feature selection criteria based on information theory.
 * <p>
 * This class implements feature selection criteria based on mean conditional entropy
 * and coefficient of determination (COD). It supports two types of penalization for
 * non-observed instances: "no_obs" (penalty for non-observed instances) and "poor_obs"
 * (penalty for poorly observed instances).
 * <p>
 * The Tsallis entropy (q-entropy) is adopted as a generalization of the Shannon entropy,
 * with the parameter q controlling the degree of generalization. The traditional
 * Shannon entropy is obtained when q = 1, while the coefficient of determination (COD)
 * is used when q = 0.
 *
 * @see fs.Classifier
 * @see utilities.RadixSort
 */
class Criteria {

    /**
     * Vector to store the conditional probability distribution of predictors given a target gene.
     * This table is populated during entropy calculations and can be used for further analysis.
     */
    public static Vector probtable;

    /**
     * Calculates a numerical position for an instance based on its binary representation.
     * <p>
     * Given training samples sorted by feature subspace values, this method computes
     * a numerical position for the instance by converting its binary representation
     * to a decimal number.
     *
     * @param line The line number of the instance
     * @param I Vector of feature indices to consider
     * @param A The training samples matrix
     * @return A numerical position representing the instance
     */
    private static int getPositionofInstances(int line, Vector I, char[][] A) {
        StringBuffer binnumber = new StringBuffer();
        for (int i = 0; i < I.size(); i++) {
            binnumber.append((int) A[line - 1][(Integer) I.elementAt(i)]);
        }
        int position = utilities.MathRoutines.Bin2Dec(binnumber.toString());
        return (position);
    }

    /**
     * Checks if an instance at a given line is equal to the previous one.
     * <p>
     * This method compares the feature values of two consecutive instances
     * to determine if they are identical across all selected features.
     *
     * @param line The line number to check
     * @param I Vector of feature indices to consider
     * @param A The training samples matrix
     * @return true if the instance is equal to the previous one, false otherwise
     */
    private static boolean equalInstances(int line, Vector I, char[][] A) {
        for (int i = 0; i < I.size(); i++) {
            int predictor = (Integer) I.elementAt(i);
            if (A[line - 1][predictor] !=
                    A[line][predictor]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Calculates the joint entropy of a set of predictors.
     * <p>
     * This method computes the Shannon joint entropy of a set of predictors
     * based on their distribution in the training samples. The joint entropy
     * measures the uncertainty associated with the combination of predictors.
     *
     * @param n Number of quantization values for features
     * @param predictors Vector of predictor indices to consider
     * @param A The training samples matrix
     * @param c Number of classes (used for normalization)
     * @return The joint entropy value
     */
    public static float jointentropy(int n, Vector predictors, char[][] A, int c) {
        float H = 0;
        //IOFile.PrintMatrix(A);
        RadixSort.radixSort(A, predictors, n); // sorting the samples
        //IOFile.PrintMatrix(A);
        int lines = A.length;
        float pxy = 0;
        for (int j = 0; j < lines; j++) // for each sample
        {
            if (j > 0 && !equalInstances(j, predictors, A)) // next instance?
            {
                pxy /= lines;
                // calculates the entropy of previous instance
                H -= pxy * (Math.log((double) pxy) / Math.log((double) c));
                // reset the conditional probabilities to process the next instance
                pxy = 0;
            }
            pxy++;// counter of observed instances
        }
        pxy /= lines;
        H -= pxy * (Math.log((double) pxy) / Math.log((double) c));

        //debug
        //System.out.println("Entropy = " + H);
        //System.out.print("Predictors: ");
        //for (int pred = 0; pred < predictors.size(); pred++) {
        //    System.out.print(predictors.get(pred) + "  ");
        //}
        //System.out.print("\n\n");
        //fim debug

        return (H);
    }

    /**
     * Calculates the entropy or coefficient of determination for a specific instance.
     * <p>
     * This method computes either the Shannon entropy, Tsallis entropy, or coefficient
     * of determination (COD) for a given instance, applying the specified penalty for
     * non-observed or poorly observed instances.
     *
     * @param pydx Array of conditional probabilities
     * @param px Probability of the instance occurring
     * @param type Penalty type ("poor_obs" or "no_obs")
     * @param alpha Penalty value for non-observed instances
     * @param beta Penalty value for poorly observed instances
     * @param lines Number of samples
     * @param n Number of possible values for features
     * @param dim Dimension of the feature subspace being evaluated
     * @param c Number of possible classes
     * @param q Tsallis entropy parameter (q=0 for COD, q=1 for Shannon entropy)
     * @return The calculated entropy or coefficient of determination value
     */
    public static float instanceCriterion(float[] pydx, float px,
            String type, float alpha, float beta, int lines,
            int n, int dim, int c, float q) {
        float H = 0;
        // pydx becoming probabilities
        if (type.equals("poor_obs")) {
            if (px == 1) {
                for (int k = 0; k < c; k++) {
                    if (pydx[k] > 0) {
                        pydx[k] = beta;
                    } else {
                        pydx[k] = (1 - beta) / (c - 1);
                    }
                }
            } else {
                for (int k = 0; k < c; k++) {
                    pydx[k] /= px;
                }
            }
            px /= lines;
        } else if (type.equals("no_obs")) {
            for (int k = 0; k < c; k++) {
                pydx[k] /= px;
            }
            // attributing a positive mass of probabilities
            // to non-observed instances
            //notacoes usadas no artigo da BMC Bioinformatics como comentario
            px += alpha;//(fi+alpha)//ORIGINAL
            px /= (lines + alpha * Math.pow((double) n, (double) dim));//(fi+alpha)/(alpha * M + s)//ORIGINAL

            //INICIO-NOVO-FABRICIO-TESTE
            //px = (float)(1 / Math.pow((double) n, (double) dim));
            //FIM-NOVO-FABRICIO-TESTE
        }

        if (q >= 0 && q <= 0.00001) // q == 0 -> COD
        {
            float maxProb = 0;
            for (int k = 0; k < c; k++) {
                if (pydx[k] > maxProb) {
                    maxProb = pydx[k];
                }
            }
            H = px * (1 - maxProb);
            return H;
        } else if (Math.abs(q - 1) >= 0 && Math.abs(q - 1) <= 0.00001) // if (q==1 -> Shannon Entropy)
        {
            H = 0;
        } else // if (Tsallis Entropy for q != 1)
        {
            H = 1;
        }
        for (int k = 0; k < c; k++) {
            if (pydx[k] > 0) {
                if (Math.abs(q - 1) >= 0 && Math.abs(q - 1) <= 0.00001) // if (q==1 -> Shannon Entropy)
                {
                    H -= pydx[k] * (Math.log((double) pydx[k]) / Math.log((double) c));
                } else // if (Tsallis Entropy for q != 1)
                {
                    H -= Math.pow(pydx[k], q);
                }
            }
        }
        if (Math.abs(q - 1) > 0.00001)// if (Tsallis Entropy for q != 1)
        {
            H /= q - 1;
            //double entropiamaxima = ((1 - c * Math.pow((double) 1 / c, q)) / (q - 1));

            //ponderacao para recuperar o valor real da entropia na escala utilizada (q).
            //H /= ((1 - c * Math.pow((double) 1 / c, q)) / (q - 1));//REMOVIDO PARA EXPERIMENTO ARTIGO TSALLIS.
        }
        H *= px;
        return H;
    }

    // Change from returning just float to returning a Pair object containing both values
    /**
     * Result container for entropy or COD calculations.
     * <p>
     * This inner class serves as a data structure to hold both the calculated entropy (or COD) value
     * and the probability table generated during the calculation. This allows methods to return
     * both pieces of information as a single result object.
     */
    public static class EntropyResult {
        /** The calculated entropy or coefficient of determination value */
        public final float entropy;
        /** The probability table for each combination of predictor values */
        public final Vector probtable;
        
        /**
         * Constructs an EntropyResult with the specified entropy value and probability table.
         * 
         * @param entropy The calculated entropy or coefficient of determination value
         * @param probtable Vector containing probability distributions for different predictor combinations
         */
        public EntropyResult(float entropy, Vector probtable) {
            this.entropy = entropy;
            this.probtable = probtable;
        }
    }
    
    /**
     * Calculates the Mean Conditional Entropy (MCE) or Coefficient of Determination (COD)
     * for a given feature subspace.
     * <p>
     * This method evaluates the predictive power of a feature subspace by calculating
     * either the mean conditional entropy or the coefficient of determination, depending
     * on the value of the parameter q. When q=0, the method returns COD, and when q=1,
     * it returns Shannon entropy.
     * <p>
     * The method applies the specified penalty strategy for handling non-observed or
     * poorly observed patterns in the data, which is crucial for reliable feature selection
     * in high-dimensional spaces with limited samples.
     *
     * @param type Penalty type: "poor_obs" for poorly observed instances or "no_obs" for non-observed instances
     * @param alpha Penalty value for non-observed instances
     * @param beta Penalty value for poorly observed instances
     * @param n Number of possible values for features (typically 2 for binary features)
     * @param c Number of possible classes
     * @param I Vector containing indices of features to be evaluated
     * @param A Training data matrix where rows represent samples and columns represent features
     * @param q Entropy parameter: q=0 for COD, q=1 for Shannon entropy, other values for Tsallis entropy
     * @return An EntropyResult object containing both the entropy (or COD) value and the probability table
     */
    public static EntropyResult MCE_COD(String type, float alpha, float beta, int n,
            int c, Vector I, char[][] A, float q) {
        float[] pYdX = new float[c];
        float[] pY = new float[c]; // probability distribution of the classes
        float pX = 0;
        float H = 0;
        float HY = 0;
        int lines = A.length;
        int no_obs = (int) Math.pow(n, I.size()); // number of non-observed
        // instances (initially it is set to number of possible instances)
        RadixSort.radixSort(A, I, n); // sorting the samples
        //IOFile.PrintMatrix(A);
        Vector localProbtable = new Vector(no_obs);
        for (int comb = 0; comb < no_obs; comb++) {
            localProbtable.add(comb, new float[c]);
        }

        for (int j = 0; j < lines; j++) // for each sample
        {
            if (j > 0 && !equalInstances(j, I, A)) // next instance?
            {
                no_obs--; // one more observed instance

                int position = getPositionofInstances(j, I, A);
                localProbtable.setElementAt(pYdX.clone(), position);

                // calculates the entropy of previous instance
                H += instanceCriterion(pYdX, pX, type, alpha, beta, lines, n, I.size(), c, q);
                // reset the conditional probabilities to process the next instance
                for (int k = 0; k < c; k++) {
                    pYdX[k] = 0;
                }
                pX = 0;
            }
            pYdX[A[j][A[j].length - 1]]++;//table of number of observations of
            //the target value given the pattern of the predictors

            pY[A[j][A[j].length - 1]]++;  // counting the number of
            // observations of each class
            pX++;// counter of observed instances
        }
        int position = getPositionofInstances(lines, I, A);
        localProbtable.setElementAt(pYdX.clone(), position);
        H += instanceCriterion(pYdX, pX, type, alpha, beta, lines, n, I.size(), c, q);
        no_obs--;
        // Calculating the entropy of Y if the criterion is entropy, or
        // the prior error otherwise
        HY = instanceCriterion(pY, lines, "poor_obs", 0, 1, lines, 0, 0, c, q);
        if (type.equals("no_obs") && no_obs > 0) {
            double penalization = (alpha * no_obs * HY) / (lines + alpha * Math.pow(n, I.size()));
            H += penalization;
        }
        if (q >= 0 && q <= 0.00001) // q == 0 -> COD
        {
            return new EntropyResult(H / HY, localProbtable);
        }
        // mean conditional entropy
        return new EntropyResult(H, localProbtable);
    }
}
