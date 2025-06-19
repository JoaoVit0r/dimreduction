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
/*** This class implements conditional entropy.                          ***/
/*** There are two types of mean conditional entropy:                    ***/
/*** no_obs (penalty for non-observed instances) and                     ***/
/*** poor_obs (penalty for poorly observed instances).                   ***/
/*** The Tsallis entropy (q-entropy) is adopted here, and q is a         ***/
/*** parameter. The traditional Shannon entropy is obtained for q = 1.   ***/
/***************************************************************************/
package fs;

import agn.AGN;
import java.io.Serializable;
import java.util.Vector;
import utilities.IOFile;
import utilities.MathRoutines;
import utilities.Timer;

/**
 * Implements feature selection algorithms based on information theory.
 * <p>
 * This class provides implementations of different feature selection algorithms including:
 * Sequential Forward Selection (SFS), Sequential Floating Forward Selection (SFFS), and
 * an exhaustive search method. These algorithms select the most relevant features
 * from a dataset based on entropy measures or coefficient of determination.
 * <p>
 * The feature selection process follows this workflow:
 * <ol>
 *   <li>Initialize data structures and parameters for the selected algorithm</li>
 *   <li>Execute the selected search strategy (SFS, SFFS, or exhaustive search)</li>
 *   <li>Calculate entropy values for different feature combinations</li>
 *   <li>Track and rank the best feature sets</li>
 *   <li>Handle ties between equally good feature combinations</li>
 *   <li>Return the optimal feature subset based on entropy minimization</li>
 * </ol>
 * <p>
 * The class supports different criterion functions:
 * <ul>
 *   <li>Mean Conditional Entropy (MCE): Measures the uncertainty in predicting a class given a feature set</li>
 *   <li>Coefficient of Determination (COD): Measures prediction quality based on error metrics</li>
 * </ul>
 * <p>
 * For penalization of non-observed or poorly observed instances, two strategies are available:
 * <ul>
 *   <li>no_obs: Penalty for non-observed instances using alpha parameter</li>
 *   <li>poor_obs: Penalty for poorly observed instances using beta parameter</li>
 * </ul>
 * <p>
 * The Tsallis entropy (q-entropy) is adopted as a generalized entropy measure, with traditional
 * Shannon entropy obtained when q=1 and COD when q=0.
 * <p>
 * Example usage:
 * <pre>
 * char[][] samples = loadDataMatrix();
 * FS featureSelector = new FS(samples, 2, 2, "poor_obs", 0.5f, 0.5f, 1.0f, 10);
 * featureSelector.runSFFS(3, targetIndex, null);
 * Vector selectedFeatures = featureSelector.I;
 * </pre>
 * 
 * @see fs.Criteria For the criterion functions used in feature selection
 * @see fs.Classifier For classification using selected features
 * @see fs.Preprocessing For data preprocessing methods
 * @see agn.AGN For gene network analysis using feature selection
 * @see utilities.MathRoutines For mathematical operations used in feature selection
 */
public class FS implements Serializable {    // Core feature selection data structures
    /** 
     * Selected feature set, represented as a vector of feature indices.
     * This vector contains the indices of the features selected by the algorithm
     * as the optimal feature subset.
     */
    public Vector I;
    
    /**
     * Probability table used for classification based on the selected feature set.
     * Contains the estimated conditional probabilities used to classify instances.
     */
    public Vector probtable;
    
    /**
     * Number of columns (features) in the data matrix.
     * Represents the total number of available features for selection.
     */
    public int columns;
    
    /**
     * Global entropy value for the best feature set found.
     * Lower values indicate better feature sets with more predictive power.
     */
    public float hGlobal;
    
    /**
     * Training data matrix where rows represent samples and columns represent features.
     * The last column typically contains class labels for supervised learning.
     */
    public char[][] A;
    
    // Configuration parameters for entropy calculation
    /**
     * Number of possible values for each feature (typically 2 for binary features).
     * Used to calculate the complexity of the state space and entropy values.
     */
    public int n;
    
    /**
     * Number of possible classes for the target variable.
     * Used for calculating conditional entropy and classification.
     */
    public int c;
    
    /**
     * Type of penalty to apply: "poor_obs" or "no_obs".
     * Controls how to handle missing or underrepresented combinations of feature values.
     */
    public String type;
    
    /**
     * Penalty value for non-observed instances.
     * Used when type="no_obs" to assign probability mass to unseen feature combinations.
     */
    public float alpha;
    
    /**
     * Confidence value for poorly observed instances.
     * Used when type="poor_obs" to adjust probabilities for rare feature combinations.
     */
    public float beta;
    
    /**
     * Tsallis entropy parameter (q=0 for COD, q=1 for Shannon entropy).
     * Controls the form of the entropy formula used for feature evaluation.
     */
    public float q;
    
    /**
     * Maximum number of iterations based on dataset size.
     * Automatically calculated based on dataset characteristics to limit search space.
     */
    public int itmax;
    
    // Result tracking and storage
    /**
     * List of feature sets and their criterion values, sorted by quality.
     * Each entry is a vector containing [entropy_value, feature_set_vector].
     */
    public Vector<Vector> resultlist;
    
    /**
     * Maximum number of results to keep in the result list.
     * Limits storage requirements while preserving the best solutions found.
     */
    public int resultlistsize;
    
    /**
     * Maximum value for the criterion function (used for normalization).
     * Typically set to 1.0 for entropy-based measures.
     */
    public float maxresultvalue;
    
    // Structures for handling different feature set cardinalities
    /**
     * Array storing the best entropy values for each feature set cardinality.
     * Index i contains the entropy for the best feature set of size i.
     */
    public float[] bestentropy;
    
    /**
     * Array storing the best feature sets for each cardinality.
     * Index i contains the vector of features for the best set of size i.
     */
    public Vector[] bestset;
    
    /**
     * Array storing entropy values for tied feature sets at each cardinality.
     * Used when multiple feature sets have the same entropy value.
     */
    public float[] tiesentropy;
    
    /**
     * Array storing tied feature sets for each cardinality.
     * Used to collect all feature sets with equal criterion values for tie-breaking.
     */
    public Vector[] ties;
    
    /**
     * Array storing joint entropies for tied predictor sets.
     * Used for breaking ties between equally good feature sets.
     */
    public float[] jointentropiesties;
    
    /**
     * Timer for measuring execution time of different operations.
     * Used to track performance and optimize critical sections.
     */
    public static Timer timer = new Timer();    /**
     * Constructs a new Feature Selection object with the specified parameters.
     * <p>
     * This constructor initializes the feature selection algorithm with the given
     * training samples and parameters for entropy calculation. It prepares all
     * necessary data structures for feature selection execution and sets up the
     * configuration for the criterion function evaluation.
     * <p>
     * The constructor automatically calculates the maximum number of iterations (itmax)
     * based on the size of the input data matrix to ensure reasonable computational
     * complexity while still exploring the feature space effectively.
     *
     * @param samples Training data matrix where rows represent samples and columns represent features
     * @param npv Number of possible values for each feature (typically 2 for binary features)
     * @param nc Number of possible classes for the target variable
     * @param typeMCE_COD Type of penalty to apply: "poor_obs" or "no_obs"
     * @param alphaPenalty Penalty value for non-observed instances (used when typeMCE_COD="no_obs")
     * @param betaConfidence Confidence value for poorly observed instances (used when typeMCE_COD="poor_obs")
     * @param qentropy Tsallis entropy parameter (q=0 for COD, q=1 for Shannon entropy)
     * @param maxresultlistsize Maximum number of results to keep in the result list
     * 
     * @see #runSFS(boolean, int) For executing Sequential Forward Selection
     * @see #runSFFS(int, int, AGN) For executing Sequential Floating Forward Selection
     * @see #runExhaustive(int, int, Vector) For executing Exhaustive Search
     */
    public FS(char[][] samples, int npv, int nc, String typeMCE_COD,
            float alphaPenalty, float betaConfidence, float qentropy,
            int maxresultlistsize) {
        I = new Vector();
        hGlobal = 1.0f;
        A = samples;
        columns = A[0].length;
        n = npv;
        c = nc;
        type = typeMCE_COD;
        alpha = alphaPenalty;
        beta = betaConfidence;
        q = qentropy;
        resultlistsize = maxresultlistsize;
        maxresultvalue = 1;
        resultlist = new Vector();
        itmax = (int) Math.floor(Math.log(A.length) / Math.log(n));
        probtable = new Vector();
    }
    /**
     * Inserts a feature set and its corresponding criterion value into the result list.
     * <p>
     * This method adds a new feature set to the result list and maintains the list
     * sorted by criterion value (ascending order). The list stores pairs of [criterion_value, feature_set]
     * where lower criterion values represent better feature combinations.
     * <p>
     * If the result list exceeds its maximum capacity (defined by resultlistsize),
     * only the feature sets with the lowest criterion values are retained. This ensures
     * that memory usage remains bounded while preserving the most promising solutions.
     * <p>
     * For each feature set added to the result list, this method:
     * <ol>
     *   <li>Creates a new Vector item with the criterion value at position 0</li>
     *   <li>Copies the feature indices to a new Vector at position 1</li>
     *   <li>Adds the item to the result list or replaces an existing item if better</li>
     *   <li>Maintains the list in sorted order using a selection sort algorithm</li>
     * </ol>
     *
     * @param I Vector of feature indices representing the selected feature set
     * @param hmin Criterion value (entropy or COD) for the feature set, where lower values are better
     * 
     * @see fs.Preprocessing#SelectionSort(Vector) For the sorting algorithm used to maintain order
     * @see #runSFS(boolean, int) For the SFS algorithm that uses this method
     * @see #runSFFS(int, int, AGN) For the SFFS algorithm that uses this method
     * @see #runSFFS_stack(int, int, AGN) For the stack-based SFFS implementation
     * @see #runExhaustive(int, int, Vector) For the exhaustive search algorithm
     * 
     * @throws ClassCastException If the feature indices in Vector I are not Integer objects
     * 
     * @TODO Consider implementing a more efficient data structure like a priority queue
     * to avoid the need for explicit sorting after each insertion
     */
    public void InsertinResultList(Vector I, float hmin) {
        Vector item = new Vector();
        item.add(hmin);//adiciona o valor da funcao criterio na posicao 0
        //adiciona os indices das caracteristicas selecionadas como um vetor
        //na posicao 1.
        Vector features = new Vector();
        for (int i = 0; i < I.size(); i++) {
            int f = (Integer) I.get(i);
            features.add(f);
        }
        item.add(features);
        if (resultlist.size() < resultlistsize) {
            resultlist.add(item);
            //Preprocessing.QuickSort(resultlist, 0, resultlist.size() - 1);
            if (resultlist.size() > 1) {
                Preprocessing.SelectionSort(resultlist);
            }
        } else {
            float vi = (Float) item.get(0);
            float vs = (Float) resultlist.get(resultlistsize - 1).get(0);
            if (vi < vs) {
                resultlist.set(resultlistsize - 1, item);
                //Preprocessing.QuickSort(resultlist, 0, resultlistsize - 1);
                if (resultlist.size() > 1) {
                    Preprocessing.SelectionSort(resultlist);
                }
            }
        }
        /*
        //just for confering
        for (int i = 0; i < resultlist.size(); i++) {
        Vector ri = resultlist.get(i);
        float fsvalue = (Float) ri.get(0);
        IOFile.PrintAndLog("index = " + i + "  FS Value = " + fsvalue + " features = ");
        Vector ritem = (Vector) ri.get(1);
        for (int j = 0; j < ritem.size(); j++) {
        IOFile.PrintAndLog((Integer) ritem.get(j) + "-");
        }
        IOFile.PrintlnAndLog("\n");
        }
        IOFile.PrintlnAndLog("\n\n");
         */
    }
    /**
     * Resolves ties between feature sets with equal criterion values.
     * <p>
     * When multiple feature sets of the same cardinality achieve identical criterion values,
     * this method selects the "best" set by calculating and comparing their joint entropies.
     * The joint entropy measures the total information content of the feature set, with higher
     * values indicating more diverse and informative features.
     * <p>
     * The tie-breaking process involves:
     * <ol>
     *   <li>Checking if any ties exist for the given cardinality</li>
     *   <li>Calculating joint entropy for each tied feature set</li>
     *   <li>Selecting the feature set with the maximum joint entropy</li>
     *   <li>Updating the selected feature set (I) with the winning feature combination</li>
     * </ol>
     * <p>
     * This approach prefers feature sets that contain more diverse information while
     * maintaining the same predictive power according to the primary criterion function.
     *
     * @param i The cardinality (size) of feature sets being evaluated
     * 
     * @see #Minimal(int) For finding minimal entropy feature sets using this method
     * @see #MinimalMA(int) For the Marie-Anne variant that also uses this method
     * @see Criteria#jointentropy(int, Vector, char[][], int) For joint entropy calculation
     * 
     * @throws NullPointerException If ties[i] is null but tiesentropy[i] is not 1
     * 
     * @FIXME The early return when ties[i] is null OR tiesentropy[i] is 1 may hide problems
     * that should be handled more explicitly
     */
    public void BreakTies(int i) {//i == cardinalidade de preditores
        if (ties[i] == null || tiesentropy[i] == 1) {
            return;//algum problema...
        }
        jointentropiesties = new float[ties[i].size()];
        float maxjointentropy = 0;
        int maxjointentropyposition = 0;
        for (int p = 0; p < ties[i].size(); p++) {
            Vector predictors = (Vector) ties[i].get(p);
            jointentropiesties[p] = Criteria.jointentropy(n, predictors, A, c);
            if (jointentropiesties[p] > maxjointentropy) {
                maxjointentropy = jointentropiesties[p];
                maxjointentropyposition = p;
                /*
                //debug
                if (p > 1) {
                IOFile.PrintlnAndLog("houve alteracao!");
                IOFile.PrintlnAndLog("max entropy = " + maxjointentropy);
                IOFile.PrintAndLog("Predictors: ");
                for (int pred = 0; pred < predictors.size(); pred++) {
                IOFile.PrintAndLog(predictors.get(pred) + "  ");
                }
                IOFile.PrintAndLog("\n\n");
                //fim debug
                }
                 */
            }
        }
        I = (Vector) ties[i].get(maxjointentropyposition);
    }
    /**
     * Finds the feature set with minimal entropy across all cardinalities.
     * <p>
     * This method implements the standard feature selection strategy that searches
     * through all stored feature sets of different cardinalities (up to the specified
     * maximum size) and selects the one with the absolute minimum entropy value,
     * regardless of cardinality. This approach favors the most predictive feature
     * set, even if it contains fewer features.
     * <p>
     * The selection process follows these steps:
     * <ol>
     *   <li>Search through best feature sets of all cardinalities (from 1 to maxsetsize)</li>
     *   <li>Select the feature set with the lowest entropy value (most predictive)</li>
     *   <li>If ties exist, use the BreakTies method to select based on maximum joint entropy</li>
     *   <li>Recalculate the criterion function for the final set to update the probability table</li>
     *   <li>Store the result in class variables (I, hGlobal, probtable) for later use</li>
     * </ol>
     * <p>
     * This is the standard feature selection approach used in most information theory
     * based feature selection algorithms, emphasizing the most predictive feature combination
     * regardless of set size.
     *
     * @param maxsetsize Maximum cardinality of feature sets to consider (inclusive)
     * 
     * @see #BreakTies(int) For resolving ties between feature sets with equal criterion values
     * @see #MinimalMA(int) For variant that prefers larger feature sets
     * @see Criteria#MCE_COD(String, float, float, int, int, Vector, char[][], float) For criterion function
     * 
     * @TODO Consider implementing early stopping when a feature set achieves perfect prediction (entropy=0)
     */
    public void Minimal(int maxsetsize) {
        int posminimal = 0;
        //recupera o conjunto de preditores com MENOR cardinalidade e que apresente a menor entropia == resposta da funcao criterio utilizada.
        for (int i = 1; i <= maxsetsize; i++) {
            if (bestentropy[i] < hGlobal) {
                hGlobal = bestentropy[i];
                I = bestset[i];
                posminimal = i;
            }
        }
        if (ties[posminimal] != null && ties[posminimal].size() > 1) {
            BreakTies(posminimal);
        }
        Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, I, A, q);
        float cfvalue = result.entropy;
        //keep current conditional probability table
        probtable = (Vector) result.probtable.clone();
    }
    /**
     * Finds the feature set with minimal entropy and maximum cardinality.
     * <p>
     * This is a variant of the Minimal method specifically designed for the Marie-Anne
     * experiment. It prioritizes feature sets with larger cardinality when multiple
     * sets achieve the same entropy value. The method searches through all feature
     * sets of different cardinalities and selects the largest one with the minimum
     * or equal-to-minimum entropy value.
     * <p>
     * The key difference from the standard Minimal method is in the condition for
     * updating the best feature set. While Minimal uses a strict inequality (H &lt; hGlobal),
     * MinimalMA uses a less-than-or-equal comparison (H &lt;= hGlobal), which means:
     * <ul>
     *   <li>If a larger feature set has the same entropy as a smaller one, the larger set is preferred</li>
     *   <li>The search continues through all cardinalities, always selecting the largest set with minimal entropy</li>
     *   <li>This approach maximizes the number of features while maintaining predictive power</li>
     * </ul>
     * <p>
     * After selecting the feature set, the method resolves any ties using joint entropy
     * and updates the probability table for classification purposes.
     *
     * @param maxsetsize Maximum cardinality of feature sets to consider (inclusive)
     * 
     * @see #Minimal(int) For the standard feature selection approach
     * @see #BreakTies(int) For resolving ties between feature sets with equal criterion values
     * @see #runSFFS_stack(int, int, AGN) For the stack-based SFFS implementation using this method
     * @see agn.MainMarieAnne For the Marie-Anne experiment that uses this method
     * 
     * @FIXME Consider documenting the scientific rationale behind preferring larger feature sets
     * in the Marie-Anne experiment context
     */
    public void MinimalMA(int maxsetsize) {
        int posminimal = 0;
        //recupera o conjunto de preditores com MAIOR cardinalidade e que apresente a menor entropia == resposta da funcao criterio utilizada.
        for (int i = 1; i <= maxsetsize; i++) {
            if (bestentropy[i] <= hGlobal) {
                hGlobal = bestentropy[i];
                I = bestset[i];
                posminimal = i;
            }
        }
        if (ties[posminimal] != null && ties[posminimal].size() > 1) {
            BreakTies(posminimal);
        }
        Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, I, A, q);
        float cfvalue = result.entropy;
        //keep current conditional probability table
        probtable = (Vector) result.probtable.clone();
    }
    /**
     * Initializes the data structures used by the feature selection algorithms.
     * <p>
     * This method prepares the necessary arrays and vectors for storing:
     * <ul>
     *   <li>The best feature set at each cardinality level</li>
     *   <li>The corresponding entropy value for each best feature set</li>
     *   <li>Tied feature sets (those with identical entropy values)</li>
     *   <li>Entropy values for tied feature sets</li>
     * </ul>
     * <p>
     * The method allocates arrays with size maxfeatures, where index 0 is not used
     * (since feature sets start with cardinality 1). All entropy values are initialized
     * to 1.0 (worst possible value), and empty vectors are created for storing feature sets.
     * <p>
     * Note that this initialization is critical for the proper functioning of the SFFS
     * algorithm and other feature selection methods that need to track the best feature
     * sets across different cardinalities.
     *
     * @param maxfeatures Maximum number of features to consider (determines array sizes)
     * 
     * @see #runSFFS(int, int, AGN) For the SFFS algorithm that uses these initialized structures
     * @see #runSFFS_stack(int, int, AGN) For the stack-based SFFS implementation
     * @see #Minimal(int) For selecting the best feature set across all cardinalities
     * @see #MinimalMA(int) For the Marie-Anne variant that uses these structures
     * 
     * @TODO Consider using more memory-efficient data structures for large feature spaces
     */
    public void Inicialize(int maxfeatures) {
        bestentropy = new float[maxfeatures];//a posicao 0 nao eh usada.
        bestset = new Vector[maxfeatures];//a posicao 0 nao eh usada.
        tiesentropy = new float[maxfeatures];//a posicao 0 nao eh usada.
        ties = new Vector[maxfeatures];//a posicao 0 nao eh usada.
        for (int i = 0; i < bestentropy.length; i++) {
            bestentropy[i] = 1;
            tiesentropy[i] = 1;
            bestset[i] = new Vector();
            ties[i] = new Vector();
        }
    }
    /**
     * Runs the Sequential Forward Selection (SFS) algorithm.
     * <p>
     * SFS is a greedy search algorithm that starts with an empty feature set and 
     * iteratively adds the feature that results in the greatest improvement
     * in the criterion function (smallest entropy or highest COD). The algorithm
     * stops when:
     * <ul>
     *   <li>The optimal entropy value of 0 is achieved</li>
     *   <li>No improvement can be obtained by adding more features</li>
     *   <li>The maximum number of features has been reached</li>
     * </ul>
     * <p>
     * The algorithm works by iteratively:
     * <ol>
     *   <li>Evaluating each candidate feature by temporarily adding it to the current feature set</li>
     *   <li>Calculating the entropy/criterion value for each resulting feature combination</li>
     *   <li>Selecting the feature that produces the lowest entropy (best improvement)</li>
     *   <li>Permanently adding this feature to the selected feature set</li>
     *   <li>Stopping if no improvement is found or other termination criteria are met</li>
     * </ol>
     * <p>
     * SFS is computationally efficient but can suffer from the "nesting effect" where
     * once a feature is selected, it cannot be removed even if later combinations would
     * benefit from its exclusion. This limitation is addressed by the SFFS algorithm.
     *
     * @param calledByExhaustive Indicates if this method is called by the exhaustive search method
     *                          (if true, itmax will be set to the size of the final feature set)
     * @param maxfeatures Maximum number of features to include in the selected feature set
     *
     * @see #runSFFS(int, int, AGN) For the SFFS algorithm that addresses SFS limitations
     * @see #runExhaustive(int, int, Vector) For the exhaustive search method
     * @see Criteria#MCE_COD(String, float, float, int, int, Vector, char[][], float) For criterion calculation
     * 
     * @FIXME Consider using a more efficient approach to evaluate candidate features
     * rather than adding/removing from the Vector for each evaluation
     * @TODO Add support for parallelizing feature evaluations to improve performance
     */
    public synchronized void runSFS(boolean calledByExhaustive, int maxfeatures) {
        int collumns = A[0].length;
        for (int i = 0; i < collumns - 1; i++) // for each dimension
        {
            float hMin = 1.1f;
            int fMin = -1;
            float H = 1;
            I.addElement(-1);
            for (int f = 0; f < collumns - 1; f++) // for each feature
            {
                if (I.contains(f)) {
                    continue; // the feature was included in the subspace already
                    // substituting the last element included by a new feature for
                    // evaluation with the subspace included previously
                }
                I.remove(I.lastElement());
                I.addElement(f);
                // calculating the mean conditional entropy for the current
                // subspace

                timer.start("MCE_COD");
                Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, I, A, q);
                H = result.entropy;
                timer.end("MCE_COD");

                //IOFile.PrintlnAndLog(I.size()+" "+H);
                if (H < hMin)// if the new entropy is the smallest of all
                // features previously evaluated with the subspace
                // previously included, sets the new entropy as the
                // lowest and the new feature as canditate to be
                // included in the final subspace
                {
                    fMin = f;
                    hMin = H;

                    //codigo para armazenar os melhores resultados.
                    InsertinResultList(I, H);

                    //}else if (H < hMin && CoD > CoDMin){
                    //    IOFile.PrintlnAndLog("Caso que a entropia nao melhora a tx de acertos.");
                }
                if (H == 0) // if the new entropy is the lowest possible (0), stops
                // and returns the subspace obtained
                {
                    break;
                }
            }
            //if (I.size() <= maxfeatures)
            if (hMin < hGlobal)// if the entropy of the subspace with dimension
            // "dim" is smaller than that of the subspace included before
            // (dim - 1)
            {
                I.remove(I.lastElement()); // remove the last element included...
                I.addElement(fMin); // adds the feature with lowest entropy
                /*
                IOFile.PrintlnAndLog("entropia global de "+hGlobal+" para "+hMin+" usando preditor: ");
                for (int e = 0; e < I.size(); e++)
                if ((Integer) I.get(e) < target)
                IOFile.PrintAndLog((Integer) I.get(e)+" ");
                else
                IOFile.PrintAndLog((((Integer) I.get(e))+1)+" ");
                IOFile.PrintAndLog("\n");
                 */
                hGlobal = hMin; // sets entropy of the new subspace as hMin
                if (hGlobal == 0 || I.size() >= maxfeatures) // if the new subspace has the lowest possible
                // entropy, stops and returns it
                {
                    break;
                }
            } else // if the entropy of the subspace with dimension "dim" is greater
            // than that of the subspace included before (dim - 1)
            {
                I.remove(I.lastElement()); // it is time to stop and return the
                // subspace obtained before (dim - 1)
                break;
            }
        }
        if (calledByExhaustive) // if it is running Exhaustive Search, it is made a call to SFS in order to
        // get the ideal dimension
        {
            itmax = I.size();
        } else {
            /*
            IOFile.PrintlnAndLog(hGlobal);
            for (int i = 0; i < I.size(); i++)
            IOFile.PrintAndLog(I.elementAt(i)+" ");
            IOFile.PrintlnAndLog();
             */
        }
    }
    /**
     * Updates the best feature set if a better one is found.
     * <p>
     * This utility method compares the entropy of a candidate feature set with the current
     * best entropy for that specific cardinality. If the candidate feature set has a lower 
     * entropy value (indicating better predictive power), it replaces the current best set
     * for that cardinality.
     * <p>
     * The method performs the following operations:
     * <ol>
     *   <li>Determines the cardinality (size) of the candidate feature set</li>
     *   <li>Compares its entropy with the best entropy recorded for that cardinality</li>
     *   <li>If the candidate is better, clears the current best feature set</li>
     *   <li>Copies all feature indices from the candidate set to the best set</li>
     *   <li>Updates the bestentropy array with the new minimum entropy value</li>
     * </ol>
     * <p>
     * This method is a critical component for tracking the best feature sets of each
     * cardinality throughout the feature selection process, enabling algorithms to
     * compare sets of different sizes and select the optimal cardinality.
     *
     * @param bestset Vector to store the best feature set for a given cardinality
     * @param bestentropy Array storing the best entropy values for each feature set cardinality
     * @param other Candidate feature set being evaluated
     * @param entropy Entropy value for the candidate feature set
     * 
     * @see #Minimal(int) For selecting the best feature set across all cardinalities
     * @see #MinimalMA(int) For the Marie-Anne variant that uses this method
     * @see #runSFFS(int, int, AGN) Where this method is used to track best sets
     * 
     * @throws IndexOutOfBoundsException If the size of the candidate feature set is larger
     *         than the length of the bestentropy array
     * @throws NullPointerException If any of the input parameters are null
     */
    public void BestSet(Vector bestset, float[] bestentropy, Vector other,
            float entropy) {
        int size = other.size();
        if (entropy < bestentropy[size]) {
            bestentropy[size] = entropy;
            bestset.clear();
            for (int i = 0; i < size; i++) {
                bestset.add(other.elementAt(i));
            }
        }
    }
    /**
     * Runs the Sequential Floating Forward Selection (SFFS) algorithm.
     * <p>
     * SFFS extends the basic SFS algorithm by incorporating a backward step after each forward step to
     * reconsider the utility of features previously selected. This "floating" search allows the algorithm
     * to recover from poor feature selections made in earlier iterations by conditionally removing features
     * that become less useful as the feature set grows.
     * <p>
     * The algorithm works in three main phases:
     * <ol>
     *   <li>Inclusion (Forward): Add the feature that results in the best criterion improvement</li>
     *   <li>Conditional Exclusion (Backward): Find the least significant feature in the current set</li>
     *   <li>Continuation of Conditional Exclusion: Continue removing features if it improves results</li>
     * </ol>
     * <p>
     * SFFS is particularly effective at avoiding local optima that trap simpler greedy algorithms
     * because it can dynamically adjust the search path by reconsidering previous decisions. This
     * flexibility comes at a moderate computational cost increase compared to SFS but is still
     * far more efficient than exhaustive search.
     * <p>
     * The implementation tracks the best feature set for each cardinality, allowing for the selection
     * of the optimal feature subset size based on the criterion function values.
     *
     * @param maxfeatures Maximum number of features to include in the selected feature set
     * @param targetindex Index of the target variable in the data matrix
     * @param agn Reference to an AGN object (can be null), used for additional constraint checking
     *            in gene network applications
     * 
     * @see #Minimal(int) For selecting the optimal feature subset after SFFS completes
     * @see #runSFS(boolean, int) For the simpler Sequential Forward Selection algorithm
     * @see #runSFFS_stack(int, int, AGN) For the stack-based SFFS implementation
     * @see Criteria#MCE_COD(String, float, float, int, int, Vector, char[][], float) For criterion calculation
     * 
     * @FIXME Consider implementing early stopping criteria based on convergence patterns
     * to improve efficiency for large feature spaces
     * @TODO Add support for custom feature evaluation functions to extend beyond entropy-based metrics
     */
    public synchronized void runSFFS(
            int maxfeatures,
            int targetindex,
            AGN agn) {
        if (maxfeatures >= columns) {
            maxfeatures = columns - 1;
        }
        Inicialize(maxfeatures + 1);//Inicializa os atributos usados para armazenar os conjuntos e entropias empatadas.
        while (I.size() < maxfeatures) {
            float hMin = 1;
            int fMin = -1;
            float H = 1;
            I.addElement(-1);
            for (int f = 0; f < columns - 1; f++) // for each feature
            {
                if (agn != null) {
                    int predictorindex = f;
                    if (predictorindex >= targetindex) {
                        predictorindex++;
                    }
                    //nao considera os controles do array.
                    if (agn.getGenes()[predictorindex].isControl()) {
                        continue;
                    }
                }
                if (I.contains(f)) {
                    continue; // the feature was included in the subspace already
                    // substituting the last element included by a new feature for
                    // evaluation with the subspace included previously
                }
                I.remove(I.lastElement());
                I.addElement(f);
                //debug
                //if (targetindex == 4 && (f == 22 || f == 25 || f == 31 || f == 46)) {
                //    IOFile.PrintlnAndLog("debug-predictor");
                //}
                //fim-debug
                Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, I, A, q);
                H = result.entropy;
                //IOFile.PrintlnAndLog(I.size()+" "+H);
                if (H < hMin)// && Math.abs(H - hMin) > 0.01)//if the new entropy is the smallest of all
                // features previously evaluated with the subspace
                // previously included, sets the new entropy as the
                // lowest and the new feature as canditate to be
                // included in the final subspace
                {
                    fMin = f;
                    hMin = H;
                    //codigo para armazenar os melhores resultados.
                    InsertinResultList(I, H);
                    //}else if (H < hMin && CoD > CoDMin){
                    //    IOFile.PrintlnAndLog("Caso que a entropia nao melhora a tx de acertos.");
                }

                //se houve empate ou reducao de entropia
                //os empates armazenam tambem a resposta do algoritmo e nao soh os empates.
                if (Math.abs(H - hMin) < 0.00001) {
                    if (ties[I.size()] == null) {
                        ties[I.size()] = new Vector();                    //para os casos de empate.
                    }
                    if (H < tiesentropy[I.size()]) {
                        //houve reducao de entropia.
                        ties[I.size()].clear();
                        tiesentropy[I.size()] = H;
                    }
                    //adiciona o conjunto empatado no vetor.
                    Vector titem = new Vector();
                    for (int cc = 0; cc < I.size(); cc++) {
                        titem.add(I.get(cc));
                    }
                    ties[I.size()].add(titem);
                }
            }
            if (I.size() <= maxfeatures && fMin != -1) //if (hMin[0] < hGlobal[0]) // if the entropy of the subspace with dimension
            // "dim" is smaller than that of the subspace included before
            // (dim - 1)
            {
                I.remove(I.lastElement()); // remove the last element included...
                I.addElement(fMin); // adds the feature with lowest entropy
                /* saida da execucao no console.
                IOFile.PrintlnAndLog("entropia global de "+hGlobal[0]+" para "+hMin[0]+" usando preditor: ");
                //IOFile.PrintAndLog("entropia local de "+hGlobal[1]+" para "+hMin[1]+" usando preditor: ");
                for (int e = 0; e < I.size(); e++)
                IOFile.PrintAndLog((Integer) I.get(e)+" ");
                IOFile.PrintAndLog("\n");
                 */
                //concidional para armazenar o melhor conjunto de cada tamanho <= maxfeatures.
                if (bestset[I.size()] == null) {
                    bestset[I.size()] = new Vector();
                    for (int s = 0; s < I.size(); s++) {
                        bestset[I.size()].add(I.elementAt(s));
                    }
                    bestentropy[I.size()] = hMin;
                } else {
                    BestSet(bestset[I.size()], bestentropy, I, hMin);
                }
                //heuristica para para execucao, no caso de nao melhorar os resultados.
                int repeticoes = 0;
                for (int be = 1; be < bestentropy.length - 1; be++) {
                    if (bestentropy[be] < 1 && Math.abs(bestentropy[be] - bestentropy[be + 1]) < 0.001) {
                        repeticoes++;
                    }
                }
                if (repeticoes > 1) {
                    break;
                }
                boolean again = true;
                while (I.size() > 2 && again) {
                    //inicio do passo 2 e 3 do sffs: exclusao condicional e
                    //continuacao da exclusao condicional, enquanto os conjuntos
                    //selecionados forem melhores que os seus predecessores
                    int combinations = I.size();
                    float le = 1; //melhor entropia encontrada com as exclusoes
                    int lmf = -1; //caracteristica menos importante
                    for (int comb = 0; comb < combinations; comb++) {
                        Vector xk = new Vector();
                        //monta o vetor com -1 caracteristica e testa todas as combinacoes.
                        for (int nc = 0; nc < combinations; nc++) {
                            if (nc != comb) {
                                xk.add(I.elementAt(nc));
                            }
                        }
                        Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, xk, A, q);
                        float nh = result.entropy;
                        //BestSet(bestset[xk.size()], bestentropy, xk, nh[0]);
                        //armazena a melhor solucao e qual foi a caracteristica excluida.
                        if (nh < le) {
                            le = nh;
                            lmf = (Integer) I.elementAt(comb);
                        }
                    }
                    if (le < bestentropy[I.size() - 1] && lmf != fMin) {
                        I.remove((Integer) lmf);
                        BestSet(bestset[I.size()], bestentropy, I, le);

                        //codigo para armazenar os melhores resultados.
                        InsertinResultList(I, le);

                        again = true;
                        fMin = -1;//a verificacao entre lmf e fMin so vale para
                        //o passo 2, logo eh verificado apenas na primeira passagem.
                        //houve reducao de entropia, atualiza o vetor e empates.
                        ties[I.size()].clear();
                        tiesentropy[I.size()] = le;
                        //adiciona o conjunto empatado no vetor.
                        Vector titem = new Vector();
                        for (int cc = 0; cc < I.size(); cc++) {
                            titem.add(I.get(cc));
                        }
                        ties[I.size()].add(titem);
                    } else {
                        again = false;
                    }
                }
                if (hMin == 0) {//achou o minimo global e seus empates (se houver).
                    break;
                }
            } else // if the entropy of the subspace with dimension "dim" is greater
            // than that of the subspace included before (dim - 1)
            {
                I.remove(I.lastElement()); // it is time to stop and return the
                // subspace obtained before (dim - 1)
                break;
            }
        }
        Minimal(maxfeatures);//returns the feature set with minimal result.
    }
    /**
     * Runs the SFFS algorithm with a stack-based approach for handling tied feature sets.
     * <p>
     * This enhanced version of SFFS (Sequential Floating Forward Selection) uses a stack
     * to explore all promising feature sets that have tied criterion values. Unlike the
     * standard SFFS that only follows a single search path, this approach maintains
     * multiple candidate feature sets and explores them systematically, providing a more
     * thorough search of the feature space.
     * <p>
     * The stack-based approach offers several advantages:
     * <ul>
     *   <li>More comprehensive exploration of the feature space</li>
     *   <li>Better handling of situations where multiple feature combinations have similar performance</li>
     *   <li>Increased robustness against getting trapped in local optima</li>
     *   <li>Higher likelihood of finding the globally optimal feature subset</li>
     * </ul>
     * <p>
     * This method is particularly valuable for network inference problems in gene regulatory
     * networks where multiple regulator combinations may have equal predictive power for
     * a target gene. It maintains an expanded stack of all evaluated feature sets to avoid
     * redundant evaluations.
     * <p>
     * The implementation uses the Marie-Anne criterion (MinimalMA) rather than the standard
     * Minimal criterion, which prefers larger feature sets when criterion values are equal.
     *
     * @param maxfeatures Maximum number of features to include in the selected feature set
     * @param targetindex Index of the target variable in the data matrix
     * @param agn Reference to an AGN object (can be null), used for additional constraint checking
     *            and gene network-specific operations
     * 
     * @see #runSFFS(int, int, AGN) For the standard SFFS implementation
     * @see #ContainPredictorSet(Vector, Vector) For checking if a feature set is already in a list
     * @see #MinimalMA(int) For the Marie-Anne selection criterion that prefers larger feature sets
     * @see agn.AGN For gene network inference using the selected features
     * 
     * @FIXME The current implementation might be memory-intensive for very large feature spaces
     * due to the storage of all expanded feature combinations
     * @TODO Consider implementing a prioritization mechanism for the stack to explore more
     * promising feature sets first
     */
    public synchronized void runSFFS_stack(
            int maxfeatures,
            int targetindex,
            AGN agn) {
        //DEBUG
        IOFile.PrintAndLog("Running Target index == " + targetindex);
        if (agn != null) {
            IOFile.PrintAndLog(", name == " + agn.getGenes()[targetindex].getName());
        }
        IOFile.PrintAndLog("\n");
        //FIM-DEBUG

        if (maxfeatures >= columns) {
            maxfeatures = columns - 1;
        }
        Inicialize(maxfeatures + 1);//Inicializa os atributos usados para armazenar os conjuntos e entropias empatadas.
        //IMPLEMENTAR PILHA PARA TODOS OS PREDITORES EMPATADOS COM TAMANHO DO CONJUNTO == I.size()-1
        //TAMBEM SEREM CONSIDERADOS PELO ALGORITMO SFFS COMO POSSIVEIS RESPOSTAS == CRESCIMENTO DA REDE.
        Vector exestack = new Vector();
        Vector expandedestack = new Vector();

        Vector init = new Vector();
        init.add(-1);
        exestack.add(init);
        while (!exestack.isEmpty()) {
            float hMin = 1;
            int fMin = -1;
            float H = 1;

            //recover a tied predictor set
            I = (Vector) exestack.remove(0);
            //if (!I.equals(init)){
            expandedestack.add(I.clone());
            //DEBUG
            IOFile.PrintAndLog("\nExpanded tied predictors: ");
            IOFile.PrintPredictors(I);
            //FIM-DEBUG
            //}

            for (int f = 0; f < columns - 1; f++) // for each feature
            {
                if (agn != null) {
                    int predictorindex = f;
                    if (predictorindex >= targetindex) {
                        predictorindex++;
                    }
                    //nao considera os controles do array.
                    if (agn.getGenes()[predictorindex].isControl()) {
                        continue;
                    }
                }
                if (I.contains(f)) {
                    continue; // the feature was included in the subspace already
                    // substituting the last element included by a new feature for
                    // evaluation with the subspace included previously
                }
                I.remove(I.lastElement());
                I.addElement(f);
                // calculating the mean conditional entropy for the current
                // subspace

                //DEBUG
                //if (targetindex == 4 && (f == 22 || f == 25 || f == 31 || f == 46)) {
                //    IOFile.PrintlnAndLog("debug-predictor");
                //}
                //FIM-DEBUG
                Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, I, A, q);
                H = result.entropy;
                //IOFile.PrintlnAndLog(I.size()+" "+H);
                if (H < hMin)// && Math.abs(H - hMin) > 0.01)//if the new entropy is the smallest of all
                // features previously evaluated with the subspace
                // previously included, sets the new entropy as the
                // lowest and the new feature as canditate to be
                // included in the final subspace
                {
                    fMin = f;
                    hMin = H;

                    //codigo para armazenar os melhores resultados.
                    InsertinResultList(I, H);

                    //}else if (H < hMin && CoD > CoDMin){
                    //    IOFile.PrintlnAndLog("Caso que a entropia nao melhora a tx de acertos.");
                }

                //se houve empate ou reducao de entropia
                //os empates armazenam tambem a resposta do algoritmo e nao soh os empates.
                if (Math.abs(H - hMin) < 0.001) {
                    if (H < tiesentropy[I.size()]) {
                        //houve reducao de entropia.
                        ties[I.size()].clear();
                        tiesentropy[I.size()] = H;
                    }
                    if (Math.abs(H - tiesentropy[I.size()]) < 0.001) {
                        //adiciona o conjunto empatado no vetor se ainda nao foi inserida sua combinacao de preditores.
                        Vector titem = new Vector();
                        for (int cc = 0; cc < I.size(); cc++) {
                            titem.add(I.get(cc));
                        }
                        if (!ContainPredictorSet(ties[I.size()], titem)){
                            ties[I.size()].add(titem);
                        }
                    }
                }
            }
            if (I.size() <= maxfeatures && fMin != -1) //if (hMin[0] < hGlobal[0]) // if the entropy of the subspace with dimension
            // "dim" is smaller than that of the subspace included before
            // (dim - 1)
            {
                I.remove(I.lastElement()); // remove the last element included...
                I.addElement(fMin); // adds the feature with lowest entropy
                /* saida da execucao no console. 
                IOFile.PrintlnAndLog("entropia global de "+hGlobal[0]+" para "+hMin[0]+" usando preditor: ");
                //IOFile.PrintAndLog("entropia local de "+hGlobal[1]+" para "+hMin[1]+" usando preditor: ");
                for (int e = 0; e < I.size(); e++)
                IOFile.PrintAndLog((Integer) I.get(e)+" ");
                IOFile.PrintAndLog("\n");
                 */
                //concidional para armazenar o melhor conjunto de cada tamanho <= maxfeatures.
                //if (bestset[I.size()] == null) {
                //    bestset[I.size()] = new Vector();
                //    for (int s = 0; s < I.size(); s++) {
                //        bestset[I.size()].add(I.elementAt(s));
                //    }
                //    bestentropy[I.size()] = hMin;
                //} else {
                //atualiza os melhores resultados obtidos no vetor bestset (separados pelo tamanho do conjunto).
                BestSet(bestset[I.size()], bestentropy, I, hMin);
                //}

                //heuristica para para execucao, no caso de nao melhorar os resultados.
                //int repeticoes = 0;
                //for (int be = 1; be < bestentropy.length - 1; be++) {
                //    if (bestentropy[be] < 1 && Math.abs(bestentropy[be] - bestentropy[be + 1]) < 0.001) {
                //        repeticoes++;
                //    }
                //}
                //RETIRADO PARA EXECUCAO EXPERIMENTO MARIE-ANNE
                //if (repeticoes > 1) {
                //    break;
                //}

                boolean again = true;
                while (I.size() > 2 && again) {
                    //inicio do passo 2 e 3 do sffs: exclusao condicional e 
                    //continuacao da exclusao condicional, enquanto os conjuntos
                    //selecionados forem melhores que os seus predecessores
                    int combinations = I.size();
                    float le = 1; //melhor entropia encontrada com as exclusoes
                    int lmf = -1; //caracteristica menos importante
                    for (int comb = 0; comb < combinations; comb++) {
                        Vector xk = new Vector();
                        //monta o vetor com -1 caracteristica e testa todas as combinacoes.
                        for (int nc = 0; nc < combinations; nc++) {
                            if (nc != comb) {
                                xk.add(I.elementAt(nc));
                            }
                        }
                        Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, xk, A, q);
                        float nh = result.entropy;
                        //BestSet(bestset[xk.size()], bestentropy, xk, nh[0]);
                        //armazena a melhor solucao e qual foi a caracteristica excluida.
                        if (nh < le) {
                            le = nh;
                            lmf = (Integer) I.elementAt(comb);
                        }
                    }
                    if (le < bestentropy[I.size() - 1] && lmf != fMin) {
                        I.remove((Integer) lmf);
                        BestSet(bestset[I.size()], bestentropy, I, le);

                        //codigo para armazenar os melhores resultados.
                        InsertinResultList(I, le);

                        again = true;
                        fMin = -1;//a verificacao entre lmf e fMin so vale para 
                        //o passo 2, logo eh verificado apenas na primeira passagem.
                        //houve reducao de entropia, atualiza o vetor e empates.
                        ties[I.size()].clear();
                        tiesentropy[I.size()] = le;
                        //adiciona o conjunto empatado no vetor.
                        Vector titem = new Vector();
                        for (int cc = 0; cc < I.size(); cc++) {
                            titem.add(I.get(cc));
                        }
                        ties[I.size()].add(titem);
                    } else {
                        again = false;
                    }
                }
                //ATUALIZA A PILHA COM OS EMPATES (t == 0) foi o primeiro grupo de preditores jah expandido pelo SFFS.
                //DEBUG
                IOFile.PrintlnAndLog("Preditores escolhidos com cardinalidade == " + I.size());
                IOFile.PrintAndLog("Preditores escolhidos: ");
                IOFile.PrintPredictors(I);

                //FIM-DEBUG
                IOFile.PrintlnAndLog("Preditores empatados empilhados:");
                int contp = 0;
                for (int t = 0; t < ties[I.size()].size(); t++) {
                    Vector predictorset = (Vector) ((Vector) ties[I.size()].get(t)).clone();
                    predictorset.addElement(-1);
                    if (!ContainPredictorSet(exestack, predictorset)//a combinacao de preditores nao esta empilhada.
                            && !ContainPredictorSet(expandedestack, predictorset)//a combinacao de preditores nao foi expandida.
                            && predictorset.size() <= maxfeatures) {//nao atingiu o limite na cardinalidade do conjunto de preditores.
                        exestack.add(predictorset);
                        IOFile.PrintPredictors(predictorset);
                        contp++;
                    }
                }
                IOFile.PrintlnAndLog("# empilhados == " + contp);
                IOFile.PrintlnAndLog("Tamanho da pilha == " + exestack.size());
                //RETIRADO PARA EXPERIMENTO MARIE-ANNE
                //if (hMin == 0) {//achou o minimo global e seus empates (se houver).
                //    break;
                //}
                //if (hGlobal[0] == 0 || I.size() >= maxfeatures) // if the new subspace has the lowest possible
                // entropy, stops and returns it
                //    break;
            } else // if the entropy of the subspace with dimension "dim" is greater
            // than that of the subspace included before (dim - 1)
            {
                I.remove(I.lastElement()); // it is time to stop and return the
                // subspace obtained before (dim - 1)
                break;
            }
        }
        IOFile.PrintlnAndLog("Numero de conjuntos de preditores expandidos == " + expandedestack.size());
        IOFile.PrintVectorofPredictors(expandedestack);
        MinimalMA(maxfeatures);//returns the feature set with minimal result.
    }
    /**
     * Checks if a given predictor set is contained within a stack of predictor sets.
     * <p>
     * This helper method determines if a specific combination of predictors (feature indices)
     * already exists in a collection of predictor sets. The method is designed to avoid
     * redundant feature set evaluations in algorithms that maintain collections of
     * previously examined feature combinations, particularly in the stack-based SFFS algorithm.
     * <p>
     * The comparison accounts for the exact same set of predictors regardless of their
     * order in the Vector. A predictor set is considered contained in the stack if there
     * exists a set in the stack that contains all the same predictors (no more, no less).
     * <p>
     * The implementation:
     * <ol>
     *   <li>Iterates through each set in the stack</li>
     *   <li>For each stack set, counts how many predictors from the target set are present</li>
     *   <li>If the count equals the size of the target predictor set, returns true</li>
     *   <li>Otherwise, continues checking the remaining sets in the stack</li>
     * </ol>
     *
     * @param stack Vector containing predictor sets (each a Vector of Integer indices) to check against
     * @param predictorset The predictor set to look for (a Vector of Integer indices)
     * @return true if the predictor set is contained in any set in the stack, false otherwise
     * 
     * @see #runSFFS_stack(int, int, AGN) For the main algorithm that uses this method
     * 
     * @throws ClassCastException If the elements in the vectors are not Integer objects
     * @throws NullPointerException If any of the input parameters are null
     * 
     * @FIXME The current implementation has O(n²) complexity where n is the number of sets;
     * consider using a more efficient data structure for large collections
     */
    public boolean ContainPredictorSet(Vector stack, Vector predictorset) {
        for (int i = 0; i < stack.size(); i++) {
            Vector stackset = (Vector) stack.get(i);
            int count = 0;
            for (int j = 0; j < predictorset.size(); j++) {
                int predictor = (Integer) predictorset.get(j);
                if (stackset.contains(predictor)) {
                    count++;
                }
            }
            if (count == predictorset.size()) {
                return true;
            }
        }
        return false;
    }
    /**
     * Runs an exhaustive search over all possible feature combinations.
     * <p>
     * This method implements a recursive approach for exhaustive feature selection,
     * evaluating all possible combinations of features up to a maximum cardinality (itmax).
     * The exhaustive search guarantees finding the globally optimal feature subset
     * according to the criterion function, but at the cost of exponential computational
     * complexity.
     * <p>
     * The search is performed using recursive depth-first exploration of the feature space:
     * <ol>
     *   <li>For itmax=1, it evaluates each individual feature</li>
     *   <li>For itmax>1, it recursively builds all possible combinations of size itmax</li>
     *   <li>Each combination is evaluated using the criterion function</li>
     *   <li>The best solution found is continuously tracked and updated</li>
     * </ol>
     * <p>
     * Due to its computational complexity O(n choose k), where n is the number of features
     * and k is the maximum set size, this algorithm should only be used for small feature
     * spaces (typically n < 30) or when finding the guaranteed optimal solution is critical.
     * For larger feature spaces, the SFFS algorithm offers a more efficient alternative with
     * near-optimal results.
     *
     * @param it Current depth in the recursion (iteration count)
     * @param f Current feature index being considered for inclusion
     * @param tempI Temporary vector of selected features being evaluated
     * 
     * @see #runSFS(boolean, int) For a more efficient greedy search approach
     * @see #runSFFS(int, int, AGN) For a more efficient floating search approach
     * @see #InsertinResultList(Vector, float) For storing promising feature combinations
     * @see Criteria#MCE_COD(String, float, float, int, int, Vector, char[][], float) For criterion calculation
     * 
     * @FIXME The current implementation might lead to stack overflow errors for large feature spaces
     * @TODO Consider implementing a non-recursive version to improve performance and avoid stack limitations
     */
    public void runExhaustive(int it, int f, Vector tempI) {
        if (itmax == 1) {
            for (int i = 0; i < columns - 1; i++) {
                tempI.add(i);
                Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, tempI, A, q);
                float H = result.entropy;
                if (H < hGlobal) {
                    I = new Vector(tempI);
                    hGlobal = H;
                    //codigo para armazenar os melhores resultados.
                    InsertinResultList(tempI, H);
                }
                tempI.remove(tempI.lastElement());
            }
            return;
        }
        tempI.add(f);
        if (it >= itmax - 1) {
            Criteria.EntropyResult result = Criteria.MCE_COD(type, alpha, beta, n, c, tempI, A, q);
            float H = result.entropy;

            if (H < hGlobal) {
                I = new Vector(tempI);
                hGlobal = H;
                //IOFile.PrintlnAndLog(f+" "+hGlobal);
                //codigo para armazenar os melhores resultados.
                InsertinResultList(tempI, H);
            }
            return;
        }
        for (int i = f + 1; i < columns - 1; i++) {
            runExhaustive(it + 1, i, tempI);
            tempI.setSize(tempI.size() - 1);
        }
        if (it == 0 && f < columns - itmax) {
            tempI.removeAllElements();
            runExhaustive(0, f + 1, tempI);
        }
    }
}
