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
/*** This class is responsible to generate a table of conditional        ***/
/*** probabilities for classification based on the training samples and  ***/
/*** the previously selected features.                                   ***/
/*** Also, this class implements functions to classify test samples      ***/
/*** based on the table generated                                        ***/
/***************************************************************************/
package fs;

import java.util.Random;
import java.util.Vector;
import utilities.RadixSort;

/**
 * Implements a classifier based on conditional probability tables.
 * <p>
 * This class is responsible for generating a table of conditional probabilities
 * for classification based on training samples and previously selected features.
 * It also implements functions to classify test samples based on the generated table.
 * <p>
 * The classification process involves:
 * <ol>
 *   <li>Building a probability table from training data</li>
 *   <li>Classifying test samples using the table</li>
 *   <li>Using nearest neighbor approach for unseen instances</li>
 * </ol>
 * 
 * @see fs.Criteria
 * @see utilities.RadixSort
 */
class Classifier {

    /**
     * Table of conditional probabilities used for classification.
     * Each entry contains the probability distribution over classes for a specific feature combination.
     */
    Vector table;
    
    /**
     * Instances that serve as indices for the table entries.
     * Each instance represents a specific combination of feature values.
     */
    Vector instances;
    
    /**
     * Labels corresponding to the classification of test samples.
     * Each entry contains the predicted class for a test sample.
     */
    int[] labels;

    /**
     * Creates a new classifier with empty table and instances collections.
     */
    public Classifier() {
        table = new Vector();
        instances = new Vector();
    }

    /**
     * Checks if an instance at a given line is equal to the previous one.
     * <p>
     * Given training samples (A) sorted by the feature subspace (I) values, 
     * this method checks if the instance given by a certain line number is 
     * equal to the previous one by comparing all feature values.
     *
     * @param line The line number to check
     * @param I The vector of feature indices to consider
     * @param A The training samples matrix
     * @return true if the instance at the given line is equal to the previous one, false otherwise
     */
    private boolean equalInstances(int line, Vector I, char[][] A) {
        for (int i = 0; i < I.size(); i++) {
            if (A[line - 1][(Integer) I.elementAt(i)] !=
                    A[line][(Integer) I.elementAt(i)]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns the index corresponding to the maximum element of an array.
     * <p>
     * This method finds the index of the maximum value in the given array.
     * In case of ties (multiple indices with the maximum value), it randomly
     * selects one of the tied indices.
     *
     * @param v The array to search for the maximum value
     * @return The index of the maximum value, or a randomly chosen index in case of ties
     */
    private int indexMaxValue(double[] v) {
        int indexMax = -1;
        double maximum = Integer.MIN_VALUE;
        for (int i = 0; i < v.length; i++) {
            if (maximum < v[i]) {
                indexMax = i;
                maximum = v[i];
            }
        }
        Vector ties = new Vector();
        //int tie = 0;
        for (int i = 0; i < v.length; i++) {
            if (maximum == v[i]) {
                ties.add(i);
                //tie++;
                //if (tie == 1) {
                //    return -1; // more than one class with maximum value (tie)
                //} else {
                    
                //}
            }
        }
        if (ties.size() > 1){
            //sorteia um dos candidatos empatados...
            Random rn = new Random(System.nanoTime());
            int sorteio = rn.nextInt(ties.size());
            return((Integer)ties.get(sorteio));
        }else
            return indexMax;
    }

    /**
     * Calculates a unique index for a given instance based on its feature values.
     * <p>
     * This method computes a numeric index for a sample based on its feature values
     * in the specified feature subspace. The index serves as a key to look up the
     * sample in the probability table.
     *
     * @param sample The sample data as a character array
     * @param I The vector of feature indices to consider
     * @param n The number of possible values for each feature
     * @return A unique numeric index representing the instance's feature combination
     */
    private double instanceIndex(char[] sample, Vector I, int n) {
        double instance = 0;
        int dim = I.size();
        for (int i = 0; i < I.size(); i++) {
            instance += sample[(Integer) I.elementAt(dim - i - 1)] * Math.pow(n, i);
        }
        return instance;
    }

    /**
     * Adds an instance and its conditional probabilities to the classification table.
     * <p>
     * This method adds the considered instance to the instances vector and
     * its conditional probabilities to the table of conditional probabilities.
     * After adding the data, it resets the probability counters for processing
     * the next instance.
     *
     * @param sample The sample data as a character array
     * @param I The vector of feature indices to consider
     * @param pYdX Array of conditional probabilities
     * @param pX Number of times that the instance occurred
     * @param n Number of possible values for features
     * @param c Number of possible classes
     */
    public void addTableLine(char[] sample, Vector I, int[] pYdX, int pX,
            int n, int c) {
        // obtaining the index value of the considered instance
        double instance = instanceIndex(sample, I, n);

        double[] tableLine = new double[c];

        // reseting the conditional probabilities counter
        for (int k = 0; k < c; k++) {
            tableLine[k] = pYdX[k];
            pYdX[k] = 0;
        }
        pX = 0;                  // reseting instance occurrences counter
        instances.add(instance); // adding the instance
        table.add(tableLine);    // adding the corresponding conditional
    // probabilities of the processed instance
    }

    /**
     * Performs a binary search for instance indices.
     * <p>
     * This method implements an efficient binary search algorithm to find
     * a specific instance index in the sorted instances vector. It's used
     * to quickly locate instances when classifying test samples.
     *
     * @param value The instance index to search for
     * @return The position of the instance in the instances vector, or -1 if not found
     */
    public int binarySearch(double value) {
        int start = 0;
        int end = instances.size() - 1;
        while (start <= end) {
            int v = (start + end) / 2;
            if ((Double) instances.elementAt(v) == value) {
                return v;
            } else if ((Double) instances.elementAt(v) < value) {
                start = v + 1;
            } else {
                end = v - 1;
            }
        }
        return -1;
    }

    /**
     * Converts an instance index to its corresponding feature vector.
     * <p>
     * Given an instance index, the number of possible values for each feature,
     * and the dimension (number of features), this method returns the 
     * corresponding feature vector that represents the instance.
     *
     * @param instanceIndex The numeric index of the instance
     * @param n The number of possible values for each feature
     * @param d The dimension (number of features)
     * @return An array representing the feature vector corresponding to the instance index
     */
    public int[] instanceVector(double instanceIndex, int n, int d) {
        int[] V = new int[d];
        for (int i = d - 1; i >= 0; i--) {
            if (instanceIndex == 0) {
                break;
            }
            V[i] = (int) instanceIndex % n;
            instanceIndex = (double) Math.floor(instanceIndex / n);
        }
        return V;
    }

    /**
     * Calculates the Euclidean distance between two feature vectors.
     * <p>
     * This method computes the standard Euclidean distance between two
     * feature vectors, which is used for nearest neighbor classification.
     *
     * @param v1 The first feature vector
     * @param v2 The second feature vector
     * @return The Euclidean distance between the two vectors
     */
    public double euclideanDistance(int[] v1, int[] v2) {
        double quadraticSum = 0;
        for (int i = 0; i < v1.length; i++) {
            quadraticSum += Math.pow((double) v1[i] - v2[i], 2);
        }
        return Math.sqrt(quadraticSum);
    }

    /**
     * Implements Nearest Neighbors classification for an unseen instance.
     * <p>
     * Given the index of an instance, this method implements Nearest Neighbors (NN)
     * classification by comparing the instance to all observed instances.
     * It sums the conditional probabilities of the classes given all instances with
     * minimum Euclidean distance to the target instance. In case of a tie, it continues
     * analyzing instances with increasing distance until the tie is broken.
     * 
     * @param instanceIndex The index of the instance to classify
     * @param n The number of possible values for each feature
     * @param d The dimension (number of features) 
     * @param c The number of possible classes
     * @return The label with highest conditional probability based on nearest neighbors
     */
    public int nearestNeighbors(double instanceIndex, int n, int d, int c) {

        // getting the feature vector corresponding to the considered instance
        int[] instanceValues = instanceVector(instanceIndex, n, d);
        double[] distances = new double[instances.size()];

        // calculating all distances from the input instance to all observed
        // instances
        for (int i = 0; i < instances.size(); i++) {
            int[] currentInstance = instanceVector((Double) instances.elementAt(i), n, d);
            distances[i] = euclideanDistance(instanceValues, currentInstance);
        }
        double[] pYdX = new double[c];

        // this loop will finish only when the tie is broken
        while (true) {
            // obtaining the minimum distance to the input instance
            double minDist = Double.MAX_VALUE;
            for (int i = 0; i < instances.size(); i++) {
                if (distances[i] < minDist) {
                    minDist = distances[i];
                }
            }

            // summing all conditional probilities of the classes given each
            // instance with minimum distance
            for (int i = 0; i < instances.size(); i++) {
                if (distances[i] == minDist) {
                    double[] temp = (double[]) table.elementAt(i);
                    for (int j = 0; j < c; j++) {
                        pYdX[j] += temp[j];
                    }
                    distances[i] = Double.MAX_VALUE; // conditional probabilities
                // of this instances summed,
                // setting maximum value to its
                // distance
                }
            }
            // obtaining the label with maximum conditional probability
            int indexMax = indexMaxValue((double[]) pYdX);
            if (indexMax > -1) // the tie is broken?
            {
                return indexMax; // if so, returns the winner label
            }            // (otherwise, continues in the loop...)
        }
    }

    /**
     * Constructs the conditional probabilities table from training samples.
     * <p>
     * This method builds a table of conditional probabilities based on the training samples.
     * It first sorts the samples by feature values, then counts occurrences of each class
     * for each unique feature combination, and finally constructs the probability table.
     *
     * @param A Training samples represented as a matrix
     * @param I Vector containing indices of the considered features
     * @param n Number of possible values for each feature
     * @param c Number of possible classes
     */
    public void classifierTable(char[][] A, Vector I, int n, int c) {
        int lines = A.length;
        int pX = 0;
        int[] pYdX = new int[c];
        RadixSort.radixSort(A, I, n); // sorting the samples
        for (int j = 0; j < lines; j++) // for each sample...
        {
            if (j > 0 && !equalInstances(j, I, A)) // next instance?
            // adding the conditional probabilities corresponding to the
            // previous instance and reseting pYdX and pX in order to process
            // the coming instance
            {
                addTableLine(A[j - 1], I, pYdX, pX, n, c);
            }
            // accounting an observation of the given label for through the
            // current instance
            pYdX[A[j][A[j].length - 1]]++;
            pX++;
        }
        // adding the conditional probabilities corresponding to the last
        // instance
        addTableLine(A[lines - 1], I, pYdX, pX, n, c);
    }

    /**
     * Classifies test samples using the constructed probability table.
     * <p>
     * Based on the previously constructed table of conditional probabilities,
     * this method classifies each test sample using either direct lookup (for seen instances)
     * or nearest neighbor classification (for unseen instances). For each test sample,
     * it computes its instance index, searches for it in the probability table,
     * and assigns the most likely class label.
     *
     * @param A Test samples represented as a matrix
     * @param I Vector containing indices of the selected features
     * @param n Number of possible values for each feature
     * @param c Number of possible classes
     * @return Array of instance indices for the test samples
     */
    public double[] classifyTestSamples(char[][] A, Vector I, int n, int c) {
        int lines = A.length;
        labels = new int[lines];
        double[] testInstances = new double[lines];

        for (int i = 0; i < lines; i++) {
            testInstances[i] = instanceIndex(A[i], I, n);
            int index = binarySearch(testInstances[i]);
            // the indexOf function of the class Vector is very
            // inefficient to be used a lot of times (in the case of large
            // number of observed instances)

            if (index == -1) // if the instance didn�t appear in training
            // samples...
            //labels[i] = c; // assigns the label "c" (unknown)
            // gets the most appropriate label by applying Nearest Neighbors (NN)
            {
                labels[i] = nearestNeighbors(testInstances[i], n, I.size(), c);
            } else {
                // if the instance occurred, it assigns the label with major
                // conditional probability (Bayesian)
                labels[i] = indexMaxValue((double[]) table.elementAt(index));

                // but if there was a tie between labels with major conditional
                // probability, it applies Nearest Neighbors in order to break the
                // tie
                if (labels[i] == -1) {
                    labels[i] = nearestNeighbors(testInstances[i], n, I.size(), c);
                }
            }
        }
        return (testInstances);
    }
} 
