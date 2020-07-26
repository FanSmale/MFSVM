/**
 * Model
 */

package datamodel;

public class Model implements java.io.Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5942242627584404086L;

	/**
	 * Parameters
	 */
	public Parameters parameters;

	/**
	 * Number of classes, = 2 in regression/one class svm
	 */
	public int numClasses;

	/**
	 * Total #SV
	 */
	public int numSV;

	/**
	 * SVs (SV[l])
	 */
	public Node[][] SV;

	/**
	 * Coefficients for SVs in decision functions (sv_coef[k-1][l])
	 */
	public double[][] svCoefficients;

	/**
	 * Constants in decision functions (rho[k*(k-1)/2])
	 */
	public double[] rho;

	/**
	 * Pairwise probability information
	 */
	public double[] probabilitiesA;

	/**
	 * Pairwise probability information
	 */
	public double[] probabilitiesB;

	/**
	 * sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to
	 * indicate SVs in the training set
	 */
	public int[] svIndices;

	/**
	 * Label of each class (label[k]). For classification only
	 */
	public int[] label;

	/**
	 * Number of SVs for each class (nSV[k]). nSV[0] + nSV[1] + ... + nSV[k-1] =
	 * l
	 */
	public int[] nSV;
}// Of class Model
