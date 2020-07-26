package datamodel;

public class Parameters implements Cloneable, java.io.Serializable {
	/**
	 * SVM type constants.
	 */
	public static final int C_SVC = 0;
	public static final int NU_SVC = 1;
	public static final int ONE_CLASS = 2;
	public static final int EPSILON_SVR = 3;
	public static final int NU_SVR = 4;

	/**
	 * Kernel constants.
	 */
	public static final int LINEAR = 0;
	public static final int POLY = 1;
	public static final int RBF = 2;
	public static final int SIGMOID = 3;
	public static final int PRECOMPUTED = 4;

	/**
	 * SVM type. Supported by constants.
	 */
	public int svmType;

	/**
	 * Kernel type. Supported by constants.
	 */
	public int kernelType;

	/**
	 * Degree for polynomial.
	 */
	public int degree;

	/**
	 * For polynomial/rbf/sigmoid
	 */
	public double gamma;

	/**
	 * For polynomial/sigmoid
	 */
	public double coef0;

	// These are for training only
	/**
	 * In MB
	 */
	public double cacheSize;

	/**
	 * Stopping criteria
	 */
	public double eps;

	/**
	 * For C_SVC, EPSILON_SVR and NU_SVR
	 */
	public double C;
	
	/**
	 * For C_SVC
	 */
	public int nrWeight;

	/**
	 * For C_SVC
	 */
	public int[] weightLabel;
	
	/**
	 * For C_SVC
	 */
	public double[] weight;

	/**
	 * For NU_SVC, ONE_CLASS, and NU_SVR
	 */
	public double nu;
	
	/**
	 * For EPSILON_SVR
	 */
	public double p; 

	/**
	 * Use the shrinking heuristics
	 */
	public int shrinking;
	
	/**
	 * Do probability estimates
	 */
	public int probability;

	/**
	 *****************
	 * Clone it. 
	 *****************
	 */
	public Object clone() {
		try {
			return super.clone();
		} catch (CloneNotSupportedException e) {
			return null;
		}
	}//Of clone

}//Of class Parameters
