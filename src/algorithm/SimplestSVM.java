package algorithm;

import java.io.FileReader;
import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;

/**
 * The Naive Bayes algorithm..
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The Bayes project.
 * <p>
 * Progress: The very beginning.<br>
 * Written time: April 4, 2020. <br>
 * Last modify time: April 4, 2020.
 */

public class SimplestSVM {

	/**
	 * The data.
	 */
	Instances data;

	/**
	 * The number of classes. For binary classification it is 2.
	 */
	int numClasses;

	/**
	 * The number of instances.
	 */
	int numInstances;

	/**
	 * The number of conditional attributes.
	 */
	int numConditions;

	/**
	 * The prediction, including queried and predicted labels.
	 */
	int[] predicts;

	/**
	 * The weight vector. The last element is b.
	 */
	double[] w;

	/**
	 * The step length?
	 */
	double lambda = 0.0001;

	/**
	 * ??
	 */
	double lr = 0.00001;

	/**
	 * Threshold for cost.
	 */
	double threshold = 0.001;

	/**
	 * The cost?
	 */
	double cost;

	/**
	 * The gradient?
	 */
	double[] gradient;

	/**
	 * What is yp?
	 */
	double[] yp;

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The given file.
	 ********************
	 */
	public SimplestSVM(String paraFilename) {
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try

		data.setClassIndex(data.numAttributes() - 1);

		numConditions = data.numAttributes() - 1;
		numInstances = data.numInstances();
		numClasses = data.attribute(numConditions).numValues();

		// Change class into binary: -1 and 1.
		for (int i = 0; i < numInstances; i++) {
			if (data.instance(i).classValue() < 1e-6) {
				data.instance(i).setClassValue(-1);
			} else {
				data.instance(i).setClassValue(1);
			} // Of if
		} // Of for i

		w = new double[numConditions + 1];
		gradient = new double[numConditions + 1];
		cost = 0;
		yp = new double[numInstances];
	}// Of the constructor

	/**
	 ********************
	 * Compute cost and gradient.
	 * See https://zhuanlan.zhihu.com/p/31886934.
	 ********************
	 */
	public void computeCostGradient() {
		cost = 0;
		double tempActualLabel, tempValue;

		//Step 1. Compute cost.
		for (int i = 0; i < numInstances; i++) {
			yp[i] = innerProduct(data.instance(i), w);
			tempActualLabel = data.instance(i).classValue();

			//xi_i = \max(0, 1 - y_i(w x_i + b))
			tempValue = 1 - tempActualLabel * yp[i];
			if (tempValue > 0) {
				//Less than the margin.
				cost += tempValue;
			} // Of if
		} // Of for i

		for (int i = 0; i < numConditions + 1; i++) {
			cost += 0.5 * lambda * w[i] * w[i];
		} // Of for i

		//Step 2. Compute gradient
		// For conditions.
		for (int i = 0; i < numConditions + 1; i++) {
			gradient[i] = Math.abs(lambda * w[i]);

			for (int j = 0; j < numInstances; j++) {
				tempActualLabel = data.instance(j).classValue();
				tempValue = 1 - tempActualLabel * yp[j];
				if (tempValue > 0) {
					gradient[i] -= tempActualLabel * data.instance(j).value(i);
				} // Of if
			} // Of for j
		} // Of for i

		// For offset b.
		gradient[numConditions] = Math.abs(lambda * w[numConditions]);
		for (int j = 0; j < numInstances; j++) {
			tempActualLabel = data.instance(j).classValue();

			tempValue = 1 - tempActualLabel * yp[j];
			if (tempValue > 0) {
				gradient[numConditions] -= tempActualLabel
						* data.instance(j).value(numConditions);
			} // Of if
		} // Of for j
	}// Of computeCostGradient

	/**
	 ********************
	 * Update.
	 ********************
	 */
	public void update() {
		for (int i = 0; i < numConditions + 1; i++) {
			w[i] -= lr * gradient[i];
		} // Of for i
	}// Of update

	/**
	 ********************
	 * Train. Compute w.
	 * 
	 * @param paraLambda
	 *            The lambda value.
	 ********************
	 */
	public void train(double paraLambda, int paraRounds) {
		lambda = paraLambda;

		for (int i = 0; i < paraRounds; i++) {
			computeCostGradient();
			System.out.println("Cost = " + cost + ", w = " + Arrays.toString(w));

			if (cost < threshold) {
				break;
			} // Of if

			update();
		} // Of for i
	}// Of train

	/**
	 ********************
	 * Classify all instances, the results are stored in predicts[].
	 ********************
	 */
	public void classify() {
		predicts = new int[numInstances];
		for (int i = 0; i < numInstances; i++) {
			predicts[i] = classify(data.instance(i));
		} // Of for i

		System.out.println("Predicts = " + Arrays.toString(predicts));
	}// Of classify

	/**
	 ********************
	 * Classify an instances.
	 ********************
	 */
	public int classify(Instance paraInstance) {
		double tempResult = innerProduct(paraInstance, w);

		System.out.println("y' = " + tempResult);
		if (tempResult < 0) {
			return -1;
		} // Of if

		return 1;
	}// Of classify

	/**
	 ********************
	 * Compute inner product.
	 * 
	 * @param paraArray1
	 *            The first vector.
	 * @param paraArray2
	 *            The second vector. These vectors should have the same length.
	 *            For simplicity they are not checked here.
	 * @return The inner product.
	 ********************
	 */
	public double innerProduct(double[] paraVector1, double[] paraVector2) {
		double resultValue = 0;

		for (int i = 0; i < paraVector1.length; i++) {
			resultValue += paraVector1[i] * paraVector2[i];
		} // Of for i

		return resultValue;
	}// Of innerProduct

	/**
	 ********************
	 * Compute inner product.
	 * 
	 * @param paraInstance
	 *            The given instance vector.
	 * @param paraArray2
	 *            The second vector. These vectors should have the same length.
	 *            For simplicity they are not checked here.
	 * @return The inner product.
	 ********************
	 */
	public double innerProduct(Instance paraInstance, double[] paraVector2) {
		double[] tempArray = new double[numConditions + 1];
		for (int i = 0; i < tempArray.length; i++) {
			tempArray[i] = paraInstance.value(i);
		} // Of for i

		// Handle b.
		tempArray[numConditions] = 1;

		return innerProduct(tempArray, paraVector2);
	}// Of innerProduct

	/**
	 ********************
	 * Compute accuracy.
	 ********************
	 */
	public double computeAccuracy() {
		double tempCorrect = 0;
		for (int i = 0; i < numInstances; i++) {
			if (predicts[i] == (int) data.instance(i).classValue()) {
				tempCorrect++;
			} // Of if
		} // Of for i

		double resultAccuracy = tempCorrect / numInstances;
		return resultAccuracy;
	}// Of computeAccuracy

	/**
	 ************************* 
	 * Test numerical data.
	 ************************* 
	 */
	public static void testSVM() {
		System.out.println("Hello, SVM. Now test linear separation.");
		String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/r15.arff";
		// String tempFilename = "src/data/banana.arff";
		// String tempFilename = "src/data/wdbc_norm_ex.arff";

		SimplestSVM tempLearner = new SimplestSVM(tempFilename);
		tempLearner.train(0.0001, 100000);

		tempLearner.classify();

		System.out.println("The accuracy is: " + tempLearner.computeAccuracy());
	}// Of testSVM

	/**
	 ************************* 
	 * Test this class.
	 * 
	 * @author Fan Min
	 * @param args
	 *            The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		testSVM();
		// testNumerical();
		// System.out.println("Hello, SVM!");
	}// Of main
}// Of class SimpleSVM