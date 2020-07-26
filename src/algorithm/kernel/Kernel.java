package algorithm.kernel;

import algorithm.QMatrix;
import datamodel.Node;
import datamodel.Parameters;

/**
 * Kernel evaluation the static method k_function is for doing single kernel
 * evaluation the constructor of Kernel prepares to calculate the l*l kernel
 * matrix the member function get_Q is for getting one column from the Q Matrix
 */

public abstract class Kernel extends QMatrix {
	protected Node[][] x;
	protected final double[] x_square;

	// svm_parameter
	protected final int kernel_type;
	protected final int degree;
	protected final double gamma;
	protected final double coef0;

	public abstract float[] get_Q(int column, int len);

	public abstract double[] get_QD();

	public void swap_index(int i, int j) {
		do {
			Node[] tmp = x[i];
			x[i] = x[j];
			x[j] = tmp;
		} while (false);
		if (x_square != null)
			do {
				double tmp = x_square[i];
				x_square[i] = x_square[j];
				x_square[j] = tmp;
			} while (false);
	}

	protected static double powi(double base, int times) {
		double tmp = base, ret = 1.0;

		for (int t = times; t > 0; t /= 2) {
			if (t % 2 == 1)
				ret *= tmp;
			tmp = tmp * tmp;
		}
		return ret;
	}

	double kernel_function(int i, int j) {
		switch (kernel_type) {
		case Parameters.LINEAR:
			return dot(x[i], x[j]);
		case Parameters.POLY:
			return powi(gamma * dot(x[i], x[j]) + coef0, degree);
		case Parameters.RBF:
			return Math.exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
		case Parameters.SIGMOID:
			return Math.tanh(gamma * dot(x[i], x[j]) + coef0);
		case Parameters.PRECOMPUTED:
			return x[i][(int) (x[j][0].value)].value;
		default:
			return 0; // Unreachable
		}
	}

	public Kernel(int l, Node[][] x_, Parameters param) {
		this.kernel_type = param.kernelType;
		this.degree = param.degree;
		this.gamma = param.gamma;
		this.coef0 = param.coef0;

		x = (Node[][]) x_.clone();

		if (kernel_type == Parameters.RBF) {
			x_square = new double[l];
			for (int i = 0; i < l; i++)
				x_square[i] = dot(x[i], x[i]);
		} else
			x_square = null;
	}

	static double dot(Node[] x, Node[] y) {
		double sum = 0;
		int xlen = x.length;
		int ylen = y.length;
		int i = 0;
		int j = 0;
		while (i < xlen && j < ylen) {
			if (x[i].index == y[j].index)
				sum += x[i++].value * y[j++].value;
			else {
				if (x[i].index > y[j].index)
					++j;
				else
					++i;
			}
		}
		return sum;
	}

	public static double k_function(Node[] x, Node[] y, Parameters param) {
		switch (param.kernelType) {
		case Parameters.LINEAR:
			return dot(x, y);
		case Parameters.POLY:
			return powi(param.gamma * dot(x, y) + param.coef0, param.degree);
		case Parameters.RBF: {
			double sum = 0;
			int xlen = x.length;
			int ylen = y.length;
			int i = 0;
			int j = 0;
			while (i < xlen && j < ylen) {
				if (x[i].index == y[j].index) {
					double d = x[i++].value - y[j++].value;
					sum += d * d;
				} else if (x[i].index > y[j].index) {
					sum += y[j].value * y[j].value;
					++j;
				} else {
					sum += x[i].value * x[i].value;
					++i;
				}
			}

			while (i < xlen) {
				sum += x[i].value * x[i].value;
				++i;
			}

			while (j < ylen) {
				sum += y[j].value * y[j].value;
				++j;
			}

			return Math.exp(-param.gamma * sum);
		}
		case Parameters.SIGMOID:
			return Math.tanh(param.gamma * dot(x, y) + param.coef0);
		case Parameters.PRECOMPUTED: // x: test (validation), y: SV
			return x[(int) (y[0].value)].value;
		default:
			return 0; // Unreachable
		}
	}
}
