package util;

public class TanFuncs {
	public static double sigmoid(double x){
		//return 1.7159*tanh(0.66666667*x);
		return tanh(x);
	}

	
	public static double dsigmoid(double x){
		double tanhx = tanh(x);
		//return 0.66666667/1.7159*(1.7159+(x))*(1.7159-(x));
		return 1 - tanhx*tanhx;
	}
	
	private static double tanh(double x) {
//		double a = (((x*x + 378) * x*x + 17325)*x*x + 135135)*x;
//		double b = ((28*x*x + 3150)*x*x + 62370)*x*x + 135135;
//		double ret = a/b;
//		if(ret < -1 ) return -1;
//		if(ret > 1) return 1;
//		return ret;
		return Math.tanh(x);
	}
	
	public static void main(String[] args){
		System.err.println(Math.tanh(-10));
		System.err.println(tanh(-10));
	}

	public static void sigmoid(double[] transformedCharEmbeddings) {
		for(int i = 0; i < transformedCharEmbeddings.length; i++){
			transformedCharEmbeddings[i] = sigmoid(transformedCharEmbeddings[i]);
		}		
	}
	
	

	public static void sigmoid(double[][] matrix) {
		for(int i = 0; i < matrix.length; i++){
			for(int j = 0; j < matrix[i].length; j++){
				matrix[i][j]=sigmoid(matrix[i][j]);
			}
		}

	}
	
	public static void dsigmoid(double[] error, double[] output) {
		for(int i = 0; i < error.length; i++){
			error[i] = TanFuncs.dsigmoid(output[i]) * error[i];
		}
	}


	public static double sigmoidHard(double neuron) {
		if(neuron > 1){
			return 1;
		}
		if(neuron < -1){
			return -1;
		}
		return neuron;
	}


	public static double dsigmoidHard(double neuron, double gradient) {		
		if(neuron > 1 && gradient > 0){
			return 0;
		}
		if(neuron < -1 && gradient < 0){
			return 0;
		}
		return 1;
	}

}
