package util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.sampling.Sampling;

public class INDArrayUtils {	

	public static void rotateLeft(INDArray array, int dimX, int distance){
		double[] tmp = new double[distance];
		for(int i = 0; i < distance; i++){
			tmp[i] = array.getDouble(i);
		}
		for(int i = 0; i < dimX - distance; i++){
			array.putScalar(i, array.getDouble((i+distance)%dimX));
		}
		for(int i = dimX - distance; i < dimX; i++){
			array.putScalar(i, tmp[(i+distance)%dimX]);
		}
	}
	
	public static void shiftLeft(INDArray array, int dimX, int distance, double in){
		rotateLeft(array, dimX, distance);
		array.putScalar(dimX-1, in);
	}

	public static void rotateLeft(INDArray array, int dimX, int dimY, int distance){
		INDArray linear = array.linearView();
		rotateLeft(linear, dimX*dimY, distance);		
	}
	
	public static void shiftLeft(INDArray array, int dimX, int dimY, int distance,double in){
		INDArray linear = array.linearView();
		shiftLeft(linear, dimX*dimY, distance, in);
	}
	
	public static void capValues(INDArray array, double min, double max){		
		INDArray linear = array.linearView();
		for(int i = 0; i < linear.size(0); i++){
			double val = linear.getDouble(i);
			if(val<min){
				linear.putScalar(i, min);
			}
			else{
				if(val > max){
					linear.putScalar(i, max);
				}
			}
		}
	}
	
	public static INDArray maxOut1D(INDArray[] array, int[] indexes){
		int size = indexes.length;
		INDArray ret = Nd4j.zeros(size);
		for(int i = 0; i < size; i++){
			int maxIndex = -1;
			double maxVal=-Double.MAX_VALUE;
			for(int j = 0; j < array.length; j++){
				double val = array[j].getDouble(i);
				if(maxVal < val){
					maxVal = val;
					maxIndex = j;
				}
			}
			indexes[i] = maxIndex;
			ret.putScalar(i, maxVal);
		}
		return ret;
	}
	
	public static void main(String[] args){
		INDArray zeroMean = Nd4j.zeros(3, 3);
		double var = 0.01;
		System.err.println(zeroMean);
		INDArray randomNumbers = Sampling.normal(RandomUtils.getRandomGenerator(123), zeroMean, var);
		System.err.println(randomNumbers);
		shiftLeft(randomNumbers, 3, 3, 2, 1.0);
		System.err.println(randomNumbers);
		
//		long startRand = System.currentTimeMillis();
//		for(int i = 0; i < 10000; i++){
//			Sampling.normal(RandomUtils.getRandomGenerator(123), zeroMean, var);
//		}
//		System.err.println("took " + (System.currentTimeMillis() - startRand) + " to generate rands"); 
//
//		long startShift = System.currentTimeMillis();
//		for(int i = 0; i < 10000; i++){
//			shiftLeft(randomNumbers, 3, 3, 2, 1.0);
//		}		
//		System.err.println("took " + (System.currentTimeMillis() - startShift) + " to generate rands"); 
	}
}
