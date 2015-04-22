package util;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.activation.SoftMax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import util.CheckUtils;

public class ExpTable {

	public static int SingletonMaxExp = 50;
	public static int SingletonExpSize = 100000;
	public static int maxExpVal = 500;
	public static int SingletonIntExpSize = 100000;
	public static ExpTable singleton = new ExpTable(SingletonMaxExp, SingletonExpSize);
	public int outOfRangeCount = 0;
	
	int maxExp;
	int maxTableSize;
	double[] logTable;
	double[] expTable;
	double[] intExpTable;
	
	public static class OutOfBoundariesException extends Exception{
		
	}
	
	public ExpTable(int maxExp, int maxTableSize) {
		this.maxExp = maxExp;
		this.maxTableSize = maxTableSize;
		logTable = new double[maxTableSize];
		expTable = new double[maxTableSize];
		for(int i = 0; i < maxTableSize; i++){
			logTable[i] = getExpNoMem((i/(double)maxTableSize * 2 -1) * maxExp);
			expTable[i] = FastMath.exp((i/(double)maxTableSize * 2 -1) * maxExp);
		}
		intExpTable = new double[SingletonIntExpSize];
		for(int i = 0; i < SingletonIntExpSize; i++){
			intExpTable[i] = FastMath.exp(i);
		}
	}	

	public double getExp(double val){
		/*if(val >= maxExp){
			outOfRangeCount++;
			return getExpNoMem(val);
		}
		if(val <= -maxExp){
			outOfRangeCount++;
			return getExpNoMem(val);
		}
		return table[(int)((val + maxExp) * (maxTableSize / maxExp / 2))];*/
		return getExpNoMem(val);
	}
	
	public int getIndex(double val){
		return (int)((val + maxExp) * (maxTableSize / maxExp / 2));		
	}
	
	public double getExpD(double val){
		double exp = getExp(val);
		return (1-exp)*exp;
	}
	
	public void getExp(double[] input, double[] output){
		CheckUtils.checkEQ(input.length, output.length);
		for(int i = 0; i < input.length; i++){
			output[i] = getExp(input[i]);			
		}		
	}
	
	public void getExpD(double[] diff, double[] output, double[] diffOutput){
		CheckUtils.checkEQ(diff.length, output.length);
		CheckUtils.checkEQ(diff.length, diffOutput.length);
		for(int i = 0; i < diff.length; i++){
			diffOutput[i] = getExpD(output[i])*diff[i];
		}
	}
	
	public void getExp(double[][] input, double[][] output){
		CheckUtils.checkMXSize(input, output);		
		for(int i = 0; i < input.length; i++){
			for(int j = 0; j < input[i].length; j++){
				output[i][j] = getExp(input[i][j]);
			}
		}		
	}
	
	public void getExpD(double[][] diff, double[][] output, double[][] diffOutput){
		CheckUtils.checkMXSize(diff, output);
		CheckUtils.checkMXSize(diff, diffOutput);
		for(int i = 0; i < diff.length; i++){
			for(int j = 0; j < diff[i].length; j++){
				diffOutput[i][j] = getExpD(output[i][j])*diff[i][j];
			}
		}
	}
	
	public static double getExpSing(double val){
		return singleton.getExp(val);
	}
	
	public static double getExpDSing(double val){
		return singleton.getExpD(val);
	}
	
	public static double getExpNoMem(double val){
		if(val > 1000){
			return 1;
		}
		if(val < -1000){
			return -1;
		}
		double exp = Math.exp(val);
		return exp/(exp+1);
	}
	
	public static INDArray getExpTable(INDArray input) throws OutOfBoundariesException{
		INDArray output = Nd4j.zeros(input.shape());
		for(int i = 0 ; i < input.size(0); i++){
			int index = singleton.getIndex(input.getDouble(i));
			if(index >= singleton.expTable.length || index < 0) throw new OutOfBoundariesException(); 
			output.putScalar(i, singleton.expTable[index]);			
		}
		return output;
	}
	
	public static INDArray getExpNormTable(INDArray input){
		INDArray output = Nd4j.zeros(input.shape());
		double max = -Integer.MAX_VALUE;
		double avg = 0;
		for(int i = 0 ; i < input.size(0); i++){
			double val = input.getDouble(i);
			avg += val;
			if(val > max){
				max = val;
			}
		}
		avg/=input.size(0);
		if(max >= maxExpVal){
			avg = max - maxExpVal + 1;
		}
		
		double norm = 0;		
		for(int i = 0 ; i < input.size(0); i++){
			double val = input.getDouble(i)-avg;
			
			double modded = val % SingletonMaxExp;
			int mod = (int)(val / SingletonMaxExp);			

			double exp = singleton.expTable[singleton.getIndex(modded)];
			if(mod < 0){
				exp /= singleton.intExpTable[-mod*SingletonMaxExp];
			}
			if(mod > 0){
				exp *= singleton.intExpTable[mod*SingletonMaxExp];
			}

			output.putScalar(i, exp);
			norm+=exp;
		}
		output.divi(norm);
		return output;
	}
	
	public static void main(String[] args) throws OutOfBoundariesException{
		long start = System.currentTimeMillis();
		INDArray input = Nd4j.zeros(100);
		for(int i = 0; i < 100; i++){
			input.putScalar(i, i*2-100);
		}
		SoftMax softmaxExp = new SoftMax();	
		System.err.println(softmaxExp.apply(input));
		System.err.println(getExpNormTable(input));

		for(int i = 0; i < 1000000; i++){
			softmaxExp.apply(input);
		}
		long time = System.currentTimeMillis() - start;
		System.err.println("exp regular " + time);
		start = System.currentTimeMillis();
		for(int i = 0; i < 1000000; i++){
			getExpNormTable(input);
		}
		time = System.currentTimeMillis() - start;
		System.err.println("exp table " + time);
	}	
}
