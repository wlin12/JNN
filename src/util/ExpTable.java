package util;

import util.CheckUtils;

public class ExpTable {

	public static int SingletonMaxExp = 6;
	public static int SingletonExpSize = 1000;
	public static ExpTable singleton = new ExpTable(SingletonMaxExp, SingletonExpSize);
	public int outOfRangeCount = 0;
	
	int maxExp;
	int maxTableSize;
	double[] table;
	
	public ExpTable(int maxExp, int maxTableSize) {
		this.maxExp = maxExp;
		this.maxTableSize = maxTableSize;
		table = new double[maxTableSize];
		for(int i = 0; i < maxTableSize; i++){
			table[i] = getExpNoMem((i/(double)maxTableSize * 2 -1) * maxExp);
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
	
	public static void main(String[]  args){
		ExpTable t = new ExpTable(6, 1000);
		System.err.println(t.table[999]);
		System.err.println(t.getExp(-5.9));
	}
	
	public static double getExpNoMem(double val){
		double exp = Math.exp(val);
		return exp/(exp+1);
	}
}
