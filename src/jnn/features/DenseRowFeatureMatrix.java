package jnn.features;

import java.io.BufferedReader;
import java.io.PrintStream;
import java.util.HashSet;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import util.IOUtils;
import util.SerializeUtils;

public class DenseRowFeatureMatrix {

	int numberOfRows;
	int dim;

	DenseFeatureVector[] rows;
	long[] timePerRow; 
	boolean[] updatedOnce; // do not update missed updates before at least one valid update
	long time = 0;
	long maxTime = Long.MAX_VALUE;
	INDArray zero;

	HashSet<Integer> rowsWithGradients = new HashSet<Integer>();

	public DenseRowFeatureMatrix(int numberOfRows, int dim) {
		System.err.println("initializing row matrix with " + numberOfRows + " rows ");
		this.dim = dim;
		this.numberOfRows = numberOfRows;
		rows = new DenseFeatureVector[numberOfRows];
		for(int i = 0; i < numberOfRows; i++){
			rows[i] = new DenseFeatureVector(dim);
		}
		timePerRow = new long[numberOfRows];
		updatedOnce = new boolean[numberOfRows];
		zero = Nd4j.zeros(dim);
	}

	public void initializeUniform(double min, double max){
		for(int i = 0; i < numberOfRows; i++){
			rows[i].initializeUniform(min, max);
		}
	}

	public void normalizedInitializationHtan(int dim){
		for(int i = 0; i < numberOfRows; i++){
			rows[i].normalizedInitializationHtan(1, dim);
		}
	}

	public void normalizedInitializationSigmoid(int dim){
		for(int i = 0; i < numberOfRows; i++){
			rows[i].normalizedInitializationSigmoid(1, dim);
		}
	}

	public void storeGradient(int row, int processId, INDArray gradient){
		rows[row].storeGradients(processId, gradient);
		rowsWithGradients.add(row);
	}

	public void initialize(double[][] vals){
		for(int i = 0; i < numberOfRows; i++){
			rows[i].initialize(vals[i]);
		}
	}

	public void initializeTranspose(double[][] vals){
		double[][] transpose = new double[vals[0].length][vals.length];
		for(int i = 0; i < vals.length; i++){
			for(int j = 0; j < vals[0].length; j++){
				transpose[j][i] = vals[i][j];
			}
		}
		initialize(transpose);
	}

	public void initialize(int key, double[] vals) {
		rows[key].initialize(vals);
	}

	public void update(){
		for(int key : rowsWithGradients){
			getUpdatedVector(key).update();		
			timePerRow[key]++;
			updatedOnce[key] = true;
		}
		rowsWithGradients.clear();
		time++;
		if(time == maxTime){
			for(int i = 0; i < numberOfRows; i++){
				updateMissedUpdates(i);
				timePerRow[i] = 0;
			}
			time = 0;
		}
	}

	public void update(HashSet<Integer> keysToUpdate){
		for(int key : rowsWithGradients){
			if(keysToUpdate.contains(key)){
				getUpdatedVector(key).update();		
				timePerRow[key]++;
				updatedOnce[key] = true;
			}
		}
		rowsWithGradients.clear();
		time++;
		if(time == maxTime){
			for(int i = 0; i < numberOfRows; i++){
				updateMissedUpdates(i);
				timePerRow[i] = 0;
			}
			time = 0;
		}
	}

	public INDArray getUpdatedWeights(int row){
		return getUpdatedVector(row).getWeights();
	}

	public DenseFeatureVector getUpdatedVector(int row){
		updateMissedUpdates(row);
		return rows[row];
	}

	public INDArray getTranspose(int row){
		return rows[row].getTranspose();
	}

	public INDArray genGaussianNoise(int id){
		return rows[0].genGaussianNoise(id);
	}

	public void setL2Reg(double projectionL2) {
		for(int i = 0; i < numberOfRows; i++){
			rows[i].l2 = projectionL2;
		}
	}

	public void updateMissedUpdates(int row){
		long numberOfMissedUpdates = time - timePerRow[row];
		if(updatedOnce[row]){
			for(long l = 0; l < numberOfMissedUpdates; l++){
				rows[row].update(zero);
			}
		}
		timePerRow[row] = time;
	}

	public int getMax(int row){
		return rows[row].getMax();
	}

	public void save(PrintStream out){
		out.println(numberOfRows);
		out.println(dim);
		SerializeUtils.saveLongArray(timePerRow, out);
		SerializeUtils.saveBooleanArray(updatedOnce, out);
		out.println(time);
		for(int row = 0; row < numberOfRows; row++){
			rows[row].save(out);
		}
	}	

	public static DenseRowFeatureMatrix load(BufferedReader in){
		try{
			int numberOfRows = Integer.parseInt(in.readLine());
			int dim = Integer.parseInt(in.readLine());
			DenseRowFeatureMatrix matrix = new DenseRowFeatureMatrix(numberOfRows, dim);
			matrix.timePerRow = SerializeUtils.loadLongArray(in);
			matrix.updatedOnce = SerializeUtils.loadBooleanArray(in);
			matrix.time = Long.parseLong(in.readLine());
			matrix.zero = Nd4j.zeros(dim);
			for(int row = 0; row < numberOfRows; row++){
				matrix.rows[row] = DenseFeatureVector.load(in);
			}
			return matrix;
		}
		catch(Exception e){
			throw new RuntimeException(e);
		}
	}
	
	public static void main(String[] args){
		DenseRowFeatureMatrix test = new DenseRowFeatureMatrix(10, 5);
		test.initializeUniform(-1, 1);
		test.save(IOUtils.getPrintStream("/tmp/file"));
		
		DenseRowFeatureMatrix loaded = DenseRowFeatureMatrix.load(IOUtils.getReader("/tmp/file"));
	}
}
