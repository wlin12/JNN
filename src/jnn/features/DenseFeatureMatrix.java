package jnn.features;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;

import org.apache.commons.math.util.FastMath;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.sampling.Sampling;

import util.INDArrayUtils;
import util.IOUtils;
import util.RandomUtils;

public class DenseFeatureMatrix {
	int inputSize;
	int outputSize;
	INDArray features;
	INDArray featuresT;
	GradientStore gradientStore = new GradientStore();
	double l2 = FeatureVector.l2regularizerLambdaDefault;
	double learningRate = FeatureVector.learningRateDefault;

	//adagrad vars
	boolean useAdagrad = FeatureVector.useAdagradDefault;
	INDArray adagradQuotient;
	double adagradEps = 0.001;
	double adagradMax = 10;

	//gaussian noise
	double noiseVar = FeatureVector.noiseDevDefault;
	double noiseVarSqrt = FastMath.sqrt(noiseVar);;
	HashMap<Integer,INDArray> currentNoise = new HashMap<Integer, INDArray>();

	//momentum vars
	boolean useMomentum = FeatureVector.useMomentumDefault;
	INDArray momentumPrevUpdate;
	double momentum = FeatureVector.momentumDefault;

	//adadelta vars
	boolean useAdadelta = FeatureVector.useAdadeltaDefault;
	INDArray adadeltaRMSGradient;
	INDArray adadeltaRMSUpdate;
	double adadeltaMomentum = FeatureVector.adadeltaMomentumDefault;	
	double adadeltaEps = FeatureVector.adadeltaEpsDefault;

	//commit
	int commitMethod = FeatureVector.commitMethodDefault;

	public DenseFeatureMatrix(int inputSize, int outputSize) {
		if(inputSize == 1){
			throw new RuntimeException("input size = 1: use vector instead");
		}
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		if(useAdagrad){
			adagradQuotient = Nd4j.zeros(inputSize, outputSize);
			adagradQuotient.addi(adagradEps);
		}
		if(useMomentum){
			momentumPrevUpdate = Nd4j.zeros(inputSize, outputSize);			
		}
		if(useAdadelta){
			adadeltaRMSGradient = Nd4j.zeros(inputSize, outputSize);
			adadeltaRMSUpdate = Nd4j.zeros(inputSize, outputSize);
		}
	}

	public void initialize(double[][] vals){
		features = Nd4j.create(vals);
		featuresT = features.transpose();
	}


	public void initializeUniform(double min, double max) {
		double[][] featuresMatrixStub = new double[inputSize][outputSize];
		RandomUtils.initializeRandomMatrix(featuresMatrixStub, min, max, 1);
		features = Nd4j.create(featuresMatrixStub);
		featuresT = features.transpose();		
	}

	public void normalizedInitializationHtan(int fanin, int fanout){
		double max = Math.sqrt(6.0d/(fanout+fanin));
		double[][] featuresMatrixStub = new double[inputSize][outputSize];
		RandomUtils.initializeRandomMatrix(featuresMatrixStub, -max, max, 1);

		features = Nd4j.create(featuresMatrixStub);
		featuresT = features.transpose();
	}

	public void normalizedInitializationSigmoid(int fanin, int fanout){
		double max = 4*Math.sqrt(6.0d/(fanin+fanout));
		double[][] featuresMatrixStub = new double[inputSize][outputSize];
		RandomUtils.initializeRandomMatrix(featuresMatrixStub, -max, max, 1);		
		features = Nd4j.create(featuresMatrixStub);
		featuresT = features.transpose();
	}

	public INDArray getWeights(){
		return features;
	}

	public INDArray getTranspose(){
		return featuresT;
	}

	public void storeGradients(int processId, INDArray gradient){
		gradientStore.addGradient(processId, gradient);		
	}

	public void update(){
		INDArray gradient = null;
		if(commitMethod==0){
			gradient = gradientStore.getGradientAvg();
		}
		else{
			gradient = gradientStore.getGradientSum();			
		}
		if(gradient == null) return;
		INDArray gradientL2 = gradient.sub(features.mul(l2));
		if(useAdagrad){
			getAdagradGradient(gradientL2);			
			features.addi(gradientL2.mul(learningRate));
		}
		else if(useMomentum){
			getMomentumGradient(gradientL2);			
			features.addi(gradientL2.mul(learningRate));
		}
		else if(useAdadelta){
			getAdadeltaGradient(gradientL2);
			features.addi(gradientL2);
		}
		else{
			features.addi(gradientL2.mul(learningRate));			
		}
		capValues(FeatureVector.maxVal);
		featuresT = features.transpose();
		gradientStore.init();
	}

	protected void getAdagradGradient(INDArray gradient){
		adagradQuotient.addi(gradient.mul(gradient));
		for(int i = 0; i < inputSize; i++){
			for(int j = 0; j < outputSize; j++){
				double adagradQ = adagradQuotient.getDouble(i,j);				
				if(adagradMax < adagradQ){
					adagradQuotient.putScalar(new int[]{i,j}, adagradMax);
					adagradQ = adagradMax;
				}
				gradient.putScalar(new int[]{i,j}, gradient.getDouble(i,j) / Math.sqrt(adagradQ));
			}
		}
	}

	protected void getAdadeltaGradient(INDArray gradient){
		adadeltaRMSGradient = adadeltaRMSGradient.mul(adadeltaMomentum).add(gradient.mul(gradient).mul(1-adadeltaMomentum));
		gradient.muli(Transforms.sqrt(adadeltaRMSUpdate.add(adadeltaEps)).div(Transforms.sqrt(adadeltaRMSGradient.add(adadeltaEps))));
		adadeltaRMSUpdate.mul(adadeltaMomentum).add(gradient.mul(gradient).mul(1-adadeltaMomentum));
	}

	protected void getMomentumGradient(INDArray gradient){
		INDArray momemtumUpdate = momentumPrevUpdate.mul(momentum);
		gradient.addi(momemtumUpdate);
		momentumPrevUpdate = gradient.dup();
	}


	public INDArray genGaussianNoise(int id){
		if(!currentNoise.containsKey(id)){
			INDArray zeroMean = Nd4j.zeros(inputSize, outputSize);

			currentNoise.put(id,Sampling.normal(RandomUtils.getRandomGenerator(id), zeroMean, noiseVar));
		}
		else{
			RealDistribution reals = new NormalDistribution(RandomUtils.getRandomGenerator(id),0, noiseVarSqrt,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
			INDArrayUtils.shiftLeft(currentNoise.get(id), inputSize, outputSize,  RandomUtils.getRandomGenerator(id).nextInt(inputSize*outputSize), reals.sample());
		}

		//		currentNoise = Sampling.normal(RandomUtils.getRandomGenerator(id), zeroMean, noiseVar);

		return currentNoise.get(id); 
	}

	public void capValues(double max){
		INDArray linear = features.linearView();
		for(int i = 0; i < linear.size(0); i++){
			linear.putScalar(i, Math.max(-max, Math.min(max,linear.getDouble(i))));
		}
	}

	public void save(PrintStream out){
		out.println(inputSize);
		out.println(outputSize);
		out.println(useAdagrad);
		out.println(adagradEps);
		out.println(adagradMax);
		out.println(noiseVar);
		out.println(useMomentum);
		out.println(momentum);
		out.println(useAdadelta);
		out.println(adadeltaEps);
		out.println(adadeltaMomentum);
		saveMatrix(features,out);
		if(useAdagrad){
			saveMatrix(adagradQuotient,out);
		}
		if(useMomentum){
			saveMatrix(momentumPrevUpdate,out);
		}
		if(useAdadelta){
			saveMatrix(adadeltaRMSGradient,out);
			saveMatrix(adadeltaRMSUpdate,out);
		}
	}

	public void saveMatrix(INDArray matrix, PrintStream out){
		for(int row = 0; row < inputSize; row++){
			for(int col = 0; col < outputSize; col++){
				double val = matrix.getDouble(row, col);
				if(col < outputSize-1){
					out.print(val + " ");
				}
				else{
					out.println(val);					
				}
			}
		}
	}

	public static DenseFeatureMatrix load(BufferedReader in){
		try{
			int inputSize = Integer.parseInt(in.readLine());
			int outputSize = Integer.parseInt(in.readLine());
			DenseFeatureMatrix matrix = new DenseFeatureMatrix(inputSize, outputSize);
			matrix.useAdagrad = Boolean.parseBoolean(in.readLine());
			matrix.adagradEps = Double.parseDouble(in.readLine());
			matrix.adagradMax = Double.parseDouble(in.readLine());
			matrix.noiseVar = Double.parseDouble(in.readLine());
			matrix.useMomentum = Boolean.parseBoolean(in.readLine());
			matrix.momentum = Double.parseDouble(in.readLine());
			matrix.useAdadelta = Boolean.parseBoolean(in.readLine());
			matrix.adadeltaEps = Double.parseDouble(in.readLine());
			matrix.adadeltaMomentum = Double.parseDouble(in.readLine());
			matrix.features = loadMatrix(in, inputSize, outputSize);
			if(matrix.useAdagrad){
				matrix.adagradQuotient = loadMatrix(in, inputSize, outputSize);
			}
			if(matrix.useMomentum){
				matrix.momentumPrevUpdate = loadMatrix(in, inputSize, outputSize);
			}
			if(matrix.useAdadelta){
				matrix.adadeltaRMSGradient = loadMatrix(in, inputSize, outputSize);
				matrix.adadeltaRMSUpdate = loadMatrix(in, inputSize, outputSize);
			}
			matrix.featuresT = matrix.features.transpose();
			return matrix;
		} catch(Exception e){
			throw new RuntimeException(e);
		}
	}

	public static INDArray loadMatrix(BufferedReader in, int inputSize, int outputSize) throws IOException{
		INDArray matrix = Nd4j.create(inputSize, outputSize);
		for(int row = 0; row < inputSize; row++){
			String[] vals = in.readLine().split("\\s+");
			for(int col = 0; col < outputSize; col++){
				matrix.putScalar(new int[]{row, col}, Double.parseDouble(vals[col]));
			}
		}
		return matrix;
	}

	public static void main(String[] args){
		DenseFeatureMatrix matrix = new DenseFeatureMatrix(10, 5);
		matrix.initializeUniform(-0.1, 0.1);
		matrix.save(IOUtils.getPrintStream("/tmp/file"));
		
		DenseFeatureMatrix.load(IOUtils.getReader("/tmp/file")).save(System.err);		
	}
}
