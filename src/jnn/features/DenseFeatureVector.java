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

public class DenseFeatureVector {
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

	//gaussian noise
	double noiseVar = FeatureVector.noiseDevDefault;
	double noiseVarSqrt = FastMath.sqrt(noiseVar);;
	HashMap<Integer,INDArray> currentNoise = new HashMap<Integer, INDArray>();

	//commit
	int commitMethod = FeatureVector.commitMethodDefault;

	public DenseFeatureVector(int outputSize) {
		this.outputSize = outputSize;
		if(useAdagrad){
			adagradQuotient = Nd4j.zeros(outputSize);
			adagradQuotient.addi(adagradEps);
		}
		if(useMomentum){
			momentumPrevUpdate = Nd4j.zeros(outputSize);			
		}
		if(useAdadelta){
			adadeltaRMSGradient = Nd4j.zeros(outputSize);
			adadeltaRMSUpdate = Nd4j.zeros(outputSize);
		}
	}

	public void initialize(double[] vals){
		features = Nd4j.create(vals);
		featuresT = features.transpose();
	}

	public void initializeUniform(double min, double max) {
		double[] featuresMatrixStub = new double[outputSize];
		RandomUtils.initializeRandomArray(featuresMatrixStub, min, max, 1);
		features = Nd4j.create(featuresMatrixStub);
		featuresT = features.transpose();		
	}

	public void normalizedInitializationHtan(int fanin, int fanout){
		double max = Math.sqrt(6.0d/(fanout+fanin));
		double[] featuresMatrixStub = new double[outputSize];
		RandomUtils.initializeRandomArray(featuresMatrixStub, -max, max, 1);
		features = Nd4j.create(featuresMatrixStub);
		featuresT = features.transpose();
	}

	public void normalizedInitializationSigmoid(int fanin, int fanout){
		double max = 4*Math.sqrt(6.0d/(fanin+fanout));
		double[] featuresMatrixStub = new double[outputSize];
		RandomUtils.initializeRandomArray(featuresMatrixStub, -max, max, 1);

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
		update(gradient);
	}

	public void update(INDArray gradient) {
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
		for(int j = 0; j < outputSize; j++){
			double adagradQ = adagradQuotient.getDouble(j);				
			gradient.putScalar(new int[]{j}, gradient.getDouble(j) / Math.sqrt(adagradQ));
		}
	}

	protected void getMomentumGradient(INDArray gradient){
		INDArray momemtumUpdate = momentumPrevUpdate.mul(momentum);
		gradient.addi(momemtumUpdate);
		momentumPrevUpdate = gradient.dup();
	}

	protected void getAdadeltaGradient(INDArray gradient){
		adadeltaRMSGradient = adadeltaRMSGradient.mul(adadeltaMomentum).add(gradient.mul(gradient).mul(1-adadeltaMomentum));
		gradient.muli(Transforms.sqrt(adadeltaRMSUpdate.add(adadeltaEps)).div(Transforms.sqrt(adadeltaRMSGradient.add(adadeltaEps))));
		adadeltaRMSUpdate.mul(adadeltaMomentum).add(gradient.mul(gradient).mul(1-adadeltaMomentum));
	}

	public INDArray genGaussianNoise(int id){
		if(!currentNoise.containsKey(id)){
			INDArray zeroMean = Nd4j.zeros(outputSize);
			currentNoise.put(id,Sampling.normal(RandomUtils.getRandomGenerator(id), zeroMean, noiseVar));
		}
		else{
			RealDistribution reals = new NormalDistribution(RandomUtils.getRandomGenerator(id),0, noiseVarSqrt,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
			INDArrayUtils.shiftLeft(currentNoise.get(id), outputSize, RandomUtils.getRandomGenerator(id).nextInt(outputSize), reals.sample());
		}
		return currentNoise.get(id); 
	}

	public void capValues(double max){
		INDArray linear = features.linearView();
		for(int i = 0; i < linear.size(0); i++){
			linear.putScalar(i, Math.max(-max, Math.min(max,linear.getDouble(i))));
		}
	}

	public void save(PrintStream out){
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
		saveVector(features,out);
		if(useAdagrad){
			saveVector(adagradQuotient,out);
		}
		if(useMomentum){
			saveVector(momentumPrevUpdate,out);
		}
		if(useAdadelta){
			saveVector(adadeltaRMSGradient,out);
			saveVector(adadeltaRMSUpdate,out);
		}
	}

	public void saveVector(INDArray vector, PrintStream out){
		for(int col = 0; col < outputSize; col++){
			double val = vector.getDouble(col);
			if(col < outputSize-1){
				out.print(val + " ");
			}
			else{
				out.println(val);					
			}
		}
	}

	public static DenseFeatureVector load(BufferedReader in){
		try{
			int outputSize = Integer.parseInt(in.readLine());
			DenseFeatureVector vector = new DenseFeatureVector(outputSize);
			vector.useAdagrad = Boolean.parseBoolean(in.readLine());
			vector.adagradEps = Double.parseDouble(in.readLine());
			vector.adagradMax = Double.parseDouble(in.readLine());
			vector.noiseVar = Double.parseDouble(in.readLine());
			vector.useMomentum = Boolean.parseBoolean(in.readLine());
			vector.momentum = Double.parseDouble(in.readLine());
			vector.useAdadelta = Boolean.parseBoolean(in.readLine());
			vector.adadeltaEps = Double.parseDouble(in.readLine());
			vector.adadeltaMomentum = Double.parseDouble(in.readLine());
			vector.features = loadVector(in, outputSize);
			if(vector.useAdagrad){
				vector.adagradQuotient = loadVector(in, outputSize);
			}
			if(vector.useMomentum){
				vector.momentumPrevUpdate = loadVector(in, outputSize);
			}
			if(vector.useAdadelta){
				vector.adadeltaRMSGradient = loadVector(in, outputSize);
				vector.adadeltaRMSUpdate = loadVector(in, outputSize);
			}
			vector.featuresT = vector.features.transpose();
			return vector;
		} catch(Exception e){
			throw new RuntimeException(e);
		}
	}

	public static INDArray loadVector(BufferedReader in, int outputSize) throws IOException{
		INDArray matrix = Nd4j.create(outputSize);
		String[] vals = in.readLine().split("\\s+");
		for(int col = 0; col < outputSize; col++){
			matrix.putScalar(col, Double.parseDouble(vals[col]));
		}
		return matrix;
	}

	public int getMax() {
		int index = 0;
		double max = features.getDouble(0);
		for(int i = 1; i < outputSize; i++){
			double val = features.getDouble(i);
			if(max < val){
				index = i;
				max = val;
			}
		}
		return index;
	}

	public static void main(String[] args){
		DenseFeatureVector matrix = new DenseFeatureVector(10);
		matrix.initializeUniform(-0.1, 0.1);
		matrix.save(IOUtils.getPrintStream("/tmp/file"));		
		
		DenseFeatureVector.load(IOUtils.getReader("/tmp/file")).save(System.err);		
	}
}
