package jnn.functions.parametrized;

import java.io.BufferedReader;
import java.io.PrintStream;

import jnn.features.DenseFeatureMatrix;
import jnn.features.DenseFeatureVector;
import jnn.features.FeatureVector;
import jnn.functions.DenseToDenseTransform;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;

import org.nd4j.linalg.api.ndarray.INDArray;

import util.IOUtils;

public class DenseFullyConnectedLayer extends Layer implements DenseToDenseTransform{
	public int inputDim;
	public int outputDim;
	public DenseFeatureMatrix weights;
	public DenseFeatureVector bias;
	public boolean addNoise = FeatureVector.addNoiseDefault;
	public boolean useBias = true;

	public DenseFullyConnectedLayer(int inputDim, int outputDim) {
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		weights = new DenseFeatureMatrix(inputDim, outputDim);		
		bias = new DenseFeatureVector(outputDim);
		if(inputDim != 0){
			initialize(false, true);
		}
	}

	public void initialize(boolean sigmoid, boolean tanh){
		if(FeatureVector.initializationType == 0){
			initializeUniform();
		}
		else {

			if(sigmoid){
				initializeForLogisticSigmoid();
			}
			else if(tanh){
				initializeForTanhSigmoid();
			}
			else{
				initializeUniform();
			}
		}
	}

	public void initializeForTanhSigmoid(){
		weights.normalizedInitializationHtan(inputDim+1, outputDim);
		bias.normalizedInitializationHtan(inputDim+1, outputDim);		
	}

	public void initializeForLogisticSigmoid(){
		weights.normalizedInitializationSigmoid(inputDim+1, outputDim);
		bias.normalizedInitializationSigmoid(inputDim+1, outputDim);		
	}

	public void initializeUniform(){
		weights.initializeUniform(-0.1,0.1);
		bias.initializeUniform(-0.1,0.1);
	}

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		INDArray W = weights.getWeights();
		INDArray x = input.getOutputRange(inputStart, inputEnd);
		INDArray yBias = bias.getWeights();

		if(mapping.isTrain() && addNoise){
			W = W.add(weights.genGaussianNoise(mapping.getId()));
		}

		INDArray y = x.mmul(W);

		if(useBias){
			if(mapping.isTrain() && addNoise){
				yBias = yBias.add(bias.genGaussianNoise(mapping.getId()));
			}
			y = y.add(yBias);
		}
		output.setOutputRange(outputStart, outputEnd, y);		
	}
	
	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		INDArray WTranspose = weights.getTranspose();
		INDArray x = input.getOutputRange(inputStart, inputEnd);

		INDArray yGrad = output.getErrorRange(outputStart, outputEnd);

		INDArray xGrad = yGrad.mmul(WTranspose);
		INDArray WGrad = x.transpose().mmul(yGrad);
		input.setErrorRange(inputStart, inputEnd, xGrad);

		weights.storeGradients(mapping.getId(), WGrad);
		if(useBias){
			INDArray biasGrad = yGrad.dup();
			bias.storeGradients(mapping.getId(), biasGrad);
		}
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		if(useBias){
			bias.update();
		}
		weights.update();
	}
	
	public void save(PrintStream out){
		out.println(inputDim);
		out.println(outputDim);
		out.println(addNoise);
		out.println(useBias);		
		weights.save(out);
		bias.save(out);
	}
	
	public static DenseFullyConnectedLayer load(BufferedReader in) {
		try {
			int inputDim = Integer.parseInt(in.readLine());
			int outputDim = Integer.parseInt(in.readLine());
			DenseFullyConnectedLayer layer = new DenseFullyConnectedLayer(inputDim, outputDim);
			layer.addNoise = Boolean.parseBoolean(in.readLine());
			layer.useBias = Boolean.parseBoolean(in.readLine());
			layer.weights = DenseFeatureMatrix.load(in);
			layer.bias = DenseFeatureVector.load(in);
			return layer;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public String toString() {
		return "this is a " + inputDim + "x" + outputDim+" fully connected layer" + "\n weights:\n" + weights.getWeights() + "\n bias:\n" + bias.getWeights();
	}

	public static void main(String[] args){
		testSaveLoad();
	}
	
	public static void testSaveLoad(){
		DenseFullyConnectedLayer connectedLayer = new DenseFullyConnectedLayer(3, 5);
		double[][] weights = new double[][]{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}};
		double[] bias = new double[]{1,2,3,4,5};
		connectedLayer.weights.initialize(weights);
		connectedLayer.bias.initialize(bias);
		connectedLayer.save(IOUtils.getPrintStream("/tmp/file"));

		System.err.println("original parameters");
		connectedLayer.save(System.err);

		DenseFullyConnectedLayer loadedLayer = DenseFullyConnectedLayer.load(IOUtils.getReader("/tmp/file"));
		System.err.println("loaded parameters");
		loadedLayer.save(System.err);
	}
}
