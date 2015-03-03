package jnn.functions.composite;

import jnn.features.DenseFeatureMatrix;
import jnn.features.DenseFeatureVector;
import jnn.features.FeatureVector;
import jnn.functions.DenseArrayToDenseArrayTransform;
import jnn.functions.DenseArrayToDenseTransform;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.neuron.DenseNeuronArray;
import jnn.training.TreeInference;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import util.INDArrayUtils;
import util.PrintUtils;
import util.RandomUtils;

public class ConvolutionLayer extends RNN implements DenseArrayToDenseTransform, DenseArrayToDenseArrayTransform{

	public static final String CONVOLUTION_KEY = "convolution";
	public static final String MAX_POOLING_INDEXES = "maxpoolindexes";
	
	int windowSize;
	
	DenseFeatureMatrix[] convolutionWeights;
	DenseFeatureVector bias;
	
	public ConvolutionLayer(int windowSize, int unitInputDim, int unitOutputDim) {
		super(unitInputDim,unitInputDim,unitOutputDim);
		this.windowSize = windowSize;
		this.inputDim = unitInputDim;
		this.outputDim = unitOutputDim;
		convolutionWeights = new DenseFeatureMatrix[windowSize*2+1];
		for(int i = 0; i < windowSize*2+1; i++){
			convolutionWeights[i] = new DenseFeatureMatrix(unitInputDim, unitOutputDim);
			convolutionWeights[i].initializeUniform(-0.1,0.1);

		}
		bias = new DenseFeatureVector(unitOutputDim);
		bias.initializeUniform(-0.1,0.1);
	}
	
	public INDArray[] convolute(DenseNeuronArray[] input, int start, int end, Mapping map){
		INDArray[] convoluted = new INDArray[input.length];
		for(int i = 0; i < input.length; i++){
			INDArray convolutedI = Nd4j.zeros(outputDim);
			for(int w = -windowSize; w <= windowSize; w++){
				int inputIndex = i + w;
				if(inputIndex >= 0 && inputIndex < input.length){
					convolutedI.addi(input[inputIndex].getOutputRange(start, end).mmul(convolutionWeights[w+windowSize].getWeights()));
				}
			}
			convolutedI.addi(bias.getWeights());
			convoluted[i]=convolutedI;
		}
		return convoluted;
	}
	
	public void convoluteBackwards(DenseNeuronArray[] input, INDArray[] yGrad, int inputStart, int inputEnd, int outputStart, int outputEnd, Mapping map){
		INDArray biasGrad = Nd4j.zeros(outputDim);
		INDArray[] xGrad = new INDArray[input.length];
		INDArray[] WGrad = new INDArray[windowSize*2+1];
		
		for(int w = 0; w < windowSize*2+1;w++){
			WGrad[w] = Nd4j.zeros(inputDim, outputDim);			
		}
		
		for(int i = 0; i < input.length; i++){
			xGrad[i] = Nd4j.zeros(inputDim);;
			for(int w = -windowSize; w <= windowSize; w++){
				int inputIndex = i + w;
				if(inputIndex >= 0 && inputIndex < input.length){
					
					xGrad[i].addi(yGrad[i].mmul(convolutionWeights[w+windowSize].getTranspose()));
					WGrad[w+windowSize].addi(input[inputIndex].getOutputRange(inputStart, inputEnd).transpose().mmul(yGrad[i]));
				}
			}
			biasGrad.addi(yGrad[i]);
			input[i].setErrorRange(inputStart, inputEnd, xGrad[i]);
		}
		bias.storeGradients(map.getId(), biasGrad);
		for(int w = 0; w < windowSize*2+1;w++){
			convolutionWeights[w].storeGradients(map.getId(), WGrad[w]);			
		}
	}	
	
	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDense mapping) {
		INDArray[] convolutions = convolute(input, inputStart, inputEnd, mapping);
		int[] indexes = new int[outputDim];
		INDArray maxOut = INDArrayUtils.maxOut1D(convolutions, indexes);
		mapping.setForwardParam(MAX_POOLING_INDEXES, indexes);
		output.setOutputRange(outputStart, outputEnd, maxOut);
	}
	
	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray[] output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDenseArray mapping) {
		INDArray[] convolutions = convolute(input, inputStart, inputEnd, mapping);
		for(int i = 0; i < input.length; i++){
			output[i].setOutputRange(outputStart, outputEnd, convolutions[i]);
		}
	}
	
	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDense mapping) {
		int[] indexes = (int[]) mapping.getForwardParam(MAX_POOLING_INDEXES);
		INDArray[] yGrads = new INDArray[input.length];
		for(int i = 0; i < input.length; i++){
			yGrads[i] = Nd4j.zeros(outputDim);
		}
		
		INDArray yGrad = output.getErrorRange(outputStart, outputEnd);
		for(int y = 0; y < yGrad.size(0); y++){
			double grad = yGrad.getDouble(y);
			int index = indexes[y];
			yGrads[index].putScalar(y, grad);
		}
		convoluteBackwards(input, yGrads, inputStart, inputEnd, outputStart, outputEnd, mapping);		

	}
	
	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray[] output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDenseArray mapping) {
		INDArray[] yGrads = new INDArray[input.length];
		for(int i = 0; i < input.length; i++){
			yGrads[i] = output[i].getErrorRange(outputStart, outputEnd);
		}

		convoluteBackwards(input, yGrads, inputStart, inputEnd, outputStart, outputEnd, mapping);		
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		bias.update();
		for(int w = 0; w < windowSize*2+1; w++){
			convolutionWeights[w].update();
		}
	}
	
	
	public static void main(String[] args){
		int stateDim = 200;
		int inputDim = 50;
		int instances = 1000;
		double learningRate = 0.1;
		FeatureVector.useAdagradDefault = true;
		FeatureVector.commitMethodDefault = 0;
		ConvolutionLayer recNN = new ConvolutionLayer(5, inputDim, stateDim);

		int len = 5;
		double[][][] input = new double[instances][][];
		double[][] expected = new double[instances][];
		for(int i = 0; i < input.length; i++){
			input[i] = new double[len][];
			for(int j = 0; j < len; j++){
				input[i][j] = new double[inputDim];
				RandomUtils.initializeRandomArray(input[i][j], 0, 1);
			}
			expected[i] = new double[len];
			expected[i] = new double[stateDim];
			RandomUtils.initializeRandomArray(expected[i], 0, 1);				
		}
	
		long startTime = System.currentTimeMillis();
		for(int iteration = 0; iteration < 1000; iteration++){
			long iterationStart = System.currentTimeMillis();
			int i = iteration % instances;
			TreeInference inference = new TreeInference(0);
			DenseNeuronArray[] inputNeurons = new DenseNeuronArray[len];
			DenseNeuronArray states = new DenseNeuronArray(stateDim);
			states.setName("output");
			for(int j = 0; j < len; j++){
				inputNeurons[j] = new DenseNeuronArray(inputDim);
				inference.addNeurons(0, inputNeurons[j]);
			}
			inference.addNeurons(1, states);
			
			inference.addMapping(new OutputMappingDenseArrayToDense(inputNeurons, states, recNN));
			inference.init();
			for(int j = 0; j < len; j++){
				inputNeurons[j].init();
				inputNeurons[j].loadFromArray(input[i][j]);
			}
			inference.forward();
			double error = 0;
			states.computeErrorTan(expected[i]);
			error += states.sqError();
			error /= len;
			inference.backward();			
			inference.commit(learningRate);

			PrintUtils.printDoubleArray("output = ", states.copyAsArray(), false);
			PrintUtils.printDoubleArray("error = ", states.copyErrorAsArray(), false);
			System.err.println("error " + error);
			
			double avgTime = (System.currentTimeMillis() - startTime)/(iteration+1);
			System.err.println("avg time " + avgTime);
			System.err.println("this iteration " + (System.currentTimeMillis() - iterationStart));
		}
	}
}
