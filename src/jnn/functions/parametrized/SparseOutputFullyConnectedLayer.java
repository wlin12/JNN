package jnn.functions.parametrized;

import java.util.Map.Entry;

import jnn.features.DenseRowFeatureMatrix;
import jnn.features.FeatureVector;
import jnn.functions.DenseToSparseTransform;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingDenseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import jnn.training.TreeInference;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SparseOutputFullyConnectedLayer extends Layer implements DenseToSparseTransform{

	int inputDim;
	int outputDim;

	DenseRowFeatureMatrix weights;
	DenseRowFeatureMatrix bias;

	boolean useBias = true;
	
	public SparseOutputFullyConnectedLayer(int inputDim, int outputDim) {
		super();
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		weights = new DenseRowFeatureMatrix(outputDim, inputDim);
		bias = new DenseRowFeatureMatrix(outputDim, 1);
		initialize(false, true);
	}

	public void initialize(double[][] vals){
		weights.initializeTranspose(vals);
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
		weights.normalizedInitializationHtan(outputDim*inputDim);
		bias.normalizedInitializationHtan(outputDim*inputDim);
	}

	public void initializeForLogisticSigmoid(){
		weights.normalizedInitializationSigmoid(outputDim*inputDim);
		bias.normalizedInitializationSigmoid(outputDim*inputDim);
	}

	public void initializeUniform(){
		weights.initializeUniform(-0.1,0.1);
		bias.initializeUniform(-0.1, 0.1);
	}

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			SparseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseToSparse mapping) {
		INDArray x = input.getOutputRange(inputStart, inputEnd);
		for(Entry<Integer, Double> entry : output.getNonZeroEntries()){
			double val = x.mmul(weights.getTranspose(entry.getKey())).getDouble(0);
			if(useBias){
				val += bias.getUpdatedWeights(entry.getKey()).getDouble(0);
			}
			entry.setValue(val);
		}
	}

	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			SparseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseToSparse mapping) {
		INDArray x = input.getOutputRange(inputStart, inputEnd);
		for(Entry<Integer, Double> entry : output.getNonZeroEntries()){
			double gradient = output.getError(entry.getKey());
			INDArray wGrad = x.mul(gradient);
			weights.storeGradient(entry.getKey(), mapping.getId(), wGrad);

			INDArray xGrad = weights.getUpdatedWeights(entry.getKey()).mul(gradient);
			input.setErrorRange(inputStart, inputEnd, xGrad);

			if(useBias){
				INDArray biasGrad = Nd4j.zeros(1);
				biasGrad.putScalar(0, gradient);
				bias.storeGradient(entry.getKey(), mapping.getId(), biasGrad);
			}
		}
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		bias.update();
		weights.update();
	}

	@Override
	public String toString() {
		String ret = "This is a sparse output layer with dim " + inputDim + "x" + outputDim;
		return ret;
	}

	public static void main(String[] args){
		test2();
	}

	//sparse output
	public static void test(){
		TreeInference inference = new TreeInference(0);

		SparseOutputFullyConnectedLayer connectedLayer = new SparseOutputFullyConnectedLayer(3, 5);
		DenseFullyConnectedLayer connectedLayer2 = new DenseFullyConnectedLayer(3, 5);

		double[][] weights = new double[][]{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}};
		double[] bias = new double[]{1,2,3,4,5};
		connectedLayer2.weights.initialize(weights);
		connectedLayer2.bias.initialize(bias);
		connectedLayer.initialize(weights);
		connectedLayer.bias.initializeTranspose(new double[][]{bias});

		long timeStart = System.currentTimeMillis();
		for(int i = 0; i < 10; i++){
			DenseNeuronArray input = new DenseNeuronArray(3);
			input.init();
			input.addNeuron(0, 1);
			input.addNeuron(1, 2);
			input.addNeuron(2, 3);

			DenseNeuronArray input2 = new DenseNeuronArray(3);
			input2.init();
			input2.addNeuron(0, 1);
			input2.addNeuron(1, 2);
			input2.addNeuron(2, 3);

			inference.addNeurons(0, input);
			inference.addNeurons(0, input2);
			SparseNeuronArray output = new SparseNeuronArray(5);
			output.init();
			output.addNeuron(2);
			output.addNeuron(4);
			inference.addNeurons(1, output);

			DenseNeuronArray output2 = new DenseNeuronArray(5);		
			output2.init();
			inference.addNeurons(1, output2);

			Mapping map = new OutputMappingDenseToSparse(input, output, connectedLayer);
			map.setParentInference(inference);		
			inference.addMapping(map);

			Mapping map2 = new OutputMappingDenseToDense(input2, output2, connectedLayer2);
			map2.setParentInference(inference);		
			inference.addMapping(map2);

			inference.init();
			inference.forward();

			output.addError(2, 1);
			output.addError(4, 2);
			output2.addError(2, 1);
			output2.addError(4, 2);

			inference.backward();		
			inference.commit(0.1);
			System.err.println("input 1 " + input);
			System.err.println("input 2 " + input2);
			System.err.println("output 1 " + output);
			System.err.println("output 2 " + output2);

		}
		System.err.println("experiment took " + (System.currentTimeMillis() - timeStart) + " millis");
	}
	
	//check it does not take more time for more outputs
	public static void test2(){
		TreeInference inference = new TreeInference(0);

		SparseOutputFullyConnectedLayer connectedLayer = new SparseOutputFullyConnectedLayer(3, 500000);

		long timeStart = System.currentTimeMillis();
		for(int i = 0; i < 10; i++){
			DenseNeuronArray input = new DenseNeuronArray(3);
			input.init();
			input.addNeuron(0, 1);
			input.addNeuron(1, 2);
			input.addNeuron(2, 3);

			inference.addNeurons(0, input);
			SparseNeuronArray output = new SparseNeuronArray(500000);
			output.init();
			output.addNeuron(2);
			output.addNeuron(4);
			inference.addNeurons(1, output);

			Mapping map = new OutputMappingDenseToSparse(input, output, connectedLayer);
			map.setParentInference(inference);		
			inference.addMapping(map);

			inference.init();
			inference.forward();

			output.addError(2, 1);
			output.addError(4, 2);

			inference.backward();		
			inference.commit(0.1);

		}
		System.err.println("experiment took " + (System.currentTimeMillis() - timeStart) + " millis");
	}
}
