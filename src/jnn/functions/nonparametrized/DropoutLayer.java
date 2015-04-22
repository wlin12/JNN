package jnn.functions.nonparametrized;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.SparseToSparseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;

public class DropoutLayer extends Layer implements DenseToDenseTransform, SparseToSparseTransform{

	private static final String DROPPED_KEY = "dropped";

	public static HashMap<Double, DropoutLayer> singletons = new HashMap<Double, DropoutLayer>();
	public double dropoutProb;

	public static DropoutLayer get(double dropout){
		DropoutLayer ret = singletons.get(dropout);
		if(ret != null){
			return ret;
		}
		ret = new DropoutLayer(dropout);
		singletons.put(dropout, ret);
		return ret;
	}

	private DropoutLayer(double dropoutProb) {
		super();
		this.dropoutProb = dropoutProb;
	}

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		if(!mapping.getSubInference().isTrain() || Math.random()>dropoutProb){
			for(int i = 0; i < inputDim; i++){
				output.addNeuron(i+outputStart, input.getNeuron(i+inputStart));
			}
			mapping.setForwardParam(DROPPED_KEY, false);
		}
		else{
			mapping.setForwardParam(DROPPED_KEY, true);
		}
	}

	@Override
	public void forward(SparseNeuronArray input, int inputStart, int inputEnd,
			SparseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseToSparse mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		if(!mapping.getSubInference().isTrain() || Math.random()>dropoutProb){
			Set<Integer> indexes = input.getNonZeroKeys();
			for(int i : indexes){
				output.addNeuron(i+outputStart,input.getOutput(i+inputStart));
			}
			mapping.setForwardParam(DROPPED_KEY, false);
		}
		else{
			mapping.setForwardParam(DROPPED_KEY, true);
		}
	}

	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		if(!(Boolean)mapping.getForwardParam(DROPPED_KEY)){
			for(int i = 0; i < inputDim; i++){	
				input.addError(i+inputStart, output.getError(i+outputStart));
			}
		}
	}

	@Override
	public void backward(SparseNeuronArray input, int inputStart, int inputEnd,
			SparseNeuronArray output, int outputStart, int outputEnd, OutputMappingSparseToSparse mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		if(!(Boolean)mapping.getForwardParam(DROPPED_KEY)){
			Set<Integer> indexes = input.getNonZeroKeys();
			for(int i : indexes){
				input.addError(i+inputStart, output.getError(i+outputStart));
			}
		}
	}
}
