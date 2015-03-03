package jnn.functions.parametrized;

import jnn.features.DenseFeatureVector;
import jnn.functions.VoidToDenseTransform;
import jnn.mapping.OutputMappingVoidToDense;
import jnn.neuron.DenseNeuronArray;

public class StaticLayer extends Layer implements VoidToDenseTransform{
	public int outputDim;
	public DenseFeatureVector outputVec;
	
	public StaticLayer(int outputDim) {
		this.outputDim = outputDim;
		outputVec = new DenseFeatureVector(outputDim);
		outputVec.normalizedInitializationHtan(1, outputDim);
	}	
	
	@Override
	public void forward(DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingVoidToDense mapping) {
		output.setOutputRange(outputStart, outputEnd, outputVec.getWeights());
	}
	@Override
	public void backward(DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingVoidToDense mapping) {
		outputVec.storeGradients(mapping.getId(), output.getErrorRange(outputStart, outputEnd));
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		outputVec.update();
	}	

	@Override
	public String toString() {
		return "this is a " + outputDim+" dim static layer" + "\n weights:\n" + outputVec.getWeights();
	}
	
//	public void save(PrintStream out) {
//		outputVec.save(out);
//	}
//
//	public void load(Scanner reader) {
//		outputVec.load(reader);
//	}
	
	
}
