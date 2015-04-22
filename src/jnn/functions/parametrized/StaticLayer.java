package jnn.functions.parametrized;

import java.io.BufferedReader;
import java.io.PrintStream;

import jnn.features.DenseFeatureMatrix;
import jnn.features.DenseFeatureVector;
import jnn.functions.VoidToDenseTransform;
import jnn.mapping.OutputMappingVoidToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.GlobalParameters;

public class StaticLayer extends Layer implements VoidToDenseTransform{
	public int outputDim;
	public DenseFeatureVector outputVec;
	
	public StaticLayer(int outputDim) {
		this.outputDim = outputDim;
		outputVec = new DenseFeatureVector(outputDim);		
		outputVec.initializeUniform(-0.1,0.1);
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
	
	public void save(PrintStream out) {
		out.println(outputDim);
		outputVec.save(out);
	}

	public static StaticLayer load(BufferedReader in) {
		try {
			int outputDim = Integer.parseInt(in.readLine());
			StaticLayer layer = new StaticLayer(outputDim);
			layer.outputVec = DenseFeatureVector.load(in);
			return layer;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	
}
