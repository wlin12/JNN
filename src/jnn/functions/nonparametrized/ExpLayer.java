package jnn.functions.nonparametrized;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;

import org.nd4j.linalg.api.activation.Exp;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ExpLayer extends Layer implements DenseToDenseTransform{
	
	public static ExpLayer singleton = new ExpLayer();
	public static Exp exp = new Exp();
	
	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		INDArray inputRange = input.getOutputRange(inputStart, inputEnd);
		INDArray inputRangeExp = exp.apply(inputRange);
		output.setOutputRange(outputStart, outputEnd, inputRangeExp);
	}
	
	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		INDArray inputRange = input.getOutputRange(inputStart, inputEnd);
		INDArray inputRangeExp = exp.apply(inputRange);
		input.setErrorRange(inputStart, inputEnd, inputRangeExp.mul(output.getErrorRange(outputStart, outputEnd)));
	}
}
