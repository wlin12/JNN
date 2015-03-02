package jnn.functions.nonparametrized;

import java.util.Set;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.SparseToSparseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;

import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Exp;
import org.nd4j.linalg.api.activation.Tanh;
import org.nd4j.linalg.api.ndarray.INDArray;

import util.TanFuncs;

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
