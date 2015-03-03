package jnn.functions.nonparametrized;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;

public class LogLayer extends Layer implements DenseToDenseTransform{
	
	public static LogLayer singleton = new LogLayer();

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		for(int i = 0; i < inputDim; i++){			
			output.addNeuron(i+outputStart,Math.log(input.getNeuron(i+inputStart)));
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
//		INDArray x = input.getOutputRange(inputStart, inputEnd);		
//		INDArray yGrad = output.getErrorRange(outputStart, outputEnd);
//		INDArray xGrad = activation.applyDerivative(x).mul(yGrad);
//		input.setErrorRange(inputStart, inputEnd, xGrad);
		for(int i = 0; i < inputDim; i++){	
			input.addError(i+inputStart, (1/input.getNeuron(i+inputStart))*output.getError(i+outputStart));
		}
	}
}
