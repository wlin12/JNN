package jnn.functions.nonparametrized;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;

public class SoftmaxLayer extends Layer implements DenseToDenseTransform{
	
	public static SoftmaxLayer singleton = new SoftmaxLayer();
	
	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		double maxInput = input.getMax();
		double norm = 0;
		double[] outputTemp = new double[input.size];
		for(int i = 0; i < inputDim; i++){						
			 outputTemp[i] = Math.exp(input.getNeuron(i+inputStart) - maxInput);
			 norm+=outputTemp[i];
		}
		for(int i = 0; i < inputDim; i++){						
			output.addNeuron(i+outputStart,outputTemp[i]/norm);
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
		for(int i = 0; i < inputDim; i++){	
			input.addError(i+inputStart, output.getError(i+outputStart));
		}
	}
}
