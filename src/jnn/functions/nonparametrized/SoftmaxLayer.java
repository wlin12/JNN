package jnn.functions.nonparametrized;

import java.util.Set;

import jnn.functions.DenseToDenseTransform;
import jnn.functions.SparseToSparseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToSparse;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import jnn.training.GraphInference;

import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.Sigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;

import util.ExpTable;
import util.LogAdd;
import util.TanFuncs;

public class SoftmaxLayer extends Layer implements DenseToDenseTransform{
	
	public static SoftmaxLayer singleton = new SoftmaxLayer();
	private static final String PROB = "prob";
	private static final String SUM = "sum";
	
	
	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		int inputDim = inputEnd - inputStart + 1;
		int outputDim = outputEnd - outputStart + 1;
		if(inputDim != outputDim){
			throw new RuntimeException(inputDim + " " + outputDim);
		}
		
		double[] exp = new double[outputDim];
		double[] prob = new double[outputDim];
		double logsum = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < inputDim; i++){		
			double val = input.getNeuron(i+inputStart);
			exp[i] = val;
			if(i==0){
				logsum=val;
			}
			else{
				logsum=LogAdd.logAdd(logsum, val);
			}
		}
		for(int i = 0; i < inputDim; i++){			
			prob[i] = Math.exp(exp[i]-logsum);
		}
				
		mapping.setForwardParam(PROB, prob);
		mapping.setForwardParam(SUM, logsum);
		
		for(int i = 0; i < inputDim; i++){			
			output.addNeuron(i+outputStart, prob[i]);
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
		
		double[] prob = (double[])mapping.getForwardParam(PROB);		
		double logsum = (double)mapping.getForwardParam(SUM);
		double[] error = new double[prob.length];
		double errorSum = 0;
		for(int i = 0; i < inputDim; i++){
			error[i] = output.getError(i+outputStart);
			errorSum+=error[i]*prob[i];
		}
		for(int i = 0; i < inputDim; i++){	
			input.addError(i+inputStart, prob[i]*(error[i] - errorSum));
		}
	}
	
	public static void main(String[] args){		
		DenseNeuronArray inputArray = new DenseNeuronArray(5);
		DenseNeuronArray softmax = new DenseNeuronArray(5);
		inputArray.init();
		inputArray.addNeuron(0, 0.2);
		inputArray.addNeuron(1, 0.3);
		inputArray.addNeuron(2, 0.4);
		inputArray.addNeuron(3, 0.5);
		inputArray.addNeuron(4, 0.6);
		
		
		GraphInference inference = new GraphInference(0, true);
		inference.addNeurons(0,inputArray);
		inference.addNeurons(softmax);
		inference.addMapping(new OutputMappingDenseToDense(inputArray, softmax, SoftmaxLayer.singleton));;
		inference.init();
		inference.forward();
		
		
		softmax.addError(0, -0.1);
		//softmax.addError(1, -0.1);
		//softmax.addError(2, -0.1);
		//softmax.addError(3, -0.1);
		//softmax.addError(4, -0.1);
		
		inference.backward();
		System.err.println(inputArray);
		System.err.println(softmax);
	}
}
