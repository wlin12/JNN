package jnn.functions.composite.attention;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

import jnn.functions.DenseArrayToDenseTransform;
import jnn.functions.nonparametrized.CopyLayer;
import jnn.functions.nonparametrized.SoftmaxLayer;
import jnn.functions.nonparametrized.TanSigmoidLayer;
import jnn.functions.nonparametrized.WeightedSumLayer;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.GraphInference;

public class AttentionLayer extends Layer implements DenseArrayToDenseTransform{
	
	public static final String OUTPUT_KEY = "outputkey";
	public static final String ATTENTION_KEY = "atentionkey";
	public static final String STATES_KEY = "stateskey";
	public static final String DEPENDENCY_KEY = "depkey";
	
	int inputDim;
	int dependencyDim;
	int stateDim;
	
	DenseFullyConnectedLayer inputToState;
	DenseFullyConnectedLayer dependencyToState;
	DenseFullyConnectedLayer stateToProbability;
	
	public AttentionLayer() {
	}
	
	public AttentionLayer(int inputDim, int dependencyDim, int stateDim) {
		super();
		this.inputDim = inputDim;
		this.dependencyDim = dependencyDim;
		this.stateDim = stateDim;
		
		inputToState = new DenseFullyConnectedLayer(inputDim, stateDim);
		dependencyToState = new DenseFullyConnectedLayer(dependencyDim, stateDim);
		stateToProbability = new DenseFullyConnectedLayer(stateDim, 1);
		dependencyToState.initializeForTanhSigmoid(inputDim + dependencyDim);
		inputToState.initializeForTanhSigmoid(inputDim + dependencyDim);
	}
	
	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDense mapping) {
		GraphInference inference = mapping.getSubInference();
		inference.addNeurons(input);
		
		DenseNeuronArray dependency = input[input.length-1];
		DenseNeuronArray dependencyState = new DenseNeuronArray(stateDim);
		inference.addNeurons(dependencyState);
		inference.addMapping(new OutputMappingDenseToDense(dependency, dependencyState, dependencyToState));		
		
		DenseNeuronArray[] states = DenseNeuronArray.asArray(input.length-1, stateDim);
		DenseNeuronArray[] statesTanh = DenseNeuronArray.asArray(input.length-1, stateDim);
		DenseNeuronArray scores = new DenseNeuronArray(input.length-1);
		inference.addNeurons(states);
		inference.addNeurons(statesTanh);
		inference.addNeurons(scores);
		for(int i = 0; i < states.length; i++){
			inference.addMapping(new OutputMappingDenseToDense(dependencyState, states[i], CopyLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(input[i], states[i], inputToState));
			inference.addMapping(new OutputMappingDenseToDense(states[i], statesTanh[i], TanSigmoidLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(0, statesTanh[i].size-1, i, i, statesTanh[i], scores, stateToProbability));
		}
		mapping.setForwardParam(STATES_KEY, states);
		mapping.setForwardParam(DEPENDENCY_KEY, dependencyState);
		
		DenseNeuronArray probabilities = new DenseNeuronArray(input.length-1);
		inference.addNeurons(probabilities);
		inference.addMapping(new OutputMappingDenseToDense(scores, probabilities, SoftmaxLayer.singleton));
		
		mapping.setForwardParam(ATTENTION_KEY, probabilities);
		
		DenseNeuronArray[] inputWithProbability = new DenseNeuronArray[input.length];
		for(int a = 0; a < input.length-1; a++){
			inputWithProbability[a] = input[a];
		}
		inputWithProbability[input.length-1] = probabilities;
		
		DenseNeuronArray outputStored = new DenseNeuronArray(inputDim);
		inference.addNeurons(outputStored);
		inference.addMapping(new OutputMappingDenseArrayToDense(inputWithProbability, outputStored, WeightedSumLayer.singleton));		
		inference.init();
		inference.forward();
		
		for(int d = 0; d < stateDim; d++){				
			output.addNeuron(d+outputStart, outputStored.getNeuron(d));
		}
		
		mapping.setForwardParam(OUTPUT_KEY, outputStored);
	}
	
	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDense mapping) {
		DenseNeuronArray outputStored = (DenseNeuronArray)mapping.getForwardParam(OUTPUT_KEY);
		for(int d = 0; d < stateDim; d++){				
			outputStored.addError(d, output.getError(d+outputStart));
		}
		mapping.getSubInference().backward();
		//DenseNeuronArray[] states = ((DenseNeuronArray[]) mapping.getForwardParam(STATES_KEY));
		//System.err.println((DenseNeuronArray) mapping.getForwardParam(ATTENTION_KEY));
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		inputToState.updateWeights(0, 0);
		dependencyToState.updateWeights(0, 0);
		stateToProbability.updateWeights(0, 0);
	}
	
	public void save(PrintStream out){
		out.println(inputDim);
		out.println(dependencyDim);
		out.println(stateDim);
		inputToState.save(out);
		dependencyToState.save(out);
		stateToProbability.save(out);
	}
	
	public static AttentionLayer load(BufferedReader in){
		AttentionLayer ret = new AttentionLayer();
		try {
			ret.inputDim = Integer.parseInt(in.readLine());
			ret.dependencyDim = Integer.parseInt(in.readLine());
			ret.stateDim = Integer.parseInt(in.readLine());
			ret.inputToState = DenseFullyConnectedLayer.load(in);
			ret.dependencyToState = DenseFullyConnectedLayer.load(in);
			ret.stateToProbability = DenseFullyConnectedLayer.load(in);
			return ret;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}
