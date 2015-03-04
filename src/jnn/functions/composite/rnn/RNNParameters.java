package jnn.functions.composite.rnn;

import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.StaticLayer;

public class RNNParameters {
	DenseFullyConnectedLayer inputTransformLayer;
	StaticLayer initialStateLayer;
	
	public RNNParameters(int inputDim, int stateDim) {
		inputTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
		inputTransformLayer.initialize(true, false);
		initialStateLayer = new StaticLayer(stateDim);
	}
	
	public void update(double learningRate, double momentum){
		inputTransformLayer.updateWeights(learningRate, momentum);
		initialStateLayer.updateWeights(learningRate, momentum);

	}
}