package jnn.functions.composite.lstm;

import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.StaticLayer;

public class LSTMParameters {
	DenseFullyConnectedLayer inputTransformLayer;
	DenseFullyConnectedLayer forgetTransformLayer;
	DenseFullyConnectedLayer outputTransformLayer;
	DenseFullyConnectedLayer cellTransformLayer;
	StaticLayer initialStateLayer;
	StaticLayer initialCellLayer;
	
	public LSTMParameters(int inputDim, int stateDim) {
		inputTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
		inputTransformLayer.initialize(true, false);
		forgetTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
		forgetTransformLayer.initialize(true, false);
		outputTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
		outputTransformLayer.initialize(true, false);
		cellTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
		cellTransformLayer.initialize(false, true);
		initialStateLayer = new StaticLayer(stateDim);
		initialCellLayer = new StaticLayer(stateDim);
	}
	
	public void update(double learningRate, double momentum){
		inputTransformLayer.updateWeights(learningRate, momentum);
		forgetTransformLayer.updateWeights(learningRate, momentum);
		outputTransformLayer.updateWeights(learningRate, momentum);
		cellTransformLayer.updateWeights(learningRate, momentum);
		initialCellLayer.updateWeights(learningRate, momentum);
		initialStateLayer.updateWeights(learningRate, momentum);

	}
	
	public void print(){
		System.err.println(inputTransformLayer);
		System.err.println(forgetTransformLayer);
		System.err.println(outputTransformLayer);
		System.err.println(cellTransformLayer);
		System.err.println(initialCellLayer);
		System.err.println(initialStateLayer);
	}
	
}

