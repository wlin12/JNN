package jnn.functions.composite.rnn;

import java.io.BufferedReader;
import java.io.PrintStream;

import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.StaticLayer;

public class RNNParameters {
	DenseFullyConnectedLayer inputTransformLayer;
	StaticLayer initialStateLayer;
	
	private RNNParameters() {
	}
	
	public RNNParameters(int inputDim, int stateDim) {
		inputTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
		inputTransformLayer.initialize(true, false);
		initialStateLayer = new StaticLayer(stateDim);
	}
	
	public void update(double learningRate, double momentum){
		inputTransformLayer.updateWeights(learningRate, momentum);
		initialStateLayer.updateWeights(learningRate, momentum);
	}
	
	public void save(PrintStream out){
		inputTransformLayer.save(out);
		initialStateLayer.save(out);
	}
	
	public static RNNParameters load(BufferedReader in){
		RNNParameters loaded = new RNNParameters();
		loaded.inputTransformLayer = DenseFullyConnectedLayer.load(in);
		loaded.initialStateLayer = StaticLayer.load(in);
		return loaded;
	}
}