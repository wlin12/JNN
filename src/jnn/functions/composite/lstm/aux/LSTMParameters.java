package jnn.functions.composite.lstm.aux;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.StaticLayer;

public class LSTMParameters {
	public DenseFullyConnectedLayer inputTransformLayer;
	public DenseFullyConnectedLayer forgetTransformLayer;
	public DenseFullyConnectedLayer outputTransformLayer;
	public DenseFullyConnectedLayer cellTransformLayer;
	public StaticLayer initialStateLayer;
	public StaticLayer initialCellLayer;
	
	private LSTMParameters() {
		
	}
	
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

	public void save(PrintStream out){
		out.println("lstm params");
		initialCellLayer.save(out);
		initialStateLayer.save(out);
		inputTransformLayer.save(out);
		forgetTransformLayer.save(out);
		cellTransformLayer.save(out);
		outputTransformLayer.save(out);
	}
	
	public static LSTMParameters load(BufferedReader in){
		try{in.readLine();}catch(IOException e){throw new RuntimeException();};
		LSTMParameters params = new LSTMParameters();
		params.initialCellLayer = StaticLayer.load(in);
		params.initialStateLayer = StaticLayer.load(in);
		params.inputTransformLayer = DenseFullyConnectedLayer.load(in);
		params.forgetTransformLayer = DenseFullyConnectedLayer.load(in);
		params.cellTransformLayer = DenseFullyConnectedLayer.load(in);
		params.outputTransformLayer = DenseFullyConnectedLayer.load(in);
		return params;
	}
}

