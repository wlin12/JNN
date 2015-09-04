package jnn.functions.composite;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

import jnn.neuron.DenseNeuronArray;
import jnn.training.GlobalParameters;
import util.ExpTable;
import util.SerializeUtils;
import vocab.Vocab;

public class FastNegativeSamplingLayer {
	double[] weights;
	int inputDim;	
	Vocab outputVocab;
	int samplingSize;

	public FastNegativeSamplingLayer(int inputDim, Vocab outputVocab, int samplingSize) {
		this.inputDim = inputDim;
		this.outputVocab = outputVocab;
		this.samplingSize = samplingSize;
		this.weights = new double[inputDim * outputVocab.getTypes()];
	}
	
	public double forward(DenseNeuronArray input, int output){
		double f = 0;
		for(int i = 0 ; i < inputDim; i++){
			f += input.getNeuron(i) * weights[inputDim*output + i];
		}

		if(f>ExpTable.SingletonMaxExp){
			return 1;
		}
		if(f<-ExpTable.SingletonMaxExp){
			return 0;
		}		
		return ExpTable.getExpTable(f);
	}
	
	public void backward(double f, double expected, int output, DenseNeuronArray input, double norm, double normInput){
		double g = (expected - f);
		
		for(int i = 0 ; i < inputDim; i++){
			input.addError(i,normInput * g * weights[inputDim*output + i]);
		}
		
		double val = 0;
		for(int i = 0 ; i < inputDim; i++){
			val = input.getNeuron(i) * g * GlobalParameters.learningRateDefault * norm;
			if(val > 1){
				val = 1;
			}
			else if(val < -1){
				val = -1;
			}
			weights[inputDim*output + i] += val;
		}
	}

	public void forwardBackward(DenseNeuronArray input, int outputId, double norm, double normInput) {
		double f = forward(input, outputId);		

		backward(f, 1, outputId, input, norm, normInput);
		
		for(int i = 0; i < samplingSize; i++){
			int randomWordId = outputVocab.getRandomEntryByCount().id;
			while(outputId == randomWordId) randomWordId = outputVocab.getRandomEntryByCount().id;
			f = forward(input, randomWordId);
			backward(f, 0, randomWordId, input, norm, normInput);
		}
	}
	
	public double error(DenseNeuronArray input, int outputId) {
		double f = forward(input, outputId);
		double error = 0;
		error += (1 - f);
		for(int i = 0; i < samplingSize; i++){
			int randomWordId = outputVocab.getRandomEntryByCount().id;
			while(outputId == randomWordId) randomWordId = outputVocab.getRandomEntryByCount().id;
			f = forward(input, randomWordId);
			error += f;
		}
		return error;
	}

	public void save(PrintStream out) {
		out.println(inputDim);
		out.println(samplingSize);
		SerializeUtils.saveDoubleArray(weights, out);		
	}
	
	public static FastNegativeSamplingLayer load(BufferedReader in, Vocab vocab){
		int inputDim;
		try {
			inputDim = Integer.parseInt(in.readLine());
			int samplingSize = Integer.parseInt(in.readLine());
			double[] weights = SerializeUtils.loadDoubleArray(in);
			FastNegativeSamplingLayer layer = new FastNegativeSamplingLayer(inputDim, vocab, samplingSize);
			layer.weights = weights;
			return layer;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public double getWeight(int id, int i) {
		return weights[inputDim*id + i];
	}
	
}
