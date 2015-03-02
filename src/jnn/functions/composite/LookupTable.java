package jnn.functions.composite;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;

import jnn.functions.SparseToDenseTransform;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.functions.parametrized.SparseFullyConnectedLayer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import jnn.training.TreeInference;

import org.nd4j.linalg.api.ndarray.INDArray;

import util.PrintUtils;
import vocab.VocabWithHuffmanTree;
import vocab.WordEntry;

public class LookupTable extends Layer implements SparseToDenseTransform{
	
	private static String OUTPUT_KEY = "output";
	
	SparseFullyConnectedLayer inputToDenseLayer;
	VocabWithHuffmanTree vocab;
	int vocabSize;
	int outputDim;
	int minCountToUpdate = 0;
	HashSet<Integer> keysToUpdate = null;

	ArrayList<DenseFullyConnectedLayer> hiddenLayers = new ArrayList<DenseFullyConnectedLayer>();
	ArrayList<Integer> hiddenLayersDim = new ArrayList<Integer>();
		
	public LookupTable(VocabWithHuffmanTree vocab, int outputDim) {
		super();
		this.vocabSize = vocab.getTypes();
		this.vocab = vocab;
		this.outputDim = outputDim;
		inputToDenseLayer = new SparseFullyConnectedLayer(vocabSize, outputDim);
	}
	
	public LookupTable(VocabWithHuffmanTree vocab, int outputDim, SparseFullyConnectedLayer inputLayer, ArrayList<DenseFullyConnectedLayer> hiddenLayers, ArrayList<Integer> hiddenLayersDim) {
		super();
		this.vocabSize = vocab.getTypes();
		this.vocab = vocab;
		this.outputDim = outputDim;		
		this.inputToDenseLayer = inputLayer;
		this.hiddenLayers = hiddenLayers;
		this.hiddenLayersDim = hiddenLayersDim;
	}
	
	public void initialize(double[][] vals){
		inputToDenseLayer.initialize(vals);
	}
	
	public LookupTable addHidden(int size){
		int outDim = outputDim;
		if(hiddenLayersDim.size() > 0){
			outDim = hiddenLayersDim.get(hiddenLayersDim.size()-1);
		}
		DenseFullyConnectedLayer hidden = new DenseFullyConnectedLayer(outDim, size);
		hidden.useBias = false;
		ArrayList<DenseFullyConnectedLayer> hiddenLayers = new ArrayList<DenseFullyConnectedLayer>();
		hiddenLayers.addAll(this.hiddenLayers);
		hiddenLayers.add(hidden);
		ArrayList<Integer> hiddenLayersDim = new ArrayList<Integer>();
		hiddenLayersDim.addAll(this.hiddenLayersDim);
		hiddenLayersDim.add(size);
		LookupTable newTable = new LookupTable(vocab, outputDim, inputToDenseLayer, hiddenLayers, hiddenLayersDim);
		return newTable;
	}
	
	public INDArray getWeights(int word){
		return inputToDenseLayer.getWeight(word);
	}

	@Override
	public void forward(SparseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingSparseToDense mapping) {
		TreeInference inference = mapping.getSubInference();
		inference.addNeurons(0, input);

		DenseNeuronArray current = new DenseNeuronArray(outputDim); 
		current.setName("dense word array 1");
		inference.addNeurons(current);
		
		inference.addMapping(new OutputMappingSparseToDense(inputStart, inputEnd, 0, outputDim-1, input, current, inputToDenseLayer));
		
		for(int i = 0; i < hiddenLayers.size(); i++){
			DenseNeuronArray next = new DenseNeuronArray(hiddenLayersDim.get(i));
			inference.addNeurons(next);
			inference.addMapping(new OutputMappingDenseToDense(current, next, hiddenLayers.get(i)));
			
			current = next;
			current.setName("dense word array " + (i+2));
		}
		
		mapping.setForwardParam(OUTPUT_KEY, current);
		
		inference.init();
		inference.forward();
		for(int d = 0; d < current.size; d++){				
			output.addNeuron(d+outputStart, current.getNeuron(d));
		}
	}

	@Override
	public void backward(SparseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingSparseToDense mapping) {
		
		TreeInference inference = mapping.getSubInference();
		DenseNeuronArray current = (DenseNeuronArray)mapping.getForwardParam(OUTPUT_KEY);		

		for(int d = 0; d < current.size; d++){	
			current.addError(d, output.getError(d+outputStart));
		}
		
		inference.backward();
	}	
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		if(keysToUpdate == null){
			inputToDenseLayer.updateWeights(learningRate, momentum);
		}
		else{
			inputToDenseLayer.updateWeights(keysToUpdate);			
		}
		for(DenseFullyConnectedLayer hiddenLayer : hiddenLayers){
			hiddenLayer.updateWeights(learningRate, momentum);
		}
	}
	
	public void setPretrainedWeight(int word, double[] weights, boolean addRandomNoise){
		inputToDenseLayer.setRegularizeWeights(word, weights, addRandomNoise);
	}
	
	public void setPretrainedWeight(int word, double[] weights){		
		setPretrainedWeight(word, weights, true);
	}
	
	public void minCountToUpdate(int minCount) {
		if(minCount == 0){
			keysToUpdate = null;
			return;
		}
		this.minCountToUpdate = minCount;
		keysToUpdate = new HashSet<Integer>();
		for(int i = 0; i < vocabSize; i++){
			WordEntry entry = vocab.getEntryFromId(i);
			int count = entry.count;
			if(count >= minCount){
				keysToUpdate.add(i);
			}
		}
	}
	
}
