package jnn.functions.composite;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashSet;

import jnn.functions.SparseToDenseTransform;
import jnn.functions.StringArrayToDenseArrayTransform;
import jnn.functions.StringToDenseTransform;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.functions.parametrized.SparseFullyConnectedLayer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToDense;
import jnn.mapping.OutputMappingStringToDense;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import jnn.training.GraphInference;

import org.nd4j.linalg.api.ndarray.INDArray;

import util.IOUtils;
import util.StringUtils;
import vocab.Vocab;
import vocab.WordEntry;

public class LookupTable extends Layer implements SparseToDenseTransform, StringToDenseTransform, StringArrayToDenseArrayTransform{

	private static String OUTPUT_KEY = "output";

	SparseFullyConnectedLayer inputToDenseLayer;
	Vocab vocab;
	int vocabSize;
	int outputDim;
	int minCountToUpdate = 0;
	HashSet<Integer> keysToUpdate = null;

	ArrayList<DenseFullyConnectedLayer> hiddenLayers = new ArrayList<DenseFullyConnectedLayer>();
	ArrayList<Integer> hiddenLayersDim = new ArrayList<Integer>();

	public LookupTable(Vocab vocab, int outputDim) {
		super();
		this.vocabSize = vocab.getTypes();
		this.vocab = vocab;
		this.outputDim = outputDim;
		inputToDenseLayer = new SparseFullyConnectedLayer(vocabSize, outputDim);
	}

	public LookupTable(Vocab vocab, int outputDim, SparseFullyConnectedLayer inputLayer, ArrayList<DenseFullyConnectedLayer> hiddenLayers, ArrayList<Integer> hiddenLayersDim) {
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
	
	public boolean containsWord(String word){
		return vocab.getEntry(word) != null;
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

	public void buildNetwork(SparseNeuronArray input, int inputStart, int inputEnd, DenseNeuronArray output, int outputStart, int outputEnd,
			Mapping mapping, String key){
		GraphInference inference = mapping.getSubInference();
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

		mapping.setForwardParam(key, current);		
	}
	
	@Override
	public void forward(String input, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingStringToDense mapping) {
		WordEntry word = vocab.getEntry(input);
		if(word==null){throw new RuntimeException("unknown forward:" + input);}
		SparseNeuronArray inputNeurons = new SparseNeuronArray(vocabSize);
		inputNeurons.addNeuron(word.getId(), 1);
		buildNetwork(inputNeurons, 0, vocabSize-1, output, outputStart, outputEnd, mapping, OUTPUT_KEY);
		mapping.getSubInference().init();
		mapping.getSubInference().forward();
		DenseNeuronArray rep = (DenseNeuronArray) mapping.getForwardParam(OUTPUT_KEY);
		for(int d = 0; d < rep.size; d++){				
			output.addNeuron(d+outputStart, rep.getNeuron(d));
		}
	}
	
	@Override
	public void forward(String[] input, DenseNeuronArray[] output,
			int outputStart, int outputEnd,
			OutputMappingStringArrayToDenseArray mapping) {
		for(int i = 0; i < input.length; i++){
			WordEntry word = vocab.getEntry(input[i]);
			if(word==null){
				throw new RuntimeException("unknown forward:" + input[i] + "( in " + StringUtils.arrayToString(input) + ")");
			}
			String key = input[i] + "-" + OUTPUT_KEY;
			
			if(mapping.getForwardParam(key)==null){
				SparseNeuronArray inputNeurons = new SparseNeuronArray(vocabSize);
				inputNeurons.addNeuron(word.getId(), 1);
				buildNetwork(inputNeurons, 0, vocabSize-1, output[i], outputStart, outputEnd, mapping, key);
			}
		}		
		
		mapping.getSubInference().init();
		mapping.getSubInference().forward();
		for(int i = 0; i < input.length; i++){
			String key = input[i] + "-" + OUTPUT_KEY;			
			DenseNeuronArray neurons = (DenseNeuronArray) mapping.getForwardParam(key);
			for(int d = 0; d < neurons.size; d++){				
				output[i].addNeuron(d+outputStart, neurons.getNeuron(d));
			}
		}

	}

	@Override
	public void forward(SparseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingSparseToDense mapping) {
		buildNetwork(input, inputStart, inputEnd, output, outputStart, outputEnd, mapping, OUTPUT_KEY);
		mapping.getSubInference().init();
		mapping.getSubInference().forward();
		DenseNeuronArray rep = (DenseNeuronArray) mapping.getForwardParam(OUTPUT_KEY);
		for(int d = 0; d < rep.size; d++){				
			output.addNeuron(d+outputStart, rep.getNeuron(d));
		}
	}	

	@Override
	public void backward(String input, DenseNeuronArray output,
			int outputStart, int outputEnd, OutputMappingStringToDense mapping) {
		GraphInference inference = mapping.getSubInference();
		DenseNeuronArray current = (DenseNeuronArray)mapping.getForwardParam(OUTPUT_KEY);		

		for(int d = 0; d < current.size; d++){	
			current.addError(d, output.getError(d+outputStart));
		}

		inference.backward();
	}
	
	@Override
	public void backward(String[] input, DenseNeuronArray[] output,
			int outputStart, int outputEnd,
			OutputMappingStringArrayToDenseArray mapping) {
		GraphInference inference = mapping.getSubInference();
		for(int i = 0; i < input.length; i++){
			WordEntry word = vocab.getEntry(input[i]);
			if(word==null){throw new RuntimeException("unknown forward:" + input);}
			String key = input[i] + "-" + OUTPUT_KEY;
						
			DenseNeuronArray neurons = (DenseNeuronArray) mapping.getForwardParam(key);
			for(int d = 0; d < neurons.size; d++){	
				neurons.addError(d, output[i].getError(d+outputStart));
			}
		}
		inference.backward();
	}

	@Override
	public void backward(SparseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingSparseToDense mapping) {

		GraphInference inference = mapping.getSubInference();
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
			long count = entry.count;
			if(count >= minCount){
				keysToUpdate.add(i);
			}
		}
	}

	public void save(PrintStream out){
		vocab.saveVocab(out);
		out.println(outputDim);
		out.println(minCountToUpdate);
		inputToDenseLayer.save(out);

		out.println(hiddenLayers.size());
		for(int i = 0; i < hiddenLayers.size(); i++){
			out.println(hiddenLayersDim.get(i));
			hiddenLayers.get(i).save(out);
		}

	}

	public static LookupTable load(BufferedReader in){
		try{
			Vocab vocab = Vocab.loadVocab(in);
			int outputDim = Integer.parseInt(in.readLine());
			int minCountToUpdate = Integer.parseInt(in.readLine());
			LookupTable table = new LookupTable(vocab, outputDim);
			table.inputToDenseLayer = SparseFullyConnectedLayer.load(in);
			table.minCountToUpdate(minCountToUpdate);

			int hiddenLayerNumber = Integer.parseInt(in.readLine());
			for(int i = 0; i < hiddenLayerNumber; i++){
				int hiddenSize = Integer.parseInt(in.readLine());
				table.hiddenLayersDim.add(hiddenSize);
				DenseFullyConnectedLayer hidden = DenseFullyConnectedLayer.load(in);
				table.hiddenLayers.add(hidden);
			}

			return table;
		} catch(IOException e){
			throw new RuntimeException(e);
		}
	}
	
	public DenseNeuronArray getWord(String input){
		DenseNeuronArray output = new DenseNeuronArray(outputDim);
		output.init();
		WordEntry word = vocab.getEntry(input);
		if(word==null){throw new RuntimeException("unknown forward:" + input);}
		SparseNeuronArray inputNeurons = new SparseNeuronArray(vocabSize);
		inputNeurons.addNeuron(word.getId(), 1);
		Mapping dummyMap = new OutputMappingStringToDense(input, output, this);
		GraphInference dummyInference = new GraphInference(0, true);
		dummyMap.setParentInference(dummyInference);
		buildNetwork(inputNeurons, 0, vocabSize-1, output, 0, outputDim, dummyMap, OUTPUT_KEY);
		return output;
	}


	public static void main(String[] args){
		testSaveLoad();
	}

	public static void testSaveLoad(){
		Vocab vocab = new Vocab();
		vocab.addWordToVocab("hello");
		vocab.addWordToVocab("world");
		vocab.addWordToVocab("!");
		vocab.sortVocabByCount();
		vocab.generateHuffmanCodes();

		LookupTable table = new LookupTable(vocab, 10);
		table=table.addHidden(20);
		table=table.addHidden(30);
		System.err.println("original table");
		table.save(System.err);
		table.save(IOUtils.getPrintStream("/tmp/file"));

		System.err.println("loaded table");
		LookupTable loadedTable = LookupTable.load(IOUtils.getReader("/tmp/file"));
		loadedTable.save(System.err);

		String word = "hello";
		for(int i = 0; i < 10; i++){
			GraphInference inference = new GraphInference(0, true);
			DenseNeuronArray projection = new DenseNeuronArray(30);
			inference.addNeurons(1,projection);
			inference.addMapping(new OutputMappingStringToDense(word,projection, table));
			inference.init();
			inference.forward();

			projection.addError(0, 1 - projection.getNeuron(0));
			inference.backward();
			inference.commit(0);
			System.err.println(projection);
		}
	}
}
