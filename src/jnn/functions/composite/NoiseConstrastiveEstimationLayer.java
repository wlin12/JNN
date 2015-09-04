package jnn.functions.composite;

import java.io.BufferedReader;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Iterator;

import jnn.functions.DenseToStringTransform;
import jnn.functions.parametrized.SparseOutputFullyConnectedLayer;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingDenseToSparse;
import jnn.mapping.OutputMappingDenseToString;
import jnn.mapping.OutputMappingStringToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.objective.WordSoftmaxSparseObjective;
import jnn.training.GlobalParameters;
import jnn.training.GraphInference;
import util.MapUtils;
import util.TopNList;
import vocab.Vocab;
import vocab.WordEntry;

public class NoiseConstrastiveEstimationLayer extends AbstractSofmaxObjectiveLayer implements DenseToStringTransform{

	private static final String OUTPUT_KEY = "output";
	private static final String KEYS = "keys";
	
	int inputDim;
		
	Vocab outputVocab;
	String UNK;
	SparseOutputFullyConnectedLayer sparseOutput;
	int samplingSize;
	int dropoutStartId;

	double[] noiseDistribution;
		
	public NoiseConstrastiveEstimationLayer(int inputDim, Vocab outputVocab, int samplingSize, String UNK) {
		super();		
		this.inputDim = inputDim;
		this.samplingSize = samplingSize;
		this.outputVocab = outputVocab;
		this.UNK = UNK;
		sparseOutput = new SparseOutputFullyConnectedLayer(inputDim, outputVocab.getTypes());
		//sparseOutput.setUseBias(false);
		sparseOutput.setMomentum(false);
		dropoutStartId = (int)(0.95*outputVocab.getTypes());
		buildNoiseDistribution();
	}	

	public void buildNoiseDistribution(){
		noiseDistribution = new double[outputVocab.getTypes()];
		for(int i = 0; i < outputVocab.getTypes(); i++){
//			System.err.println(outputVocab.getEntryFromId(i).word);
//			System.err.println(outputVocab.getEntryFromId(i).count);
			noiseDistribution[i] = outputVocab.getEntryFromId(i).count/(double)outputVocab.getTokens();
		}
	}
	
	public WordSoftmaxSparseObjective buildInference(DenseNeuronArray input, int inputStart, int inputEnd, String expected, GraphInference inference){
		WordEntry expectedEntry = outputVocab.getEntry(expected);
		if(expectedEntry==null){
			expectedEntry = outputVocab.getEntry(UNK);			
		}
//		if(expectedEntry==null || (inference.isTrain() && expectedEntry.id > dropoutStartId && FastMath.random() > 0.5)){
//			expectedEntry = outputVocab.getEntry(UNK);
//		}
		int expectedId = expectedEntry.id;
		inference.addNeurons(0,input);
		SparseNeuronArray sparseNeuron = new SparseNeuronArray(outputVocab.getTypes());
		HashMap<Integer, Double> keys = null;
		if(inference.isTrain()){
			keys = new HashMap<Integer, Double>();
			addRandomSamplingNeurons(sparseNeuron, expectedId,keys);	
		}
		else{
			addAllNeurons(sparseNeuron);
		}
		inference.addNeurons(sparseNeuron);
		inference.addMapping(new OutputMappingDenseToSparse(input, sparseNeuron, sparseOutput));
		WordSoftmaxSparseObjective objective = new WordSoftmaxSparseObjective(sparseNeuron, expectedId, keys);
		return objective;
	}
	
	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		WordSoftmaxSparseObjective obj = buildInference(input, inputStart, inputEnd, output.getExpected(), mapping.getSubInference());
		mapping.getSubInference().init();
		mapping.getSubInference().forward();
		mapping.setForwardParam(OUTPUT_KEY, obj);
		if(mapping.getSubInference().isTrain()){			
			obj.addErrorNCE(mapping.getSubInference().getNorm(),samplingSize, noiseDistribution);	
		}
		else{
			obj.computeLoglikelihood(samplingSize, noiseDistribution);
		}
		output.setScore(obj.getLL());
	}
	
	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		mapping.getSubInference().backward();
	}
	
	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			StringNeuronArray[] output,
			OutputMappingDenseArrayToStringArray mapping) {
		WordSoftmaxSparseObjective[] objs = new WordSoftmaxSparseObjective[input.length];
		for(int i = 0; i < input.length; i++){
			objs[i] = buildInference(input[i], inputStart, inputEnd, output[i].getExpected(), mapping.getSubInference());	
		}
		mapping.getSubInference().init();
		mapping.getSubInference().forward();
		for(int i = 0; i < input.length; i++){
			if(mapping.getSubInference().isTrain()){			
				objs[i].addErrorNCE(mapping.getSubInference().getNorm(), samplingSize, noiseDistribution);
			}
			else{
				objs[i].computeLoglikelihood(samplingSize, noiseDistribution);
				//System.err.println(output[i].expected + " " + objs[i].getLL());
			}
			output[i].setScore(objs[i].getLL());
		}

		mapping.setForwardParam(OUTPUT_KEY, objs);
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, StringNeuronArray[] output,
			OutputMappingDenseArrayToStringArray mapping) {
		mapping.getSubInference().backward();
	}
	
	public void addRandomSamplingNeurons(SparseNeuronArray output, int correctIndex, HashMap<Integer, Double> keys){		
		output.addNeuron(correctIndex);
		for(int i = 0; i < samplingSize;){
			int wordId = outputVocab.getRandomEntryByCount().id;
			if(wordId != correctIndex){
				output.addNeuron(wordId);
				i++;
				MapUtils.add(keys, wordId, 1.0);
			}
			else{
				i--;
			}
		}
	}
	
	public void addAllNeurons(SparseNeuronArray output){		
		for(int i = 0; i < outputVocab.getTypes(); i++){
			output.addNeuron(i);			
		}
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		sparseOutput.updateWeights(learningRate, momentum);
	}
	
	public void save(PrintStream out){
		out.println(inputDim);
		out.println(samplingSize);
		out.println(UNK);

		outputVocab.saveVocab(out);
		sparseOutput.save(out);
	}

	@Override
	public String decode(DenseNeuronArray input) {
		GraphInference inference = new GraphInference(0, false);
		WordSoftmaxSparseObjective objective = buildInference(input, 0, inputDim, null, inference);
		inference.init();
		inference.forward();
		return outputVocab.getEntryFromId(objective.getNCEMaxIndex(samplingSize, noiseDistribution)).word;
	}
	
	@Override
	public TopNList<String> getTopN(DenseNeuronArray input, int n) {
		GraphInference inference = new GraphInference(0, false);
		WordSoftmaxSparseObjective objective = buildInference(input, 0, inputDim, null, inference);
		inference.init();
		inference.forward();
		TopNList<Integer> topN = objective.getNCETopN(n, samplingSize, noiseDistribution);
		Iterator<Double> scoreIt = topN.getObjectScore().iterator();	
		Iterator<Integer> wordIt = topN.getObjectList().iterator();
		TopNList<String> topNString = new TopNList<String>(n);		
		while(wordIt.hasNext()){
			topNString.add(outputVocab.getEntryFromId(wordIt.next()).word, scoreIt.next());
		}
		return topNString;
	}
	
	public static NoiseConstrastiveEstimationLayer load(BufferedReader in) {
		try {
			int inputDim = Integer.parseInt(in.readLine());
			int samplingSize = Integer.parseInt(in.readLine());
			String UNK = in.readLine();
			Vocab vocab = Vocab.loadVocab(in);
			NoiseConstrastiveEstimationLayer layer = new NoiseConstrastiveEstimationLayer(inputDim, vocab, samplingSize,UNK);
			layer.sparseOutput = SparseOutputFullyConnectedLayer.load(in);
			layer.dropoutStartId = (int)(0.99*vocab.getTypes());
			layer.buildNoiseDistribution();
			return layer;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static void main(String[] args){
		GlobalParameters.useMomentumDefault = true;
		Vocab vocab = new Vocab();
		vocab.addWordToVocab("p", 11);
		vocab.addWordToVocab("tiger", 10);
		vocab.addWordToVocab("monkey", 7);
		vocab.addWordToVocab("cat", 3);
		vocab.addWordToVocab("dog", 3);
		vocab.addWordToVocab("mouse", 2);
		vocab.addWordToVocab("spider", 1);
		vocab.addWordToVocab("bat", 1);
		vocab.addWordToVocab("rhino", 1);
		vocab.addWordToVocab("whale", 1);
		vocab.addWordToVocab("eagle", 1);
		vocab.addWordToVocab("dove", 1);
		vocab.addWordToVocab("rat", 1);
		vocab.addWordToVocab("pidgeon", 1);
		vocab.addWordToVocab("ant", 1);
		vocab.addWordToVocab("python", 1);
		vocab.sortVocabByCount();
		
		int hiddenDim = 5;
		LookupTable input = new LookupTable(vocab, hiddenDim);
		NoiseConstrastiveEstimationLayer output = new NoiseConstrastiveEstimationLayer(hiddenDim, vocab, 5, "<unk>");
		
		for(int i = 0; i < 100000000; i++){
			GraphInference inference = new GraphInference(0, true);
			WordEntry entry = vocab.getRandomEntryByCount();
			if(entry.id == vocab.getTypes()-1) entry = vocab.getRandomEntryByCount();
			
			String inputStr = entry.word;
			DenseNeuronArray hidden = new DenseNeuronArray(hiddenDim);
			inference.addNeurons(hidden);
			StringNeuronArray out = new StringNeuronArray();
			
			inference.addMapping(new OutputMappingStringToDense(inputStr, hidden, input));
			inference.addMapping(new OutputMappingDenseToString(hidden, out, output));
			
			out.setExpected(inputStr);
			
			inference.init();
			inference.forward();
			inference.backward();
			inference.commit(0);
			
			if(i % 10 == 0){
				double ll = 0;
				for(int word = 0; word < vocab.getTypes(); word++){
					
				inference = new GraphInference(0, false);
				inputStr = vocab.getEntryFromId(word).word;
				hidden = new DenseNeuronArray(hiddenDim);
				inference.addNeurons(hidden);
				out = new StringNeuronArray();
				
				inference.addMapping(new OutputMappingStringToDense(inputStr, hidden, input));
				inference.addMapping(new OutputMappingDenseToString(hidden, out, output));
				out.setExpected(inputStr);

				inference.init();
				inference.forward();
					ll+=out.getScore()*vocab.getEntryFromId(word).count;
				}
				System.err.println(ll);
			}			
		}
	}
}
