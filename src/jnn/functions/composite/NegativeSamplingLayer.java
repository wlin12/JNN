package jnn.functions.composite;
import java.io.BufferedReader;
import java.io.PrintStream;
import java.util.HashMap;

import jnn.features.DenseFeatureMatrix;
import jnn.features.DenseFeatureVector;
import jnn.functions.DenseArrayToStringArrayTransform;
import jnn.functions.DenseToStringTransform;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.functions.parametrized.SparseOutputFullyConnectedLayer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingDenseToSparse;
import jnn.mapping.OutputMappingDenseToString;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.objective.WordSoftmaxSparseObjective;
import jnn.training.GraphInference;
import util.MapUtils;
import vocab.Vocab;

public class NegativeSamplingLayer extends Layer implements DenseToStringTransform{

	private static final String OUTPUT_KEY = "output";
	
	int inputDim;
		
	Vocab outputVocab;
	SparseOutputFullyConnectedLayer sparseOutput;
	int samplingSize;
		
	public NegativeSamplingLayer(int inputDim, Vocab outputVocab, int samplingSize) {
		super();
		this.inputDim = inputDim;
		this.samplingSize = samplingSize;
		this.outputVocab = outputVocab;
		sparseOutput = new SparseOutputFullyConnectedLayer(inputDim, outputVocab.getTypes());
		sparseOutput.setUseBias(false);
	}	
	
	public WordSoftmaxSparseObjective buildInference(DenseNeuronArray input, int inputStart, int inputEnd, String expected, GraphInference inference){
		int expectedId = outputVocab.getEntry(expected).id;
		HashMap<Integer, Double> keys = new HashMap<Integer, Double>();

		inference.addNeurons(0,input);
		SparseNeuronArray sparseNeuron = new SparseNeuronArray(outputVocab.getTypes());		
		addNegativeSamplingNeurons(sparseNeuron, expectedId,keys);
		inference.addNeurons(sparseNeuron);
		inference.addMapping(new OutputMappingDenseToSparse(input, sparseNeuron, sparseOutput));
		inference.init();
		inference.forward();
		
		WordSoftmaxSparseObjective objective = new WordSoftmaxSparseObjective(sparseNeuron, expectedId,keys);
		return objective;
	}
	
	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		WordSoftmaxSparseObjective obj = buildInference(input, inputStart, inputEnd, output.getExpected(), mapping.getSubInference());
		mapping.setForwardParam(OUTPUT_KEY, obj);
	}
	
	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		WordSoftmaxSparseObjective obj = (WordSoftmaxSparseObjective)mapping.getForwardParam(OUTPUT_KEY);
		obj.addError(mapping.getSubInference().getNorm());	
		mapping.getSubInference().backward();
	}
	
	public void addNegativeSamplingNeurons(SparseNeuronArray output, int correctIndex, HashMap<Integer, Double> keys){		
		output.addNeuron(correctIndex);
		while(output.getNonZeroEntries().size()<=samplingSize){
			int wordId = outputVocab.getRandomEntryByCount().id;
			output.addNeuron(wordId);
			MapUtils.add(keys, wordId, 1.0);
		}
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		sparseOutput.updateWeights(learningRate, momentum);
	}
	
	public void save(PrintStream out){
		out.println(inputDim);
		out.println(samplingSize);
		outputVocab.saveVocab(out);
		sparseOutput.save(out);
	}

	
	public static NegativeSamplingLayer load(BufferedReader in) {
		try {
			int inputDim = Integer.parseInt(in.readLine());
			int samplingSize = Integer.parseInt(in.readLine());
			Vocab vocab = Vocab.loadVocab(in);
			NegativeSamplingLayer layer = new NegativeSamplingLayer(inputDim, vocab, samplingSize);
			layer.sparseOutput = SparseOutputFullyConnectedLayer.load(in);
			return layer;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
}
