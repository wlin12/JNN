package jnn.functions.composite;
import jnn.functions.DenseArrayToStringArrayTransform;
import jnn.functions.DenseToStringTransform;
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
	}	
	
	public WordSoftmaxSparseObjective buildInference(DenseNeuronArray input, int inputStart, int inputEnd, String expected, GraphInference inference){
		int expectedId = outputVocab.getEntry(expected).id;
		inference.addNeurons(0,input);
		SparseNeuronArray sparseNeuron = new SparseNeuronArray(outputVocab.getTypes());
		addNegativeSamplingNeurons(sparseNeuron, expectedId);
		inference.addNeurons(sparseNeuron);
		inference.addMapping(new OutputMappingDenseToSparse(input, sparseNeuron, sparseOutput));
		inference.init();
		inference.forward();
		WordSoftmaxSparseObjective objective = new WordSoftmaxSparseObjective(sparseNeuron, expectedId);
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
	
	public void addNegativeSamplingNeurons(SparseNeuronArray output, int correctIndex){		
		output.addNeuron(correctIndex);
		while(output.getNonZeroEntries().size()<=samplingSize){
			int wordId = outputVocab.getRandomEntryByCount().id;
			output.addNeuron(wordId);
		}
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		sparseOutput.updateWeights(learningRate, momentum);
	}
	
}
