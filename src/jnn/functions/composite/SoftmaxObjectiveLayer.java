package jnn.functions.composite;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

import org.apache.commons.math3.util.FastMath;

import jnn.functions.DenseArrayToStringArrayTransform;
import jnn.functions.DenseToStringTransform;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingDenseToString;
import jnn.mapping.OutputMappingStringArrayToStringArray;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.objective.WordSoftmaxDenseObjective;
import jnn.training.GraphInference;
import util.TopNList;
import vocab.Vocab;
import vocab.WordEntry;

public class SoftmaxObjectiveLayer extends AbstractSofmaxObjectiveLayer implements DenseToStringTransform, DenseArrayToStringArrayTransform{

	private static final String OUTPUT_KEY = "output";

	Vocab vocab;
	int inputDim;
	String UNK;
	int dropoutStartId;
	boolean stochasticDropout = false;

	DenseFullyConnectedLayer inputToVocab;	

	public SoftmaxObjectiveLayer(Vocab vocab, int inputDim, String UNK) {
		this(vocab, inputDim, UNK, false);
	}
	
	public SoftmaxObjectiveLayer(Vocab vocab, int inputDim, String UNK, boolean stochasticDropout) {
		super();
		this.vocab = vocab;
		this.inputDim = inputDim;
		this.UNK = UNK;
		this.stochasticDropout = stochasticDropout;
		inputToVocab = new DenseFullyConnectedLayer(inputDim, vocab.getTypes());
		dropoutStartId = (int)(0.95*vocab.getTypes());
	}

	public DenseNeuronArray buildInference(DenseNeuronArray input, int inputStart, int inputEnd, GraphInference inference){
		inference.addNeurons(0, input);
		DenseNeuronArray outputNeurons = new DenseNeuronArray(vocab.getTypes());
		inference.addNeurons(1, outputNeurons);
		inference.addMapping(new OutputMappingDenseToDense(input, outputNeurons, inputToVocab));
		return outputNeurons;
	}

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		DenseNeuronArray outputNeurons = buildInference(input, inputStart, inputEnd, mapping.getSubInference());
		mapping.getSubInference().init();
		mapping.getSubInference().forward();

		mapping.setForwardParam(OUTPUT_KEY, outputNeurons);

		output.setOutput(vocab.getEntryFromId(outputNeurons.maxIndex()).word);
		if(output.getExpected() != null){
			WordEntry expectedEntry = vocab.getEntry(output.getExpected());
			if(expectedEntry==null){
				expectedEntry = vocab.getEntry(UNK);
			}
			int expectedIndex = expectedEntry.id;
			WordSoftmaxDenseObjective objective = new WordSoftmaxDenseObjective(outputNeurons, expectedIndex);
			if(output.negative){
				objective.addNegativeSoftmaxError(mapping.getSubInference().getNorm());			
			}
			else{
				objective.addSoftmaxError(mapping.getSubInference().getNorm());
			}
			output.setScore(objective.getLL());
		}
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			StringNeuronArray[] output, OutputMappingDenseArrayToStringArray mapping) {
		DenseNeuronArray[] outputNeuronPerPos = new DenseNeuronArray[input.length];
		for(int w = 0; w < input.length; w++){
			outputNeuronPerPos[w] = buildInference(input[w], inputStart, inputEnd, mapping.getSubInference());						
		}
		mapping.getSubInference().init();
		mapping.getSubInference().forward();		

		mapping.setForwardParam(OUTPUT_KEY, outputNeuronPerPos);
		for(int w = 0; w < input.length; w++){
			output[w].setOutput(vocab.getEntryFromId(outputNeuronPerPos[w].maxIndex()).word);
		}

		for(int w = 0; w < input.length; w++){
			if(output[w].getExpected() != null){

				WordEntry expectedEntry = vocab.getEntry(output[w].getExpected());
				if(expectedEntry==null){
					expectedEntry = vocab.getEntry(UNK);
				}
				int expectedIndex = expectedEntry.id;
				WordSoftmaxDenseObjective objective = new WordSoftmaxDenseObjective(outputNeuronPerPos[w], expectedIndex);			
				if(output[w].negative){
					objective.addNegativeSoftmaxError(mapping.getSubInference().getNorm());
				}
				else{
					objective.addSoftmaxError(mapping.getSubInference().getNorm());				
				}
				output[w].setScore(objective.getLL());
			}
		}

	}

	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {		
		mapping.getSubInference().backward();
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, StringNeuronArray[] output,
			OutputMappingDenseArrayToStringArray mapping) {
		mapping.getSubInference().backward();		
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		inputToVocab.updateWeights(learningRate, momentum);
	}

	public TopNList<String> getTopN(DenseNeuronArray input, int n){
		GraphInference inference = new GraphInference(0, false);
		DenseNeuronArray outputNeurons = buildInference(input, 0, vocab.getTypes(), inference);
		inference.init();
		inference.forward();
		TopNList<String> list = new TopNList<String>(n);
		for(int i = 0; i < vocab.getTypes(); i++){
			list.add(vocab.getEntryFromId(i).word,outputNeurons.getNeuron(i));
		}
		return list;
	}

	public String decode(DenseNeuronArray input) {
		GraphInference inference = new GraphInference(0, false);
		DenseNeuronArray output = buildInference(input, 0, vocab.getTypes(), inference);
		inference.init();
		inference.forward();
		return vocab.getEntryFromId(output.maxIndex()).word;
	}

	public void save(PrintStream out){
		out.println("word softmax parameters");
		out.println(inputDim);
		out.println(UNK);
		vocab.saveVocab(out);
		inputToVocab.save(out);
	}

	public static SoftmaxObjectiveLayer load(BufferedReader in){
		try {
			in.readLine();
			int inputDim = Integer.parseInt(in.readLine());
			String UNK = in.readLine();
			Vocab vocab = Vocab.loadVocab(in);
			SoftmaxObjectiveLayer layer = new SoftmaxObjectiveLayer(vocab, inputDim,UNK);
			layer.inputToVocab = DenseFullyConnectedLayer.load(in);
			layer.dropoutStartId = (int)(0.95*vocab.getTypes());
			return layer;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	public static void main(String[] args){

		Vocab outputVocab = new Vocab();
		outputVocab.addWordToVocab("ola");
		outputVocab.addWordToVocab("mundo");
		outputVocab.sortVocabByCount();
		outputVocab.generateHuffmanCodes();

		SoftmaxObjectiveLayer softmax = new SoftmaxObjectiveLayer(outputVocab, 10, "<unk>");

		GraphInference inference = new GraphInference(0, true);
		DenseNeuronArray[] neurons = DenseNeuronArray.asArray(3, 10);
		inference.addNeurons(0, neurons);
		for(int i = 0; i < 3; i++){
			neurons[i].randInitialize();
		}
		StringNeuronArray[] outputs = StringNeuronArray.asArray(3);
		inference.addNeurons(1, outputs);

		inference.addMapping(new OutputMappingDenseArrayToStringArray(neurons, outputs, softmax));

		inference.init();
		inference.forward();
		StringNeuronArray.setExpectedArray(new String[]{"ola", "ola", "mundo"}, outputs);
		inference.backward();
		inference.commit(0);
		inference.printNeurons();

	}

}
