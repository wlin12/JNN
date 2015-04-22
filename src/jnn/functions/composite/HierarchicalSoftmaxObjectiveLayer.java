package jnn.functions.composite;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.math3.util.FastMath;

import jnn.functions.DenseArrayToStringArrayTransform;
import jnn.functions.DenseToStringTransform;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingDenseToString;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.objective.WordSoftmaxDenseObjective;
import jnn.training.GraphInference;
import util.IOUtils;
import util.TopNList;
import vocab.Vocab;
import vocab.WordEntry;

public class HierarchicalSoftmaxObjectiveLayer extends AbstractSofmaxObjectiveLayer{

	private static final String OUTPUT_KEY = "output";

	Vocab vocab;
	int inputDim;
	String UNK;
	int dropoutStartId;
	
	int numberOfHuffmanNodes;

	DenseFullyConnectedLayer[] inputToNode;	
	HashSet<Integer> usedNodes = new HashSet<Integer>();

	public HierarchicalSoftmaxObjectiveLayer(Vocab vocab, int inputDim, String UNK) {
		this.vocab = vocab;
		this.inputDim = inputDim;
		this.UNK = UNK;

		numberOfHuffmanNodes = vocab.getNumberOfHuffmanNodes();
		inputToNode = new DenseFullyConnectedLayer[numberOfHuffmanNodes];
		for(int i = 0; i < numberOfHuffmanNodes; i++){
			inputToNode[i] = new DenseFullyConnectedLayer(inputDim, vocab.getHuffmanNodeNumberOfChildren(i));
		}
		dropoutStartId = (int)(0.95*vocab.getTypes());
	}

	public DenseNeuronArray[] buildInference(DenseNeuronArray input, int[] nodes, int inputStart, int inputEnd, GraphInference inference){
		inference.addNeurons(0, input);
		DenseNeuronArray[] outputNeurons = new DenseNeuronArray[nodes.length];
		for(int i = 0; i < nodes.length; i++){
			outputNeurons[i] = new DenseNeuronArray(vocab.getHuffmanNodeNumberOfChildren(nodes[i]));
			inference.addNeurons(1, outputNeurons[i]);
			inference.addMapping(new OutputMappingDenseToDense(input, outputNeurons[i], inputToNode[nodes[i]]));
		}

		return outputNeurons;
	}

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		String expected = output.getExpected();
		WordEntry expectedEntry = vocab.getEntry(expected);
		if(expectedEntry==null || (mapping.getSubInference().isTrain() && expectedEntry.id > dropoutStartId && FastMath.random() > 0.5)){
			expectedEntry = vocab.getEntry(UNK);
		}
		int[] nodes = vocab.getHuffmanNodesForEntry(expectedEntry);
		int[] codes = expectedEntry.code;

		DenseNeuronArray[] outputNeurons = buildInference(input,nodes, inputStart, inputEnd, mapping.getSubInference());
		mapping.getSubInference().init();
		mapping.getSubInference().forward();

		mapping.setForwardParam(OUTPUT_KEY, outputNeurons);

		output.setScore(0);
		for(int i = 0; i < codes.length; i++){
			WordSoftmaxDenseObjective objective = new WordSoftmaxDenseObjective(outputNeurons[i], codes[i]);
			if(output.negative){
				objective.addNegativeSoftmaxError(mapping.getSubInference().getNorm());			
			}
			else{
				objective.addSoftmaxError(mapping.getSubInference().getNorm());			
			}
			output.setScore(output.getScore() + objective.getLL());
		}
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			StringNeuronArray[] output, OutputMappingDenseArrayToStringArray mapping) {

		DenseNeuronArray[][] outputNeuronPerPos = new DenseNeuronArray[input.length][];
		int[][] codes = new int[input.length][];
		for(int w = 0; w < input.length; w++){
			String expected = output[w].getExpected();
			WordEntry expectedEntry = vocab.getEntry(expected);
			if(expectedEntry==null || (mapping.getSubInference().isTrain() && expectedEntry.id > dropoutStartId && FastMath.random() > 0.5)){
				expectedEntry = vocab.getEntry(UNK);
			}
			int[] nodes = vocab.getHuffmanNodesForEntry(expectedEntry);
			codes[w] = expectedEntry.code;
			outputNeuronPerPos[w] = buildInference(input[w],nodes, inputStart, inputEnd, mapping.getSubInference());						
		}
		mapping.getSubInference().init();
		mapping.getSubInference().forward();		

		mapping.setForwardParam(OUTPUT_KEY, outputNeuronPerPos);		

		for(int w = 0; w < input.length; w++){
			output[w].setScore(0);
			for(int i = 0; i < codes[w].length; i++){
				WordSoftmaxDenseObjective objective = new WordSoftmaxDenseObjective(outputNeuronPerPos[w][i], codes[w][i]);			
				if(output[w].negative){
					objective.addNegativeSoftmaxError(mapping.getSubInference().getNorm());
				}
				else{
					objective.addSoftmaxError(mapping.getSubInference().getNorm());				
				}
				output[w].setScore(output[w].getScore() + objective.getLL());
			}
		}

	}

	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {		
		mapping.getSubInference().backward();
		String expected = output.getExpected();
		WordEntry expectedEntry = vocab.getEntry(expected);
		int[] nodes = vocab.getHuffmanNodesForEntry(expectedEntry);
		for(int n : nodes){
			usedNodes.add(n);
		}
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, StringNeuronArray[] output,
			OutputMappingDenseArrayToStringArray mapping) {
		for(int w = 0; w < input.length; w++){
			String expected = output[w].getExpected();
			WordEntry expectedEntry = vocab.getEntry(expected);
			int[] nodes = vocab.getHuffmanNodesForEntry(expectedEntry);
			for(int n : nodes){
				usedNodes.add(n);
			}
		}
		mapping.getSubInference().backward();		
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		for(int i : usedNodes){
			inputToNode[i].updateWeights(learningRate, momentum);
		}
	}

	public String decode(DenseNeuronArray input) {
		int nodeId = vocab.getInitialHuffmanNode();
		while(true){
			GraphInference inference = new GraphInference(0, false);
			inference.addNeurons(0, input);
			DenseNeuronArray output = new DenseNeuronArray(vocab.getHuffmanNodeNumberOfChildren(nodeId));
			inference.addNeurons(1,output);
			inference.addMapping(new OutputMappingDenseToDense(input, output, inputToNode[nodeId]));
			inference.init();
			inference.forward();
			int viterbi = output.maxIndex();
			int[] children = vocab.getHuffmanNodeChildren(nodeId);
			int nextNode = children[viterbi];
			if(vocab.isHuffmanNode(nextNode)){
				nodeId = vocab.idToHuffmanNode(nextNode);
			}
			else{
				return vocab.getEntryFromId(nextNode).word;
			}
		}
	}
	
	public TopNList<String> getTopN(DenseNeuronArray input, int n){
		TopNList<String> ret = new TopNList<String>(n);
		ret.add(decode(input), 1);
		return ret;//TODO
	}

	public void save(PrintStream out){
		out.println("word softmax parameters");
		out.println(inputDim);
		out.println(UNK);
		vocab.saveVocab(out);
		for(int i = 0; i < numberOfHuffmanNodes; i++){
			inputToNode[i].save(out);
		}
	}

	public static HierarchicalSoftmaxObjectiveLayer load(BufferedReader in){
		try {
			in.readLine();
			int inputDim = Integer.parseInt(in.readLine());
			String UNK = in.readLine();
			Vocab vocab = Vocab.loadVocab(in);
			HierarchicalSoftmaxObjectiveLayer layer = new HierarchicalSoftmaxObjectiveLayer(vocab, inputDim, UNK);
			for(int i = 0; i < layer.numberOfHuffmanNodes; i++){
				layer.inputToNode[i] = DenseFullyConnectedLayer.load(in);
			}
			return layer;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	

	public static void main(String[] args){

		Vocab outputVocab = new Vocab();
		outputVocab.addWordToVocab("p", 11);
		outputVocab.addWordToVocab("tiger", 10);
		outputVocab.addWordToVocab("monkey", 7);
		outputVocab.addWordToVocab("cat", 3);
		outputVocab.addWordToVocab("dog", 3);
		outputVocab.addWordToVocab("mouse", 2);
		outputVocab.addWordToVocab("spider", 1);
		outputVocab.addWordToVocab("bat", 1);
		outputVocab.addWordToVocab("rhino", 1);
		outputVocab.sortVocabByCount();
		outputVocab.generateHuffmanCodesForNAryTree(3);
		outputVocab.printWordCounts();
		outputVocab.printHuffmanTree();
		HierarchicalSoftmaxObjectiveLayer softmax = new HierarchicalSoftmaxObjectiveLayer(outputVocab, 10, "<unk>");

		for(int e = 0; e < 1000; e++){
			GraphInference inference = new GraphInference(0, true);
			DenseNeuronArray[] neurons = DenseNeuronArray.asArray(3, 10);
			inference.addNeurons(0, neurons);
			for(int i = 0; i < 3; i++){
				neurons[i].randInitialize();
				neurons[i].addNeuron(i, 10);
			}
			StringNeuronArray[] outputs = StringNeuronArray.asArray(3);
			inference.addNeurons(1, outputs);

			inference.addMapping(new OutputMappingDenseArrayToStringArray(neurons, outputs, softmax));
			StringNeuronArray.setExpectedArray(new String[]{"monkey", "rhino", "tiger"}, outputs);

			inference.init();
			inference.forward();
			inference.backward();
			inference.commit(0);
			inference.printNeurons();
			for(int i = 0; i < 3; i++){			
				System.err.println(softmax.decode(neurons[i]));
			}
		}
		
		String file = "/tmp/file";
		PrintStream out = IOUtils.getPrintStream(file);		
		softmax.save(out);
		out.close();
		
		BufferedReader in = IOUtils.getReader(file);
		HierarchicalSoftmaxObjectiveLayer loaded = HierarchicalSoftmaxObjectiveLayer.load(in);
		DenseNeuronArray[] neurons = DenseNeuronArray.asArray(3, 10);
		for(int i = 0; i < 3; i++){
			neurons[i].randInitialize();
			neurons[i].addNeuron(i, 10);
		}
		for(int i = 0; i < 3; i++){			
			System.err.println(loaded.decode(neurons[i]));
		}

	}

}
