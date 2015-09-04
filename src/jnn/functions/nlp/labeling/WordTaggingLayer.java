package jnn.functions.nlp.labeling;

import java.util.ArrayList;
import java.util.LinkedList;

import util.LangUtils;
import vocab.Vocab;
import jnn.functions.StringArrayToStringArrayTransform;
import jnn.functions.composite.SoftmaxObjectiveLayer;
import jnn.functions.nlp.words.WordRepresentationLayer;
import jnn.functions.nlp.words.WordWithContextRepresentation;
import jnn.functions.nlp.words.features.CapitalizationWordFeatureExtractor;
import jnn.functions.nlp.words.features.CharSequenceExtractor;
import jnn.functions.nlp.words.features.LowercasedWordFeatureExtractor;
import jnn.functions.nlp.words.features.WordRepresentationSetup;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.mapping.OutputMappingStringArrayToStringArray;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.training.GlobalParameters;
import jnn.training.GraphInference;

public class WordTaggingLayer extends Layer implements StringArrayToStringArrayTransform{	
	
	
	WordRepresentationLayer wordLayer;
	WordWithContextRepresentation wordToContextLayer;
	SoftmaxObjectiveLayer contextToPOSLayer;	

	public WordTaggingLayer(WordRepresentationLayer wordLayer,
			WordWithContextRepresentation wordToContextLayer,
			SoftmaxObjectiveLayer contextToPOSLayer) {
		super();
		this.wordLayer = wordLayer;
		this.wordToContextLayer = wordToContextLayer;
		this.contextToPOSLayer = contextToPOSLayer;
	}

	public StringNeuronArray[] buildInference(String[] input, StringNeuronArray[] output, GraphInference inference){
		DenseNeuronArray[] wordProjections = DenseNeuronArray.asArray(input.length, wordLayer.getOutputDim());
		DenseNeuronArray[] contextVectors = DenseNeuronArray.asArray(input.length, wordToContextLayer.getOutputDim());
		
		inference.addNeurons(1, wordProjections);
		inference.addMapping(new OutputMappingStringArrayToDenseArray(input, wordProjections, wordLayer));
		inference.addNeurons(2, contextVectors);
		inference.addMapping(new OutputMappingDenseArrayToDenseArray(wordProjections, contextVectors, wordToContextLayer));
		inference.addNeurons(3, output);
		inference.addMapping(new OutputMappingDenseArrayToStringArray(contextVectors, output, contextToPOSLayer));
		
		inference.init();
		inference.forward();
		return output;
	}
	
	@Override
	public void forward(String[] input, StringNeuronArray[] output,
			OutputMappingStringArrayToStringArray mapping) {
		
		buildInference(input, output, mapping.getSubInference());
	}
	
	@Override
	public void backward(String[] input, StringNeuronArray[] output,
			OutputMappingStringArrayToStringArray mapping) {
		mapping.getSubInference().backward();
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		wordLayer.updateWeights(learningRate, momentum);
		wordToContextLayer.updateWeights(learningRate, momentum);
		contextToPOSLayer.updateWeights(learningRate, momentum);		
	}
	
	public void updateWeights(double learningRate, double momentum, boolean updateRep) {
		if(updateRep){
			wordLayer.updateWeights(learningRate, momentum);
		}
		wordToContextLayer.updateWeights(learningRate, momentum);
		contextToPOSLayer.updateWeights(learningRate, momentum);		
	}
	
	public static void main(String[] args){
		GlobalParameters.useMomentumDefault = true;
		GlobalParameters.learningRateDefault = 0.01;
		
		String[] inputSents = new String[]{"I want to play Hearthstone , but I need to make this code work first .",
				"Working in voice interaction is much better than working at Google , right Tiago ?",
				"Audimus is the best thing ever invented in the history of Mankind !"};
		
		//convert input into labelled data
		String[][] inputSentsWithoutPunctuation = new String[inputSents.length][];
		String[][] punctuationLabels = new String[inputSents.length][];
		for(int i = 0; i < inputSents.length; i++){
			LinkedList<String> tokens = new LinkedList<String>();
			LinkedList<String> labels = new LinkedList<String>();
			String[] inputTokens = inputSents[i].split("\\s+");
			for(int w = 0; w < inputTokens.length; w++){
				if(LangUtils.isPunct(inputTokens[w])){
					labels.removeLast();
					labels.addLast(inputTokens[w]);
				}
				else{
					labels.addLast("nopunct");
					tokens.addLast(inputTokens[w]);
				}
			}
			inputSentsWithoutPunctuation[i] = tokens.toArray(new String[0]);
			punctuationLabels[i] = labels.toArray(new String[0]);
		}		
		
		// loading vocabs
		Vocab inputVocab = new Vocab();		
		Vocab outputVocab = new Vocab();
		
		for(int i = 0; i < inputSents.length; i++){
			String[] input = inputSentsWithoutPunctuation[i];
			String[] output = punctuationLabels[i];
			
			for(int w = 0; w < input.length; w++){
				inputVocab.addWordToVocab(input[w],1);
				outputVocab.addWordToVocab(output[w],1);
			}
		}
		
		inputVocab.sortVocabByCount();
		inputVocab.generateHuffmanCodes();
		outputVocab.sortVocabByCount();
		outputVocab.generateHuffmanCodes();

		// building word representation layer
		int wordProjectionSize = 50;
		int charProjectionSize = 50;
		int charStateSize = 150;
		
		WordRepresentationSetup setup = new WordRepresentationSetup(inputVocab, wordProjectionSize, charProjectionSize, charStateSize);
		setup.addFeatureExtractor(new LowercasedWordFeatureExtractor());
		setup.addFeatureExtractor(new CapitalizationWordFeatureExtractor());
		//setup.addSequenceExtractor(new CharSequenceExtractor());
		WordRepresentationLayer wordRepLayer = new WordRepresentationLayer(setup);

		// building context layer
		int contextStateSize = 50;
		WordWithContextRepresentation contextRepLayer = new WordWithContextRepresentation(wordRepLayer.getOutputDim(), contextStateSize);				
		contextRepLayer.setBLSTMModel();
		//contextRepLayer.setWindowMode(5);
		
		// building softmax layer
		SoftmaxObjectiveLayer labelSoftmaxLayer = new SoftmaxObjectiveLayer(outputVocab, contextStateSize, "<unk>");
		
		// building tagger
		WordTaggingLayer tagger = new WordTaggingLayer(wordRepLayer, contextRepLayer, labelSoftmaxLayer);
		for(int e = 0; e < 1000; e++){
			for(int i = 0; i < inputSents.length; i++){
				String[] input = inputSentsWithoutPunctuation[i];
				String[] output = punctuationLabels[i];
				
				StringNeuronArray[] predicted = StringNeuronArray.asArray(input.length);
				GraphInference inference = new GraphInference(0, true);
				inference.setNorm(1/(double)input.length);
				inference.addNeurons(1,predicted);
				inference.addMapping(new OutputMappingStringArrayToStringArray(input, predicted, tagger));
				inference.init();
				inference.forward();
				StringNeuronArray.setExpectedArray(output, predicted);
				inference.backward();
				inference.commit(0);
				inference.printNeurons();
				
			}
		}
		
	}
}
