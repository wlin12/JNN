package jnn.functions.nlp.app.pretraining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import jnn.functions.composite.FastNegativeSamplingLayer;
import jnn.functions.composite.NegativeSamplingLayer;
import jnn.functions.nlp.words.WordRepresentationLayer;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.neuron.DenseNeuronArray;
import jnn.objective.WordSoftmaxSparseObjective;
import jnn.training.GraphInference;
import vocab.Vocab;
import vocab.WordEntry;

public class StructuredSkipngramInstance {

	int id;
	Vocab outputVocab;
	WordRepresentationLayer wordRep;
	NegativeSamplingLayer softmax;
	FastNegativeSamplingLayer[] fastSoftmax;
	double norm;
	int windowSize;
	double error = 0;

	// sentences
	String[][] batchWindows;

	//WindowWordPairs
	WindowWordPairs wordPairs;
	HashSet<String> words;

	ArrayList<WordSoftmaxSparseObjective> objectives;

	public StructuredSkipngramInstance(int id, String[][] batchWindows, Vocab outputVocab,
			WordRepresentationLayer wordRep, NegativeSamplingLayer softmax, FastNegativeSamplingLayer[] fastSoftmax, int windowSize, double norm) {
		super();
		this.id = id;
		this.batchWindows = batchWindows;
		this.outputVocab = outputVocab;
		this.norm = norm;
		this.windowSize = windowSize;
		this.softmax = softmax;
		this.wordRep = wordRep;
		this.fastSoftmax = fastSoftmax;
	}

	public StructuredSkipngramInstance(int id, WindowWordPairs wordPairs, HashSet<String> words, Vocab outputVocab,
			WordRepresentationLayer wordRep, NegativeSamplingLayer softmax, FastNegativeSamplingLayer[] fastSoftmax, int windowSize, double norm) {
		super();
		this.id = id;
		this.outputVocab = outputVocab;
		this.norm = norm;
		this.windowSize = windowSize;
		this.softmax = softmax;
		this.wordRep = wordRep;
		this.fastSoftmax = fastSoftmax;
		this.words = words;
		this.wordPairs = wordPairs;
	}

	//	public void train(){
	//		buildNetwork(batchWindows);
	//	}
	//
	//	public void buildNetwork(String[][] sentences){
	//		GraphInference inference = new GraphInference(id, false);
	//		inference.setNorm(norm);
	//		HashMap<String, DenseNeuronArray> wordRepresentations = new HashMap<String, DenseNeuronArray>();
	//		HashMap<String, StringNeuronArray> outputWords = new HashMap<String, StringNeuronArray>();
	//
	//		for(String[] window : sentences){
	//			for(int i = 0; i < sent.tokens.length; i++){			
	//				String word = sent.tokens[i];
	//				String wordLc = word.toLowerCase();
	//				WordEntry entry = outputVocab.getEntry(wordLc);
	//
	//				if(entry != null){
	//					if(!outputWords.containsKey(wordLc)){
	//						StringNeuronArray outputNeurons = new StringNeuronArray();
	//						outputNeurons.setExpected(wordLc);
	//						inference.addNeurons(2, outputNeurons);
	//						outputWords.put(wordLc, outputNeurons);
	//					}					
	//				}
	//
	//				if(!wordRepresentations.containsKey(word)){
	//					DenseNeuronArray wordNeurons = new DenseNeuronArray(wordRep.getOutputDim());
	//					inference.addNeurons(1, wordNeurons);
	//					inference.addMapping(new OutputMappingStringToDense(word, wordNeurons, wordRep));
	//					wordRepresentations.put(word, wordNeurons);
	//				}
	//			}
	//		}
	//
	//		for(InputSentence sent : sentences){
	//			String[] words = sent.tokens;
	//			for(int i = 0; i < words.length; i++){
	//				String outputWord = words[i];
	//				String outputWordLc = outputWord.toLowerCase();
	//
	//				if(outputWords.containsKey(outputWordLc)){
	//					StringNeuronArray outputWordNeuron = outputWords.get(outputWordLc);
	//					for(int w = -windowSize; w <=windowSize; w++){
	//						if(w != 0) {
	//							int conditionedWordIndex = i + w;
	//							if(conditionedWordIndex >= 0 && conditionedWordIndex < words.length){
	//								DenseNeuronArray conditionedWordRep = wordRepresentations.get(words[conditionedWordIndex]);
	//								inference.addMapping(new OutputMappingDenseToString(conditionedWordRep, outputWordNeuron, softmax));
	//							}
	//						}
	//					}
	//				}		
	//			}
	//		}
	//
	//		inference.init();
	//		inference.forward();
	//		inference.backward();
	//	}	

	public void trainFast(){
		buildNetworkFast(batchWindows);
	}

	public void buildNetworkFast(String[][] windows){
		GraphInference inference = new GraphInference(id, true);
		inference.setNorm(norm);
		HashMap<String, DenseNeuronArray> wordRepresentations = new HashMap<String, DenseNeuronArray>();

		for(String[] window : windows){
			String word = window[windowSize];
			if(!wordRepresentations.containsKey(word)){
				DenseNeuronArray wordNeurons = new DenseNeuronArray(wordRep.getOutputDim());
				inference.addNeurons(1, wordNeurons);
				wordRepresentations.put(word, wordNeurons);
			}
		}

		String[] wordArray = new String[wordRepresentations.size()];
		DenseNeuronArray[] wordReps = new DenseNeuronArray[wordRepresentations.size()];
		int index = 0;
		for(Entry<String, DenseNeuronArray> entry : wordRepresentations.entrySet()){
			wordArray[index] = entry.getKey();
			wordReps[index] = entry.getValue();
			index++;
		}
		inference.addMapping(new OutputMappingStringArrayToDenseArray(wordArray, wordReps, wordRep));

		inference.init();
		inference.forward();

		for(String[] window : windows){
			DenseNeuronArray center = wordRepresentations.get(window[windowSize]);
			for(int i = 0; i < windowSize*2+1; i++){
				if(i != windowSize){
					String outputWordLc = window[i];
					WordEntry outputWordEntry = outputVocab.getEntry(outputWordLc);
					if(outputWordEntry != null){
						int negativeSamplingIndex = i;
						if(negativeSamplingIndex>windowSize){
							negativeSamplingIndex--;
						}
						fastSoftmax[negativeSamplingIndex].forwardBackward(center, outputWordEntry.id,norm,norm);								
					}
				}
			}			
		}

		inference.backward();
	}

	public void trainWindowFast() {
		GraphInference inference = new GraphInference(id, true);
		inference.setNorm(norm); // doesnt use this in fast inference
		HashMap<String, DenseNeuronArray> wordRepresentations = new HashMap<String, DenseNeuronArray>();

		for(String word : words){
			DenseNeuronArray wordNeurons = new DenseNeuronArray(wordRep.getOutputDim());
			inference.addNeurons(1, wordNeurons);
			wordRepresentations.put(word, wordNeurons);
		}

		String[] wordArray = new String[wordRepresentations.size()];
		DenseNeuronArray[] wordReps = new DenseNeuronArray[wordRepresentations.size()];
		int index = 0;
		for(Entry<String, DenseNeuronArray> entry : wordRepresentations.entrySet()){
			wordArray[index] = entry.getKey();
			wordReps[index] = entry.getValue();
			index++;
		}
		inference.addMapping(new OutputMappingStringArrayToDenseArray(wordArray, wordReps, wordRep));
		inference.init();
		inference.forward();

		for(String s : words){
			DenseNeuronArray rep = wordRepresentations.get(s);			
			for(int w = -windowSize; w <= windowSize; w++){
				if(w != 0){
					index = w + windowSize;
					if(w > 0){
						index--;
					}
					HashMap<String, Integer> wordCounts = wordPairs.pairsPerWindow[index].getWordsFor(s);
					double total = 0;
					for(Entry<String,Integer> entry : wordCounts.entrySet()){
						total += entry.getValue();
					}

					for(Entry<String,Integer> entry : wordCounts.entrySet()){
						//System.err.println(w+ "->"+entry.getValue()/(total * 2 * windowSize));
						//System.err.println(fastSoftmax[index].getWeight(outputVocab.getEntry(entry.getKey()).id, 0));
						WordEntry outputEntry = outputVocab.getEntry(entry.getKey());
						if(outputEntry!=null){
							fastSoftmax[index].forwardBackward(rep, outputEntry.id,norm * entry.getValue()/total, entry.getValue()/(total * 2 * windowSize));
						}
					}
				}
			}
			//System.err.println(rep);
			wordPairs.remove(s);
		}
		inference.backward();

	}

	public void error() {
		error(batchWindows);
	}

	public void error(String[][] windows){
		GraphInference inference = new GraphInference(id, true);
		HashMap<String, DenseNeuronArray> wordRepresentations = new HashMap<String, DenseNeuronArray>();

		for(String[] window : windows){
			String word = window[windowSize];
			if(!wordRepresentations.containsKey(word)){
				DenseNeuronArray wordNeurons = new DenseNeuronArray(wordRep.getOutputDim());
				inference.addNeurons(1, wordNeurons);
				wordRepresentations.put(word, wordNeurons);
			}
		}

		String[] wordArray = new String[wordRepresentations.size()];
		DenseNeuronArray[] wordReps = new DenseNeuronArray[wordRepresentations.size()];
		int index = 0;
		for(Entry<String, DenseNeuronArray> entry : wordRepresentations.entrySet()){
			wordArray[index] = entry.getKey();
			wordReps[index] = entry.getValue();
			index++;
		}
		inference.addMapping(new OutputMappingStringArrayToDenseArray(wordArray, wordReps, wordRep));

		inference.init();
		inference.forward();

		for(String[] window : windows){
			DenseNeuronArray center = wordRepresentations.get(window[windowSize]);
			for(int i = 0; i < windowSize*2+1; i++){
				if(i != windowSize){
					String outputWordLc = window[i];
					WordEntry outputWordEntry = outputVocab.getEntry(outputWordLc);
					if(outputWordEntry != null){
						int negativeSamplingIndex = i;
						if(negativeSamplingIndex>windowSize){
							negativeSamplingIndex--;
						}
						error += fastSoftmax[negativeSamplingIndex].error(center, outputWordEntry.id);								
					}
				}
			}			
		}
	}
}
