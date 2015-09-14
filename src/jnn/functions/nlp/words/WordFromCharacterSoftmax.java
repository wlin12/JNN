package jnn.functions.nlp.words;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import util.LangUtils;
import util.TopNList;
import vocab.Vocab;
import vocab.WordEntry;
import jnn.decoder.DecoderInterface;
import jnn.decoder.state.DecoderState;
import jnn.functions.DenseArrayToStringArrayTransform;
import jnn.functions.DenseToStringTransform;
import jnn.functions.composite.AbstractSofmaxObjectiveLayer;
import jnn.functions.composite.HierarchicalSoftmaxObjectiveLayer;
import jnn.functions.composite.LookupTable;
import jnn.functions.composite.NoiseConstrastiveEstimationLayer;
import jnn.functions.composite.SoftmaxObjectiveLayer;
import jnn.functions.composite.lstm.LSTMDecoder;
import jnn.functions.composite.lstm.LSTMDecoderState;
import jnn.functions.composite.lstm.LSTMDecoderWithAlignmentState;
import jnn.functions.composite.lstm.aux.LSTMMapping;
import jnn.functions.nonparametrized.CopyLayer;
import jnn.functions.parametrized.StaticLayer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingDenseToString;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.mapping.OutputMappingVoidToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.training.GraphInference;

public class WordFromCharacterSoftmax extends AbstractSofmaxObjectiveLayer implements DenseToStringTransform, DenseArrayToStringArrayTransform{

	private static final String OUTPUT_KEY = "output_key";
	private static final String STATE_KEY = "state_key";
	
	Vocab charVocab;

	LSTMDecoder decoder;
	LookupTable letterProjection;
	StaticLayer initialStateLayer;
	StaticLayer initialCellLayer;	
	AbstractSofmaxObjectiveLayer letterPredictionLayer;

	int externalInputDim;
	int letterDim;
	int stateDim;
	String SOS = "<s>";
	String EOS = "</s>";
	String UNK = "<unk>";
	int dropoutStartId = -1;
	int beamSize = 20;
	int softmaxType = 0; //0-> softmax 1-> hierarchical softmax 2-> nce softmax
	
	public static class SequenceWordState extends LSTMDecoderWithAlignmentState{
		String outputString;
		int numberOfWords;
		public SequenceWordState(double score, boolean isFinal,
				DenseNeuronArray output, DenseNeuronArray lstmState,
				DenseNeuronArray lstmCell, DenseNeuronArray input,
				String outputString, int numberOfWords) {
			super(score, isFinal, output, lstmState, lstmCell, input);
			this.outputString = outputString;
			this.numberOfWords = numberOfWords;
		}

		public String[] getOutput(){
			if(outputString.length() <=1){
				return new String[0];
			}
			return outputString.substring(1, outputString.length()).split("\\s+");
		}
	}

	private WordFromCharacterSoftmax() {
	}

	public WordFromCharacterSoftmax(Vocab wordVocab, int softmaxType, int externalInputDim, int letterDim, int stateDim, int samplingRate) {
		this.externalInputDim = externalInputDim;
		this.letterDim = letterDim;
		this.stateDim = stateDim;
		this.softmaxType = softmaxType;

		decoder = new LSTMDecoder(externalInputDim + letterDim, stateDim);
		initialStateLayer = new StaticLayer(stateDim);
		initialCellLayer = new StaticLayer(stateDim);
		charVocab = new Vocab();
		for(int wordId = 0; wordId < wordVocab.getTypes(); wordId++){
			String word = wordVocab.getEntryFromId(wordId).word;
			String[] chars = LangUtils.splitWord(word);
			for(int c = 0; c < chars.length; c++){
				charVocab.addWordToVocab(chars[c], wordVocab.getEntryFromId(wordId).count);
			}
		}
		charVocab.addWordToVocab(SOS, 2);
		charVocab.addWordToVocab(EOS, wordVocab.getTokens());
		charVocab.addWordToVocab(UNK, 2);
		charVocab.sortVocabByCount();
		charVocab.generateHuffmanCodes();
		dropoutStartId = (int)(0.95*charVocab.getTypes());
		letterProjection = new LookupTable(charVocab, letterDim);
		if(softmaxType == 0){
			letterPredictionLayer = new SoftmaxObjectiveLayer(charVocab, stateDim, UNK);
		}
		else if(softmaxType == 1){
			letterPredictionLayer = new HierarchicalSoftmaxObjectiveLayer(charVocab, stateDim, UNK);
		}
		else if(softmaxType == 2){
			letterPredictionLayer = new NoiseConstrastiveEstimationLayer(stateDim, charVocab, samplingRate, UNK);
		}
		else {
			throw new RuntimeException("unknown softmax");
		}
	}

	public DenseNeuronArray[] buildLetterProjections(String[] letters, DenseNeuronArray input, GraphInference inference){
		String[] lettersWithUnks = new String[letters.length];
		for(int i = 0; i < letters.length; i++){
			WordEntry entry = charVocab.getEntry(letters[i]);
			if(entry == null){
				lettersWithUnks[i] = UNK;
			}
			else{
				lettersWithUnks[i] = letters[i];
			}
		}
		DenseNeuronArray[] letterAndInputProjections = DenseNeuronArray.asArray(lettersWithUnks.length, letterDim + externalInputDim);
		DenseNeuronArray.asArray(lettersWithUnks.length, letterDim+externalInputDim, "letter projections");
		inference.addNeurons(letterAndInputProjections);
		inference.addMapping(new OutputMappingStringArrayToDenseArray(lettersWithUnks, 0, letterDim-1,letterAndInputProjections, letterProjection));
		for(int i = 0; i < letterAndInputProjections.length; i++){
			inference.addMapping(new OutputMappingDenseToDense(0,externalInputDim-1,letterDim, externalInputDim+letterDim-1,input, letterAndInputProjections[i], CopyLayer.singleton));
		}
		return letterAndInputProjections;
	}

	public DenseNeuronArray[] buildLetterProjectionsAndInit(String[] letters, DenseNeuronArray input){
		GraphInference inference = new GraphInference(0, false); 
		inference.addNeurons(0, input);		
		DenseNeuronArray[] ret = buildLetterProjections(letters, input, inference);
		inference.init();
		inference.forward();
		return ret;
	}

	public StringNeuronArray[] buildInference(DenseNeuronArray input, String word, Mapping mapping){
		GraphInference inference = mapping.getSubInference();
		inference.addNeurons(0, input);
		String[] chars = LangUtils.splitWord(word);
		String[] outputWithSOS = new String[chars.length + 1];

		for(int i = 0; i < chars.length; i++){
			outputWithSOS[i+1] = chars[i];
		}
		outputWithSOS[0] = SOS;
		
		DenseNeuronArray[] letterAndInputProjections = buildLetterProjections(outputWithSOS, input, inference);

		//lstm encoding -> lstm decoding
		LSTMDecoderState[] states = LSTMDecoderState.buildStateSequence(letterAndInputProjections, stateDim);
		DenseNeuronArray[] lstmStates = LSTMDecoderState.getStates(states); 
		DenseNeuronArray[] lstmCells = LSTMDecoderState.getCells(states);
		DenseNeuronArray initialState = new DenseNeuronArray(stateDim);
		DenseNeuronArray initialCell = new DenseNeuronArray(stateDim);
		inference.addNeurons(lstmCells);
		inference.addNeurons(lstmStates);
		inference.addNeurons(initialState);
		inference.addNeurons(initialCell);
		inference.addMapping(new OutputMappingVoidToDense(initialState, initialStateLayer));
		inference.addMapping(new OutputMappingVoidToDense(initialCell, initialCellLayer));

		inference.addMapping(new LSTMMapping(states, initialState, initialCell, decoder));
		StringNeuronArray[] outputs = StringNeuronArray.asArray(states.length);
		inference.addNeurons(outputs);
		OutputMappingDenseArrayToStringArray predictionMapping = new OutputMappingDenseArrayToStringArray(lstmStates, outputs, letterPredictionLayer);
		inference.addMapping(predictionMapping);

		String[] expectedWordChars = LangUtils.splitWord(word);
		for(int i = 0; i < expectedWordChars.length; i++){
			String c = String.valueOf(expectedWordChars[i]);
			WordEntry entry = charVocab.getEntry(c); 
			if(entry == null){
				outputs[i].setExpected(UNK);
			}
			else{
				outputs[i].setExpected(c);
			}
		}
		outputs[expectedWordChars.length].setExpected(EOS);		
		mapping.setForwardParam(STATE_KEY+word, states);
		predictionMapping.getSubInference().setNorm(inference.getNorm()/outputs.length);
		return outputs;
	}

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		StringNeuronArray[] outputNeurons = buildInference(input, output.getExpected(), mapping);
		mapping.getSubInference().init();
		mapping.getSubInference().forward();
		mapping.setForwardParam(OUTPUT_KEY, outputNeurons);
		String word = "";
		for(int i = 0; i < outputNeurons.length-1; i++){
			word+=outputNeurons[i].getOutput();
		}
		if(!outputNeurons[outputNeurons.length-1].getOutput().equals(EOS)){
			word+="|x|";
		}
		output.setOutput(word);
		double score = 0;
		for(int i = 0; i < outputNeurons.length; i++){			
			score += outputNeurons[i].getScore();
		}
		output.setScore(score);
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			StringNeuronArray[] output,
			OutputMappingDenseArrayToStringArray mapping) {
		StringNeuronArray[][] outputNeuronsPerPos = new StringNeuronArray[input.length][];
		for(int pos = 0; pos < input.length; pos++){
			outputNeuronsPerPos[pos] = buildInference(input[pos], output[pos].getExpected(), mapping);
		}
		mapping.getSubInference().init();
		mapping.getSubInference().forward();
		for(int pos = 0; pos < input.length; pos++){
			String word = "";
			for(int i = 0; i < outputNeuronsPerPos[pos].length-1; i++){
				word+=outputNeuronsPerPos[pos][i].getOutput();
			}
			output[pos].setOutput(word);
			if(!outputNeuronsPerPos[pos][outputNeuronsPerPos[pos].length-1].getOutput().equals(EOS)){
				word+="|x|";
			}
//			System.err.println("train-word:" + word);
//			System.err.println("train:input" + input[pos]);
//			
//			LSTMDecoderState[] states = ((LSTMDecoderState[])mapping.getForwardParam(STATE_KEY+output[pos].getExpected()));
//			for(int i = 0; i < states.length; i++){
//				System.err.println("train-input:"+states[i].input);
//				System.err.println("train-state:"+states[i].lstmState);
//			}
		}
			
		mapping.setForwardParam(OUTPUT_KEY, outputNeuronsPerPos);
		for(int pos = 0; pos < output.length; pos++){
			double score = 0;
			for(int i = 0; i < outputNeuronsPerPos[pos].length; i++){			
				score += outputNeuronsPerPos[pos][i].getScore();
			}
			output[pos].setScore(score);
		}
	}

	public void backward(DenseNeuronArray input, int inputStart, 
			int inputEnd, StringNeuronArray output, OutputMappingDenseToString mapping) {
		//		StringNeuronArray[] outputNeurons = (StringNeuronArray[]) mapping.getForwardParam(OUTPUT_KEY);
		mapping.getSubInference().backward();		


	};

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, StringNeuronArray[] output,
			OutputMappingDenseArrayToStringArray mapping) {
		//		StringNeuronArray[][] outputNeuronsPerPos = (StringNeuronArray[][]) mapping.getForwardParam(OUTPUT_KEY);
		//		for(int pos = 0; pos < input.length; pos++){
		//			String[] expectedWordChars = LangUtils.splitWord(output[pos].getExpected());
		//			for(int i = 0; i < expectedWordChars.length; i++){			
		//				outputNeuronsPerPos[pos][i].setExpected(expectedWordChars[i]);
		//			}
		//			outputNeuronsPerPos[pos][expectedWordChars.length].setExpected(EOS);
		//		}
		mapping.getSubInference().backward();
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		decoder.updateWeights(learningRate, momentum);
		letterProjection.updateWeights(learningRate, momentum);
		initialStateLayer.updateWeights(learningRate, momentum);
		initialCellLayer.updateWeights(learningRate, momentum);	
		letterPredictionLayer.updateWeights(learningRate, momentum);
	}

	public String decode(DenseNeuronArray input){
		return decode(input, beamSize);
	}
	
	@Override
	public TopNList<String> getTopN(DenseNeuronArray input, int n) {
		return decode(input, beamSize, n);
	}
	
	public String decode(DenseNeuronArray input, int beam){
		GraphInference inference = new GraphInference(0, false);
		DenseNeuronArray initialState = new DenseNeuronArray(stateDim);
		DenseNeuronArray initialCell = new DenseNeuronArray(stateDim);

		inference.addNeurons(initialState);
		inference.addNeurons(initialCell);
		inference.addMapping(new OutputMappingVoidToDense(initialState, initialStateLayer));
		inference.addMapping(new OutputMappingVoidToDense(initialCell, initialCellLayer));

		inference.init();
		inference.forward();

		//lstm encoding -> lstm decoding
		SequenceWordState initialDecoderState = new SequenceWordState(0, false, initialState, initialState, initialCell, buildLetterProjectionsAndInit(new String[]{SOS}, input)[0], "", 0);
		initialDecoderState.name = "initial";
		SequenceWordState finalState = (SequenceWordState) decoder.decode(initialDecoderState,beam,new DecoderInterface() {

			@Override
			public List<DecoderState> expand(DecoderState state) {
				DenseNeuronArray output = ((SequenceWordState) state).output;

				double score = ((SequenceWordState) state).score;
				int numberOfWords = ((SequenceWordState) state).numberOfWords;
				String outputStr = ((SequenceWordState) state).outputString;
				TopNList<String> topn = letterPredictionLayer.getTopN(output, beam);
				LinkedList<DecoderState> ret = new LinkedList<DecoderState>();
				Iterator<Double> scoreIt = topn.getObjectScore().iterator();
				Iterator<String> topnIt = topn.getObjectList().iterator();
				while(topnIt.hasNext()){					
					double wordScore = scoreIt.next();
					String outputWord = topnIt.next();

					if(outputWord.equals(SOS)) continue;
					boolean isFinal = outputWord.equals(EOS);

//					double newScore = ((score * numberOfWords) + wordScore)/(numberOfWords+1);
					double newScore = score + wordScore;
					String newString = outputStr;
					if(!isFinal){
						newString =  outputStr + outputWord;
					}
					SequenceWordState nextState = new SequenceWordState(newScore, isFinal, null, null, null, buildLetterProjectionsAndInit(new String[]{outputWord},input)[0], newString, numberOfWords+1);
					nextState.setPrevState(state);
					nextState.name = newString;
					ret.add(nextState);
				}
				return ret;
			}
		});
		if(finalState == null){
			return "could_not_find_path";
		}
		System.err.println(finalState.outputString);
		return finalState.outputString;
	}

	public void save(PrintStream out){
		out.println(externalInputDim);
		out.println(letterDim);
		out.println(stateDim);
		out.println(SOS);
		out.println(EOS);
		out.println(dropoutStartId);
		out.println(softmaxType);
		
		decoder.save(out);
		letterProjection.save(out);
		initialStateLayer.save(out);
		initialCellLayer.save(out);	
		letterPredictionLayer.save(out);
		charVocab.saveVocab(out);
	}

	public static WordFromCharacterSoftmax load(BufferedReader in){
		try {
			WordFromCharacterSoftmax softmax = new WordFromCharacterSoftmax();

			softmax.externalInputDim = Integer.parseInt(in.readLine());
			softmax.letterDim = Integer.parseInt(in.readLine());
			softmax.stateDim = Integer.parseInt(in.readLine());
			softmax.SOS = in.readLine();
			softmax.EOS = in.readLine();
			softmax.dropoutStartId = Integer.parseInt(in.readLine());
			softmax.softmaxType = Integer.parseInt(in.readLine());
			softmax.decoder = LSTMDecoder.load(in);
			softmax.letterProjection = LookupTable.load(in);
			softmax.initialStateLayer = StaticLayer.load(in);
			softmax.initialCellLayer = StaticLayer.load(in);
			if(softmax.softmaxType == 0){
				softmax.letterPredictionLayer = SoftmaxObjectiveLayer.load(in);
			}
			else if(softmax.softmaxType == 1){
				softmax.letterPredictionLayer = HierarchicalSoftmaxObjectiveLayer.load(in);
			}
			else if(softmax.softmaxType == 2){
				softmax.letterPredictionLayer = NoiseConstrastiveEstimationLayer.load(in);
			}
			else {
				throw new RuntimeException("unknown softmax");
			}
			softmax.charVocab = Vocab.loadVocab(in);
			return softmax;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public TopNList<String> decode(DenseNeuronArray input, int beam, int topN){
		GraphInference inference = new GraphInference(0, false);
		DenseNeuronArray initialState = new DenseNeuronArray(stateDim);
		DenseNeuronArray initialCell = new DenseNeuronArray(stateDim);

		inference.addNeurons(initialState);
		inference.addNeurons(initialCell);
		inference.addMapping(new OutputMappingVoidToDense(initialState, initialStateLayer));
		inference.addMapping(new OutputMappingVoidToDense(initialCell, initialCellLayer));

		inference.init();
		inference.forward();

		//lstm encoding -> lstm decoding
		SequenceWordState initialDecoderState = new SequenceWordState(1, false, initialState, initialState, initialCell, buildLetterProjectionsAndInit(new String[]{SOS}, input)[0], "", 0);
//		System.err.println(initialDecoderState.lstmState);
		LSTMDecoderState[] finalStates = decoder.decode(initialDecoderState,beam, topN,new DecoderInterface() {

			@Override
			public List<DecoderState> expand(DecoderState state) {
				DenseNeuronArray output = ((SequenceWordState) state).output;

				double score = ((SequenceWordState) state).score;
				int numberOfWords = ((SequenceWordState) state).numberOfWords;
				String outputStr = ((SequenceWordState) state).outputString;
				TopNList<String> topn = letterPredictionLayer.getTopN(output, beam);
				LinkedList<DecoderState> ret = new LinkedList<DecoderState>();
				Iterator<Double> scoreIt = topn.getObjectScore().iterator();
				Iterator<String> wordIt = topn.getObjectList().iterator();
				while(scoreIt.hasNext()){
					String outputWord = wordIt.next();

					double wordScore = scoreIt.next();
					if(outputWord.equals(SOS)) continue;
					boolean isFinal = outputWord.equals(EOS);					
//					double newScore = ((score * numberOfWords) + wordScore)/(numberOfWords+1);
					double newScore = score + wordScore;
					String newString = outputStr;
					if(!isFinal){
						newString =  outputStr + outputWord;
					}
					SequenceWordState nextState = new SequenceWordState(newScore, isFinal, null, null, null, buildLetterProjectionsAndInit(new String[]{outputWord},input)[0], newString, numberOfWords+1);
					nextState.setPrevState(state);
//					System.err.println(((SequenceWordState) state).input);
//					System.err.println(((SequenceWordState) state).lstmState);
					nextState.name = newString;
					ret.add(nextState);
				}
				return ret;
			}
		});
		if(finalStates.length == 0){
			TopNList<String> ret = new TopNList<String>(topN);
			ret.add("could_not_find_path", 0);
			return ret;
		}
		TopNList<String> ret = new TopNList<String>(topN);
		for(int k = 0; k < finalStates.length; k++){
			SequenceWordState state = (SequenceWordState)finalStates[k];
			ret.add(state.outputString, state.score);
		}
		//System.err.println("test:inf"+input+"->"+ret);
		return ret;
	}	
	
	public static void main(String[] args){
		Vocab wordVocab = new Vocab();
		wordVocab.addWordToVocab("doggy",2);
		wordVocab.addWordToVocab("dog",2);
		wordVocab.addWordToVocab("kitten",3);
		wordVocab.addWordToVocab("cat",2);
		wordVocab.addWordToVocab("rat",1);
		
		int inputSize = 50;
		WordFromCharacterSoftmax softmax = new WordFromCharacterSoftmax(wordVocab, 2, inputSize, 50, 150,5);
		
		DenseNeuronArray ratVector = new DenseNeuronArray(inputSize);
		DenseNeuronArray catVector = new DenseNeuronArray(inputSize);
		DenseNeuronArray kittenVector = new DenseNeuronArray(inputSize);
		DenseNeuronArray dogVector = new DenseNeuronArray(inputSize);
		DenseNeuronArray doggyVector = new DenseNeuronArray(inputSize);
		ratVector.randInitialize();
		catVector.randInitialize();
		kittenVector.randInitialize();
		dogVector.randInitialize();
		doggyVector.randInitialize();
		
		for(int e = 0; e < 1000; e++){
			GraphInference inference = new GraphInference(0, true);
			inference.addNeurons(ratVector);
			inference.addNeurons(catVector);
			inference.addNeurons(kittenVector);
			inference.addNeurons(dogVector);
			inference.addNeurons(doggyVector);
			StringNeuronArray ratOutput = new StringNeuronArray();
			StringNeuronArray catOutput = new StringNeuronArray();
			StringNeuronArray kittenOutput = new StringNeuronArray();
			StringNeuronArray dogOutput = new StringNeuronArray();
			StringNeuronArray doggyOutput = new StringNeuronArray();
			ratOutput.setExpected("rat");
			catOutput.setExpected("cat");
			kittenOutput.setExpected("kitten");
			dogOutput.setExpected("dog");
			doggyOutput.setExpected("doggy");
			inference.addNeurons(ratOutput);
			inference.addNeurons(catOutput);
			inference.addNeurons(kittenOutput);
			inference.addNeurons(dogOutput);
			inference.addNeurons(doggyOutput);
			inference.addMapping(new OutputMappingDenseArrayToStringArray(new DenseNeuronArray[]{ratVector, catVector, kittenVector, dogVector, doggyVector}, 
					new StringNeuronArray[]{ratOutput, catOutput, kittenOutput, dogOutput, doggyOutput}, softmax));
			inference.init();
			inference.forward();
			inference.backward();
			inference.commit(0);
			inference.printNeurons();
		}
		
		GraphInference inference = new GraphInference(0, false);
		inference.addNeurons(ratVector);
		inference.addNeurons(catVector);
		inference.addNeurons(kittenVector);
		inference.addNeurons(dogVector);
		inference.addNeurons(doggyVector);

		StringNeuronArray ratOutput = new StringNeuronArray();
		StringNeuronArray catOutput = new StringNeuronArray();
		StringNeuronArray kittenOutput = new StringNeuronArray();
		StringNeuronArray dogOutput = new StringNeuronArray();
		StringNeuronArray doggyOutput = new StringNeuronArray();
		ratOutput.setExpected("rat");
		catOutput.setExpected("cat");
		kittenOutput.setExpected("kitten");
		dogOutput.setExpected("dog");
		doggyOutput.setExpected("doggy");
		inference.addNeurons(ratOutput);
		inference.addNeurons(catOutput);
		inference.addNeurons(kittenOutput);
		inference.addNeurons(dogOutput);
		inference.addNeurons(doggyOutput);
		inference.addMapping(new OutputMappingDenseArrayToStringArray(new DenseNeuronArray[]{ratVector, catVector, kittenVector, dogVector, doggyVector}, 
				new StringNeuronArray[]{ratOutput, catOutput, kittenOutput, dogOutput, doggyOutput}, softmax));
		inference.init();
		inference.forward();
		inference.printNeurons();
		System.err.println(softmax.decode(ratVector,5,5));
//		System.err.println(softmax.decode(catVector));
//		System.err.println(softmax.decode(kittenVector));
//		System.err.println(softmax.decode(dogVector));
//		System.err.println(softmax.decode(doggyVector));
	}
}
