package jnn.functions.nlp.words;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import jnn.functions.StringArrayToDenseArrayTransform;
import jnn.functions.StringToDenseTransform;
import jnn.functions.composite.DeepRNN;
import jnn.functions.composite.LookupTable;
import jnn.functions.nlp.words.features.CapitalizationWordFeatureExtractor;
import jnn.functions.nlp.words.features.CharSequenceExtractor;
import jnn.functions.nlp.words.features.DoubleFeatureExtractor;
import jnn.functions.nlp.words.features.LowercasedWordFeatureExtractor;
import jnn.functions.nlp.words.features.WordRepresentationSetup;
import jnn.functions.nonparametrized.CopyLayer;
import jnn.functions.parametrized.Layer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.mapping.OutputMappingStringToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.threading.SharedDenseNeuronArray;
import jnn.training.GraphInference;
import jnn.wordsim.WordVectors;
import util.ArrayUtils;
import util.IOUtils;
import util.RandomUtils;
import util.SerializeUtils;
import vocab.Vocab;
import vocab.WordEntry;

public class WordRepresentationLayer extends Layer implements StringToDenseTransform, StringArrayToDenseArrayTransform{

	private class WordRepresentationCacheEntry{
		String input;
		GraphInference inference;
		SharedDenseNeuronArray representation;

		public WordRepresentationCacheEntry(String input,
				GraphInference inference) {
			super();
			this.input = input;
			this.inference = inference;
		}

		public void load(){
			this.representation = new SharedDenseNeuronArray(buildWordRepresentation(input, inference));
			inference.init();
			inference.forward();
			if(!inference.isTrain()){
				inference = null;
			}
		}

		public void backpropagate(){
			representation.mergeCopies();			
			this.inference.backward();
		}

		public DenseNeuronArray getOutput(){
			return representation.getCopy();
		}
	}

	private class WordRepresentationCache{
		public HashMap<String, WordRepresentationCacheEntry> cache = new HashMap<String, WordRepresentationLayer.WordRepresentationCacheEntry>();
		ArrayList<WordRepresentationCacheEntry>[] wordsPerThread;
		int hits=0;
		int misses=0;
		
		public long fill(Set<String> words, int threads, boolean train){
			long time = System.currentTimeMillis();
			if(cache.size() > 0){
				throw new RuntimeException("cache already filled with " + cache.size() + " elements");
			}
			threads = Math.min(words.size(), threads);
			wordsPerThread = new ArrayList[threads];
			for(int i = 0; i < threads; i++){
				wordsPerThread[i] = new ArrayList<WordRepresentationCacheEntry>();
			}
			int i = 0;
			for(String word : words){
				WordRepresentationCacheEntry entry = new WordRepresentationCacheEntry(word, new GraphInference(i, train));
				wordsPerThread[i].add(entry);
				cache.put(word, entry);
				i = (i+1)%threads;
			}

			Thread[] cacheThreads = new Thread[threads];
			for(int t = 0; t < threads; t++){
				int finalT = t;
				cacheThreads[t] = new Thread(){
					public void run() {
						for(WordRepresentationCacheEntry entry : wordsPerThread[finalT]){
							entry.load();
						}
					};
				};
				cacheThreads[t].start();
			}
			for(int t = 0; t < threads; t++){
				try {
					cacheThreads[t].join();
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				}
			}
			return System.currentTimeMillis() - time;
		}

		public void clear(){
			cache.clear();
			wordsPerThread = null;
			//System.err.println("number of hits = " + hits + " | number of misses = " + misses);
			hits = 0;
			misses = 0;
		}

		public void update(){
			if(cache.size()>0){
				Thread[] cacheThreads = new Thread[wordsPerThread.length];
				for(int t = 0; t < wordsPerThread.length; t++){
					int finalT = t;
					cacheThreads[t] = new Thread(){
						public void run() {
							for(WordRepresentationCacheEntry entry : wordsPerThread[finalT]){
								entry.backpropagate();
							}
						};
					};
					cacheThreads[t].start();
				}
				for(int t = 0; t < wordsPerThread.length; t++){
					try {
						cacheThreads[t].join();
					} catch (InterruptedException e) {
						throw new RuntimeException(e);
					}
				}
			}
			clear();
		}
		
		public DenseNeuronArray get(String input, GraphInference inference){
			if(cache.containsKey(input)){
				hits++;
				return cache.get(input).getOutput();
			}
			misses++;
			DenseNeuronArray rep = buildWordRepresentation(input, inference);
			return rep;
		}
	}

	private static final String WORD_REPRESENTATION_KEY = "WORD_REP"; 

	WordRepresentationCache cache = new WordRepresentationCache();
	WordRepresentationSetup setup;
	Vocab[] vocabPerFeatureSet;
	public LookupTable[] lookupTablePerFeatureSet;
	
	Vocab[] vocabPerSequenceFeatureSet;
	public LookupTable[] lookupTablePerSequentialFeatureSet;
	public DeepRNN[] sequenceEncoderLayer;
	
	int outputDim;
	int[] outputDimPerSet;
	int[] charProjectionDimPerSet;
	String[] unkTokenPerFeatureSet;
	String[] unkTokenPerSequenceSet;
	
	public WordRepresentationLayer(WordRepresentationSetup setup) {
		this.setup = setup;
		if(setup.existingWords == null) return;
		for(int i = 0; i < setup.existingWords.getTypes(); i++){
			WordEntry wordEntry = setup.existingWords.getEntryFromId(i);
			if(i == 0){
				vocabPerFeatureSet = new Vocab[setup.featureExtractors.size()];
				for(int fs = 0; fs < vocabPerFeatureSet.length; fs++){
					vocabPerFeatureSet[fs] = new Vocab();
				}	

				vocabPerSequenceFeatureSet = new Vocab[setup.sequenceExtractors.size()];
				for(int fs = 0; fs < vocabPerSequenceFeatureSet.length; fs++){
					vocabPerSequenceFeatureSet[fs] = new Vocab();
				}
			}

			if(wordEntry.count==0) continue;

			for(int fs = 0; fs < vocabPerFeatureSet.length; fs++){
				String activation = setup.featureExtractors.get(fs).extract(wordEntry.word);											
				vocabPerFeatureSet[fs].addWordToVocab(activation, wordEntry.count);
			}
			for(int fs = 0; fs < vocabPerSequenceFeatureSet.length; fs++){
				for(String activation : setup.sequenceExtractors.get(fs).extract(wordEntry.word)){
					vocabPerSequenceFeatureSet[fs].addWordToVocab(activation, wordEntry.count);
				}
			}
		}

		unkTokenPerFeatureSet = new String[vocabPerFeatureSet.length];
		for(int fs = 0; fs < vocabPerFeatureSet.length; fs++){
			if(setup.featurePreinitializationFile.get(fs) != null){
				addPreloadedEntries(setup.featurePreinitializationFile.get(fs),vocabPerFeatureSet[fs], setup.maxOccurForDropout + 1);
			}
			unkTokenPerFeatureSet[fs] = vocabPerFeatureSet[fs].genNewWord();
			vocabPerFeatureSet[fs].addWordToVocab(unkTokenPerFeatureSet[fs],10000);
			vocabPerFeatureSet[fs].sortVocabByCount(setup.maxNumberOfSparseFeatures);
		}	

		outputDimPerSet = new int[vocabPerFeatureSet.length];
		lookupTablePerFeatureSet = new LookupTable[vocabPerFeatureSet.length];
		for(int fs = 0; fs < vocabPerFeatureSet.length; fs++){
			int dim = vocabPerFeatureSet[fs].getTypes();
			if(dim > setup.projectionDim){
				dim = setup.projectionDim;
			}
			outputDimPerSet[fs] = dim;
			lookupTablePerFeatureSet[fs] = new LookupTable(vocabPerFeatureSet[fs], dim);
			if(setup.featurePreinitializationFile.get(fs) != null){
				preinitialize(setup.featurePreinitializationFile.get(fs), vocabPerFeatureSet[fs], lookupTablePerFeatureSet[fs]);
			}
		}

		unkTokenPerSequenceSet = new String[vocabPerSequenceFeatureSet.length];
		for(int fs = 0; fs < vocabPerSequenceFeatureSet.length; fs++){
			if(setup.sequencePreinitializationFile.get(fs) != null){
				addPreloadedEntries(setup.sequencePreinitializationFile.get(fs),vocabPerSequenceFeatureSet[fs], setup.maxOccurForDropout + 1);
			}
			unkTokenPerSequenceSet[fs] = vocabPerSequenceFeatureSet[fs].genNewWord();
			vocabPerSequenceFeatureSet[fs].addWordToVocab(unkTokenPerSequenceSet[fs],10000);
			vocabPerSequenceFeatureSet[fs].sortVocabByCount(setup.maxNumberOfSparseFeatures);
//			vocabPerSequenceFeatureSet[fs].generateHuffmanCodes();
		}

		charProjectionDimPerSet = new int[vocabPerSequenceFeatureSet.length];
		lookupTablePerSequentialFeatureSet = new LookupTable[vocabPerSequenceFeatureSet.length];
		sequenceEncoderLayer = new DeepRNN[vocabPerSequenceFeatureSet.length];
		for(int fs = 0; fs < vocabPerSequenceFeatureSet.length; fs++){
			int dim = vocabPerSequenceFeatureSet[fs].getTypes();
			if(dim > setup.sequenceProjectionDim){
				dim = setup.sequenceProjectionDim;
			}
			charProjectionDimPerSet[fs] = dim;
			lookupTablePerSequentialFeatureSet[fs] = new LookupTable(vocabPerSequenceFeatureSet[fs], dim);
			if(setup.sequencePreinitializationFile.get(fs) != null){
				preinitialize(setup.sequencePreinitializationFile.get(fs), vocabPerSequenceFeatureSet[fs], lookupTablePerSequentialFeatureSet[fs]);
			}

			sequenceEncoderLayer[fs] = new DeepRNN(dim);
			sequenceEncoderLayer[fs] = sequenceEncoderLayer[fs].addLayer(setup.projectionDim, setup.sequenceStateDim, setup.sequenceType, setup.sequenceForward, setup.sequenceBackward, setup.sequenceSigmoid);
		}

		outputDim = 0;
		for(int fs = 0; fs < vocabPerFeatureSet.length; fs++){
			outputDim += outputDimPerSet[fs];
		}
		outputDim += vocabPerSequenceFeatureSet.length*setup.projectionDim;
		outputDim += setup.doubleFeatureExtractors.size();
	}
	
	public void addFeatures(WordRepresentationLayer loadRep) {		
		Vocab[] newVocabPerFeatureSet = new Vocab[vocabPerFeatureSet.length + loadRep.vocabPerFeatureSet.length];
		ArrayUtils.concat(vocabPerFeatureSet, loadRep.vocabPerFeatureSet, newVocabPerFeatureSet);
		vocabPerFeatureSet = newVocabPerFeatureSet;
		
		LookupTable[] newLookupTablePerFeatureSet = new LookupTable[lookupTablePerFeatureSet.length + loadRep.lookupTablePerFeatureSet.length];
		ArrayUtils.concat(lookupTablePerFeatureSet, loadRep.lookupTablePerFeatureSet, newLookupTablePerFeatureSet);
		lookupTablePerFeatureSet = newLookupTablePerFeatureSet;
		
		Vocab[] newVocabPerSequenceFeatureSet =  new Vocab[vocabPerSequenceFeatureSet.length + loadRep.vocabPerSequenceFeatureSet.length];
		ArrayUtils.concat(vocabPerSequenceFeatureSet, loadRep.vocabPerSequenceFeatureSet, newVocabPerSequenceFeatureSet);
		vocabPerSequenceFeatureSet = newVocabPerSequenceFeatureSet;
		
		LookupTable[] newLookupTablePerSequentialFeatureSet = new LookupTable[lookupTablePerSequentialFeatureSet.length + loadRep.lookupTablePerSequentialFeatureSet.length];
		ArrayUtils.concat(lookupTablePerSequentialFeatureSet, loadRep.lookupTablePerSequentialFeatureSet, newLookupTablePerSequentialFeatureSet);
		lookupTablePerSequentialFeatureSet = newLookupTablePerSequentialFeatureSet;

		DeepRNN[] newSequenceEncoderLayer = new DeepRNN[sequenceEncoderLayer.length + loadRep.sequenceEncoderLayer.length];
		ArrayUtils.concat(sequenceEncoderLayer, loadRep.sequenceEncoderLayer, newSequenceEncoderLayer);
		sequenceEncoderLayer = newSequenceEncoderLayer;
		
		outputDimPerSet = ArrayUtils.concat(outputDimPerSet, loadRep.outputDimPerSet);
		charProjectionDimPerSet = ArrayUtils.concat(charProjectionDimPerSet, loadRep.charProjectionDimPerSet);
		
		unkTokenPerFeatureSet = ArrayUtils.concat(unkTokenPerFeatureSet, loadRep.unkTokenPerFeatureSet);
		unkTokenPerSequenceSet = ArrayUtils.concat(unkTokenPerSequenceSet, loadRep.unkTokenPerSequenceSet);
		
		outputDim += loadRep.getOutputDim();
		setup.add(loadRep.setup);
	}
	
	public void save(PrintStream out){
		out.println(outputDim);
		SerializeUtils.saveIntArray(outputDimPerSet, out);
		SerializeUtils.saveIntArray(charProjectionDimPerSet, out);
		SerializeUtils.saveStringArray(unkTokenPerFeatureSet, out);
		SerializeUtils.saveStringArray(unkTokenPerSequenceSet, out);
		for(int fs = 0; fs < vocabPerFeatureSet.length; fs++){		
			lookupTablePerFeatureSet[fs].save(out);
			vocabPerFeatureSet[fs].saveVocab(out);
		}
		for(int fs = 0; fs < vocabPerSequenceFeatureSet.length; fs++){
			lookupTablePerSequentialFeatureSet[fs].save(out);
			vocabPerSequenceFeatureSet[fs].saveVocab(out);
			sequenceEncoderLayer[fs].save(out);
		}
	}
	
	public static WordRepresentationLayer load(BufferedReader in, WordRepresentationSetup setup){
		WordRepresentationLayer layer = new WordRepresentationLayer(setup);
		try {
			layer.outputDim = Integer.parseInt(in.readLine());
			layer.outputDimPerSet = SerializeUtils.loadIntArray(in);
			layer.charProjectionDimPerSet = SerializeUtils.loadIntArray(in);
			layer.unkTokenPerFeatureSet = SerializeUtils.loadStringArray(in);
			layer.unkTokenPerSequenceSet = SerializeUtils.loadStringArray(in);
			
			layer.lookupTablePerFeatureSet = new LookupTable[layer.outputDimPerSet.length];
			layer.vocabPerFeatureSet = new Vocab[layer.outputDimPerSet.length];
			
			for(int i = 0; i < layer.outputDimPerSet.length; i++){
				layer.lookupTablePerFeatureSet[i] = LookupTable.load(in);
				layer.vocabPerFeatureSet[i] = Vocab.loadVocab(in);
			}
			
			layer.lookupTablePerSequentialFeatureSet = new LookupTable[layer.charProjectionDimPerSet.length];
			layer.vocabPerSequenceFeatureSet = new Vocab[layer.charProjectionDimPerSet.length];
			layer.sequenceEncoderLayer = new DeepRNN[layer.charProjectionDimPerSet.length];
			
			for(int i = 0; i < layer.charProjectionDimPerSet.length; i++){
				layer.lookupTablePerSequentialFeatureSet[i] = LookupTable.load(in);
				layer.vocabPerSequenceFeatureSet[i] = Vocab.loadVocab(in);
				layer.sequenceEncoderLayer[i] = DeepRNN.load(in);
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return layer;
	}

	public static void addPreloadedEntries(String trainedFile, Vocab vocab, int count){
		IOUtils.iterateFiles(new String[]{trainedFile},  new IOUtils.iterateFilesCallback(){
			@Override
			public void cb(String[] lines, int lineNumber) {
				String line = lines[0];
				String params[] = line.split("\\s+");
				if(params.length == 2 && lineNumber == 0) return;

				String word = params[0];				
				vocab.addWordToVocab(word, count);
			}
		});
	}

	public static void preinitialize(String trainedFile, Vocab vocab, LookupTable table){
		IOUtils.iterateFiles(new String[]{trainedFile},  new IOUtils.iterateFilesCallback(){
			@Override
			public void cb(String[] lines, int lineNumber) {
				String line = lines[0];
				String params[] = line.split("\\s+");
				if(params.length == 2 && lineNumber == 0) return;

				String word = params[0];
				double[] embeddings = new double[params.length-1];
				for(int i = 1; i < params.length; i++){
					embeddings[i-1] = Double.parseDouble(params[i]);
				}
				int key = vocab.getEntry(word).id;
				table.setPretrainedWeight(key, embeddings);
			}
		});
	}
	
	public long fillCache(Set<String> words, int threads, boolean train){
		return cache.fill(words, threads, train);
	}
	
	public void buildRepresentationsForVocab(Vocab vocab, int threads, double[][] ret){
		threads = Math.min(vocab.getTypes(), threads);		
		Thread[] cacheThreads = new Thread[threads];
		int dim = getOutputDim();
		int numberPerThread = vocab.getTypes()/threads;
		for(int t = 0; t < threads; t++){
			int start = t*numberPerThread;
			int end = (t+1)*numberPerThread-1;
			int threadId = t;
			if(t == threads - 1) end = vocab.getTypes();
			final int endFinal = end;
			cacheThreads[t] = new Thread(){
				public void run() {
					for(int i = start; i <= endFinal; i++){
						GraphInference inference = new GraphInference(threadId, false);
						DenseNeuronArray emb = cache.get(vocab.getEntryFromId(i).getWord(), inference);						
						inference.init();
						inference.forward();
						for(int j = 0; j < dim; j++){
							ret[i][j] = emb.getNeuron(j);
						}
					}
				};
			};
			cacheThreads[t].start();
		}
		for(int t = 0; t < threads; t++){
			try {
				cacheThreads[t].join();
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			}
		}
	}

	public DenseNeuronArray buildWordRepresentation(String input, GraphInference inference){		
		//sequence features
		DenseNeuronArray output = new DenseNeuronArray(getOutputDim());
		int offset = 0;
		inference.addNeurons(2, output);
		String outputName = "word representation for word " + input;

		for(int fs = 0; fs < vocabPerSequenceFeatureSet.length; fs++){
			String[] featureSetSequence = setup.sequenceExtractors.get(fs).extract(input);
			DenseNeuronArray[] featureSetSequenceProjections = new DenseNeuronArray[featureSetSequence.length];
			if(featureSetSequence.length > 0){
				outputName+="\n sequence feature " + fs + ":";
				for(int s = 0 ; s < featureSetSequence.length; s++){
					String activation = featureSetSequence[s];
					String name = featureSetSequence[s] + " -> ";
					WordEntry activationEntry = vocabPerSequenceFeatureSet[fs].getEntry(activation);
					if(activationEntry==null){
						activation = unkTokenPerSequenceSet[fs];
						name = featureSetSequence[s] + "(unk) ->";
					}
					else if(inference.isTrain() && activationEntry.count <= setup.maxOccurForDropout){
						if(RandomUtils.initializeRandomNumber(0, 1, 1) < setup.dropoutProbability){
							activation = unkTokenPerSequenceSet[fs];
							name = featureSetSequence[s] + "(unk dropped) ->";
						}
					}
					outputName+= name;
					featureSetSequenceProjections[s] = new DenseNeuronArray(charProjectionDimPerSet[fs]);
					inference.addNeurons(1, featureSetSequenceProjections[s]);
					featureSetSequenceProjections[s].setName(name);
					inference.addMapping(new OutputMappingStringToDense(activation, featureSetSequenceProjections[s], lookupTablePerSequentialFeatureSet[fs]));
				}
				outputName+="</s>";
			}
			int dim = setup.projectionDim;
			OutputMappingDenseArrayToDense sequenceToStateMapping = new OutputMappingDenseArrayToDense(0,charProjectionDimPerSet[fs]-1,offset,offset+dim-1,featureSetSequenceProjections,output, sequenceEncoderLayer[fs]);			
			inference.addMapping(sequenceToStateMapping);
			offset+=dim;
		}

		for(int fs = 0; fs < vocabPerFeatureSet.length; fs++){
			String activation = setup.featureExtractors.get(fs).extract(input);
			String name = "\n word feature " + activation;
			WordEntry activationEntry = vocabPerFeatureSet[fs].getEntry(activation);
			if(activationEntry==null){
				name = "\n word feature " + activation + "(unk)";
				activation = unkTokenPerFeatureSet[fs];
			}
			else if(inference.isTrain() && activationEntry.count <= setup.maxOccurForDropout){
				if(RandomUtils.initializeRandomNumber(0, 1, 1) < setup.dropoutProbability){
					name = "\n word feature " + activation + "(unk dropped)";
					activation = unkTokenPerFeatureSet[fs];
				}
			}
			outputName+=name;
			int dim = outputDimPerSet[fs];
			inference.addMapping(new OutputMappingStringToDense(activation, offset,offset+dim-1, output, lookupTablePerFeatureSet[fs]));
			offset+=dim;
		}
		
		DenseNeuronArray denseFeatures = new DenseNeuronArray(setup.doubleFeatureExtractors.size());
		denseFeatures.init();
		inference.addNeurons(0, denseFeatures);
		
		for(int f = 0; f < setup.doubleFeatureExtractors.size(); f++){
			denseFeatures.addNeuron(f, setup.doubleFeatureExtractors.get(f).extract(input));
		}
		inference.addMapping(new OutputMappingDenseToDense(0, setup.doubleFeatureExtractors.size()-1, offset,offset+setup.doubleFeatureExtractors.size()-1, denseFeatures, output, CopyLayer.singleton));
		
		output.setName(outputName);

		return output;
	}

	public void buildInference(String input, DenseNeuronArray output, int outputStart, int outputEnd, Mapping mapping){
		DenseNeuronArray rep = cache.get(input, mapping.getSubInference());
		mapping.getSubInference().init();
		mapping.getSubInference().forward();		
		mapping.setForwardParam(WORD_REPRESENTATION_KEY,rep);
		output.addNeuron(rep, 0, outputStart);//, outputEnd-outputStart+1);
		output.setName(rep.getName());
	}

	public void buildInference(String[] input, DenseNeuronArray[] output, int outputStart, int outputEnd, Mapping mapping){
		DenseNeuronArray[] rep = new DenseNeuronArray[input.length];
		for(int i = 0; i < input.length; i++){
			rep[i] = cache.get(input[i], mapping.getSubInference());
		}
		mapping.getSubInference().init();
		mapping.getSubInference().forward();		
		for(int i = 0; i < input.length; i++){
			output[i].addNeuron(rep[i], 0, outputStart);//, outputEnd-outputStart+1);
			output[i].setName(rep[i].getName());			
		}
		mapping.setForwardParam(WORD_REPRESENTATION_KEY,rep);
	}

	@Override
	public void forward(String input, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingStringToDense mapping) {
		buildInference(input, output, outputStart, outputEnd, mapping);
	}

	@Override
	public void forward(String[] input, DenseNeuronArray[] output,
			int outputStart, int outputEnd,
			OutputMappingStringArrayToDenseArray mapping) {
		buildInference(input, output, outputStart, outputEnd, mapping);
	}

	@Override
	public void backward(String input, DenseNeuronArray output,
			int outputStart, int outputEnd, OutputMappingStringToDense mapping) {
		DenseNeuronArray rep = (DenseNeuronArray) mapping.getForwardParam(WORD_REPRESENTATION_KEY);
		rep.addError(output, outputStart, 0);//, outputEnd-outputStart+1);
		mapping.getSubInference().backward();
	}

	@Override
	public void backward(String[] input, DenseNeuronArray[] output,
			int outputStart, int outputEnd,
			OutputMappingStringArrayToDenseArray mapping) {
		DenseNeuronArray[] rep = (DenseNeuronArray[]) mapping.getForwardParam(WORD_REPRESENTATION_KEY);
		for(int i = 0; i < input.length; i++){
			rep[i].addError(output[i], outputStart, 0);//, outputEnd-outputStart+1);			
		}
		mapping.getSubInference().backward();

	}
	
	public void emptyCache(){
		cache.clear();
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		cache.update();
		for(int fs = 0; fs < vocabPerFeatureSet.length; fs++){
			lookupTablePerFeatureSet[fs].updateWeights(learningRate, momentum);
		}

		for(int fs = 0; fs < vocabPerSequenceFeatureSet.length; fs++){
			lookupTablePerSequentialFeatureSet[fs].updateWeights(learningRate, momentum);
			sequenceEncoderLayer[fs].updateWeights(learningRate, momentum);
		}
	}

	public int getOutputDim() {
		return outputDim;
	}
	
	public void printSimilarityTable(int numToConsider, int numToPrint, Vocab vocab, PrintStream out){
		if(numToConsider > vocab.getTypes()){
			numToConsider = vocab.getTypes();
		}
		if(numToPrint > numToConsider){
			numToPrint = numToConsider;
		}
		DenseNeuronArray[] reps = new DenseNeuronArray[numToConsider];
		String[] words = new String[numToConsider];
		for(int i = 0; i < numToConsider; i++){
			GraphInference inference = new GraphInference(0, false);
			reps[i] = buildWordRepresentation(vocab.getEntryFromId(i).word, inference);
			words[i] = vocab.getEntryFromId(i).word;
			inference.init();
			inference.forward();
		}
		WordVectors vectors = new WordVectors(reps);
		for(int i = 0; i < numToPrint; i++){
			vectors.printTopN(i, 10, words, out);
			out.println();
		}
	}
	
	public void printSimilarityTable(int numToConsider, int numToPrint, HashSet<String> vocab, PrintStream out){
		if(numToConsider > vocab.size()){
			numToConsider = vocab.size();
		}
		if(numToPrint > numToConsider){
			numToPrint = numToConsider;
		}
		DenseNeuronArray[] reps = new DenseNeuronArray[numToConsider];
		String[] words = vocab.toArray(new String[0]);
		for(int i = 0; i < numToConsider; i++){
			GraphInference inference = new GraphInference(0, false);
			reps[i] = buildWordRepresentation(words[i], inference);
			inference.init();
			inference.forward();
		}
		WordVectors vectors = new WordVectors(reps);
		for(int i = 0; i < numToPrint; i++){
			vectors.printTopN(i, 10, words, out);
			out.println();
		}
	}
	
	public void printSimilarityTable(String pivot, HashSet<String> vocab, PrintStream out) {
		printSimilarityTable(pivot, vocab, out,"cosine");
	}

	public void printSimilarityTable(String pivot, HashSet<String> vocab, PrintStream out, String sim) {
		DenseNeuronArray[] reps = new DenseNeuronArray[vocab.size()];
		String[] words = new String[vocab.size()];
		Iterator<String> vocabIt = vocab.iterator();
		for(int i = 0; i < vocab.size(); i++){
			GraphInference inference = new GraphInference(0, false);
			words[i] = vocabIt.next();
			reps[i] = cache.get(words[i], inference);
			inference.init();
			inference.forward();
		}
		GraphInference inference = new GraphInference(0, false);
		WordVectors vectors = new WordVectors(reps);
		DenseNeuronArray pivotRep = cache.get(pivot,inference);
		inference.init();
		inference.forward();
		
		if(sim.equals("euclidean")){
			vectors.printTopNEuclidean(pivotRep.copyAsArray(), 10, words, out);
		}
		else{
			vectors.printTopN(pivotRep.copyAsArray(), 10, words, out);			
		}
		out.println();
	}
	
	public void printActivations(String pivot){
		GraphInference inference = new GraphInference(0, false);
		DenseNeuronArray pivotRep = cache.get(pivot,inference);
		inference.init();
		inference.forward();
		
	}
	
	public void printVectors(Vocab vocab,int max, PrintStream out){	
		out.println(vocab.getTypes() + " " + outputDim);
		int min = Math.min(vocab.getTypes(),max);
		for(int i = 0; i < min; i++){
			GraphInference inference = new GraphInference(0, false);
			DenseNeuronArray rep = buildWordRepresentation(vocab.getEntryFromId(i).word, inference);
			String word = vocab.getEntryFromId(i).word;
			out.print(word);
			inference.init();
			inference.forward();
			for(int j = 0 ; j < outputDim; j++){
				out.print(" " + rep.getNeuron(j));
			}
			out.println();
		}
	}
	
	public void printVectors(HashSet<String> vocab,PrintStream out){	
		out.println(vocab.size() + " " + outputDim);
		for(String word : vocab){
			GraphInference inference = new GraphInference(0, false);
			DenseNeuronArray rep = buildWordRepresentation(word, inference);
			out.print(word);
			inference.init();
			inference.forward();
			for(int j = 0 ; j < outputDim; j++){
				out.print(" " + rep.getNeuron(j));
			}
			out.println();
		}
	}

	public static void main(String[] args) throws IOException{
		Vocab vocab = new Vocab();
		vocab.addWordToVocab("I",3);
		vocab.addWordToVocab("went",3);
		vocab.addWordToVocab("shopping",3);
		vocab.addWordToVocab("in",3);
		vocab.addWordToVocab("Lisbonnnnn",3);

		WordRepresentationSetup setup = new WordRepresentationSetup(vocab, 50, 50, 150);
//		setup.addFeatureExtractor(new LowercasedWordFeatureExtractor(), "/Users/lingwang/Documents/workspace/ContinuousVectors/twitter/wordvec/twitter_structskipngram_50");
		setup.addFeatureExtractor(new LowercasedWordFeatureExtractor());
		WordRepresentationLayer layer = new WordRepresentationLayer(setup);
		
		WordRepresentationSetup setup2 = new WordRepresentationSetup(vocab, 50, 50, 150);		
		setup2.addFeatureExtractor(new CapitalizationWordFeatureExtractor());
		setup2.addSequenceExtractor(new CharSequenceExtractor());
		WordRepresentationLayer layer2 = new WordRepresentationLayer(setup2);
		layer.addFeatures(layer2);
		
		PrintStream out = IOUtils.getPrintStream("/tmp/file");
		layer.save(out);
		out.close();
		
		BufferedReader in = IOUtils.getReader("/tmp/file");
		layer = WordRepresentationLayer.load(in, setup);
		in.close();		
		
		for(int e = 0; e < 10; e++){
			GraphInference inference = new GraphInference(0, true);
			String[] input = "I am living in Lisbonnnnn".split("\\s+");
			DenseNeuronArray[] representation = DenseNeuronArray.asArray(input.length, layer.getOutputDim());
			inference.addNeurons(1, representation);
			inference.addMapping(new OutputMappingStringArrayToDenseArray(input, representation, layer));

			inference.init();
			inference.forward();
			for(int w = 0; w < input.length; w++){
				for(int i = 0; i < layer.getOutputDim(); i++){
					representation[w].addError(i, 1-representation[w].getNeuron(i));
				}
			}
			inference.backward();
			inference.printNeurons();
			inference.commit(0.1);
		}
	}
}
