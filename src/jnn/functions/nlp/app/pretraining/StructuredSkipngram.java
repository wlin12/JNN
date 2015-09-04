package jnn.functions.nlp.app.pretraining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;

import jnn.functions.composite.FastNegativeSamplingLayer;
import jnn.functions.nlp.aux.metrics.ErrorStats;
import jnn.functions.nlp.words.WordRepresentationLayer;
import jnn.functions.nlp.words.features.WordRepresentationSetup;
import jnn.training.GlobalParameters;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import util.IOUtils;
import vocab.Vocab;

public class StructuredSkipngram {

	public StructuredSkipngramSpecification setup;
	public WordRepresentationSetup wordSetup;
	public WordRepresentationLayer wordRepresentation;
	//public NegativeSamplingLayer softmax;
	public FastNegativeSamplingLayer[] softmaxFast;

	Vocab inputVocab;
	Vocab outputVocab;	

	WindowWordPairs pairs;

	long numberOfSentences = 0;
	long numberOfWords = 0;
	
	//short-term memory
	double[][] shortTermEmb;
	Vocab shortTermVocab = new Vocab();

	//type 
	BufferedReader typeReader;
	HashSet<String> currentPairs = new HashSet<String>();	

	ErrorStats devPP = new ErrorStats();	

	public StructuredSkipngram(StructuredSkipngramSpecification setup) {
		super();
		this.setup = setup;
		devPP.init();
		buildVocabulary();
		buildWordRepresentations();
		buildOutputLayer();
		//load();
		pairs = new WindowWordPairs(setup.windowSize);
		
//		shortTermEmb = new double[setup.batchSize][wordRepresentation.getOutputDim()];
	}

	public void buildVocabulary(){
		inputVocab = new Vocab();
		inputVocab.setMinOccur(setup.minCount);
		outputVocab = new Vocab();
		outputVocab.setMinOccur(setup.minCount);
		Scanner scan;
		try {
			scan = new Scanner(new File(setup.trainFile));
			while(scan.hasNext()){
				String token = scan.next();
				inputVocab.addWordToVocab(token);
				outputVocab.addWordToVocab(token.toLowerCase());
				numberOfWords++;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		outputVocab.reduceVocab();
		outputVocab.sortVocabByCount(setup.maxTypeOutput);
		outputVocab.generateHuffmanCodes();
		inputVocab.reduceVocab();
		inputVocab.sortVocabByCount();
		inputVocab.generateHuffmanCodes();
	}

	public void buildWordRepresentations(){
		wordSetup = new WordRepresentationSetup(inputVocab, setup.wordProjectionDim, setup.charProjectionDim, setup.charStateDim);
		wordSetup.loadFromString(setup.wordFeatures, setup.word2vecEmbeddings);
		wordRepresentation = new WordRepresentationLayer(wordSetup);
	}

	public void buildOutputLayer(){
		//softmax = new NegativeSamplingLayer(wordRepresentation.getOutputDim(), outputVocab, setup.negativeSamples);
		softmaxFast = new FastNegativeSamplingLayer[setup.windowSize*2];
		for(int i = 0; i < setup.windowSize*2;i++){
			softmaxFast[i] = new FastNegativeSamplingLayer(wordRepresentation.getOutputDim(), outputVocab, setup.negativeSamples);
		}
	}	

	public void train(int threads, boolean printInfo) {
		threads = Math.min(threads, setup.batchSize);
		long startTime = System.currentTimeMillis();
		int numberOfWords = 0;

		String[][][] batchesPerThread = null;
		if(setup.useShortTermMemory){
			batchesPerThread = setup.trainingData.readWindowsEquivalentSkip(setup.batchSize/threads, threads, inputVocab);
		}
		else{
			batchesPerThread = setup.trainingData.readWindowsEquivalent(setup.batchSize/threads, threads, inputVocab);			
		}
		long timeToRead = System.currentTimeMillis() - startTime;

//		shortTermVocab = new Vocab();
		HashSet<String> wordList = new HashSet<String>();
		for(int t = 0; t < threads; t++){
			for(int i = 0; i < batchesPerThread[0].length; i++){
				numberOfWords++;
				String word = batchesPerThread[t][i][setup.windowSize];
				wordList.add(word);
//				shortTermVocab.addWordToVocab(word);
			}
		}
		
//		wordRepresentation.buildRepresentationsForVocab(shortTermVocab, threads, shortTermEmb);

		long wordRepTime = wordRepresentation.fillCache(wordList, threads, true);		
		
		double norm = 1/(double)(numberOfWords*(setup.windowSize*2));
		//			double norm = 1;

		StructuredSkipngramInstance[] trainInstances = new StructuredSkipngramInstance[threads];
		Thread[] trainThreads = new Thread[threads];

		startTime = System.currentTimeMillis();
		for(int t = 0; t < threads; t++){
			final int tFinal = t;
			//trainInstances[t] = new StructuredSkipngramInstance(t, batchesPerThread[t], outputVocab, wordRepresentation, softmax,softmaxFast, setup.windowSize, norm); 
			trainInstances[t] = new StructuredSkipngramInstance(t, batchesPerThread[t], outputVocab, wordRepresentation, null,softmaxFast, setup.windowSize, norm);
			trainThreads[t] = new Thread(){
				@Override
				public void run() {
					trainInstances[tFinal].trainFast();
				}
			};
			trainThreads[t].start();
		}
		for(int t = 0; t < threads; t++){
			try {
				trainThreads[t].join();
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			}			
		}
		
		wordRepresentation.updateWeightsTimed(0, 0);
		long timeForInference = System.currentTimeMillis() - startTime;

		if(printInfo){
			long computationTime = System.currentTimeMillis() - startTime;
			System.err.println("number of words per second (train) = " + (numberOfWords/(double)computationTime) + "k");
			System.err.println("time to build word rep (train) = " +wordRepTime);
			System.err.println("time for inference (train) = " +timeForInference);
			System.err.println("read time (train) = " +timeToRead);		
			System.err.println("total time (train) = " +computationTime);		
			wordRepresentation.printCommitTimeAndReset();
			//System.err.println("cache size =" + shortTermEmbeddings.size() + (shortTermEmbeddings.size()*100/inputVocab.getTypes()) + "%");
		}
	}
	
	public void devll(int threads) {
		double error = 0;
		threads = Math.min(threads, setup.batchSize);
		long startTime = System.currentTimeMillis();
		int numberOfWords = 0;

		String[][][] batchesPerThread = null;
		setup.devData.reset();
		batchesPerThread = setup.devData.readWindowsEquivalent(setup.batchSize/threads, threads, inputVocab);			

		HashSet<String> wordList = new HashSet<String>();
		for(int t = 0; t < threads; t++){
			for(int i = 0; i < batchesPerThread[0].length; i++){
				numberOfWords++;
				String word = batchesPerThread[t][i][setup.windowSize];
				wordList.add(word);
			}
		}
		
		long wordRepTime = wordRepresentation.fillCache(wordList, threads, true);		
		
		double norm = 1/(double)(numberOfWords*(setup.windowSize*2));
		//			double norm = 1;

		StructuredSkipngramInstance[] devInstances = new StructuredSkipngramInstance[threads];
		Thread[] devThreads = new Thread[threads];

		startTime = System.currentTimeMillis();
		for(int t = 0; t < threads; t++){
			final int tFinal = t;
			//trainInstances[t] = new StructuredSkipngramInstance(t, batchesPerThread[t], outputVocab, wordRepresentation, softmax,softmaxFast, setup.windowSize, norm); 
			devInstances[t] = new StructuredSkipngramInstance(t, batchesPerThread[t], outputVocab, wordRepresentation, null,softmaxFast, setup.windowSize, norm);
			devThreads[t] = new Thread(){
				@Override
				public void run() {
					devInstances[tFinal].error();
				}
			};
			devThreads[t].start();
		}
		for(int t = 0; t < threads; t++){
			try {
				devThreads[t].join();
				error += devInstances[t].error;
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			}			
		}
		
		wordRepresentation.emptyCache();
		
		devPP.initError();
		
		devPP.addError(error);
		devPP.commitError();		
		devPP.displayResults("dev pp", true);
		PrintStream stats = IOUtils.getPrintStream(setup.outputDir + "/stats");
		devPP.displayResults("dev pp", stats, false);
		stats.close();

	}

	//	public void trainByWordPairs(int minCountToUpdate, int batchSize, int threads){
	//		HashSet<String> wordsToRun = new HashSet<String>();
	//		int numberOfWords = 0;
	//		long startTime = System.currentTimeMillis();
	//		while(true){
	//			InputSentence sentence = setup.trainingData.read();
	//			boolean[] validWords = new boolean[sentence.tokens.length];
	//			for(int i = 0; i < sentence.tokens.length; i++){
	//				String word = sentence.tokens[i];	
	//				validWords[i] = outputVocab.getEntry(word.toLowerCase()) != null;
	//			}
	//			for(int i = 0; i < sentence.tokens.length; i++){
	//				String word = sentence.tokens[i];
	//				int count = pairs.addWord(sentence.tokens, i, validWords);
	//				if(wordsToRun.contains(word)){
	//					numberOfWords++;
	//				}
	//				else{
	//					if(count>=minCountToUpdate){
	//						numberOfWords+=count;
	//						wordsToRun.add(word);
	//					}
	//				}
	//			}
	//			if(batchSize < numberOfWords) break;
	//		}
	//		long wordRepTime = wordRepresentation.fillCache(wordsToRun, threads, true);
	//		double norm = 1/(double)(numberOfWords*(setup.windowSize*2));
	//
	//		StructuredSkipngramInstance[] trainInstances = new StructuredSkipngramInstance[threads];
	//		Thread[] trainThreads = new Thread[threads];
	//
	//		Iterator<String> wordIterator = wordsToRun.iterator();
	//
	//		for(int t = 0; t < threads; t++){
	//			final int tFinal = t;
	//			HashSet<String> wordsForThread = new HashSet<String>();
	//			for(int i = 0; i < wordsToRun.size()/threads; i++){
	//				wordsForThread.add(wordIterator.next());
	//			}
	//			if(t == threads-1){
	//				while(wordIterator.hasNext()){
	//					wordsForThread.add(wordIterator.next());
	//				}
	//			}
	//			//trainInstances[t] = new StructuredSkipngramInstance(t, batchesPerThread[t], outputVocab, wordRepresentation, softmax,softmaxFast, setup.windowSize, norm); 
	//			trainInstances[t] = new StructuredSkipngramInstance(t, pairs, wordsForThread, outputVocab, wordRepresentation, null,softmaxFast, setup.windowSize, norm);
	//			trainThreads[t] = new Thread(){
	//				@Override
	//				public void run() {
	//					trainInstances[tFinal].trainWindowFast();
	//				}
	//			};
	//			trainThreads[t].start();
	//		}
	//		for(int t = 0; t < threads; t++){
	//			try {
	//				trainThreads[t].join();
	//			} catch (InterruptedException e) {
	//				throw new RuntimeException(e);
	//			}			
	//		}
	//
	//		long computationTime = System.currentTimeMillis() - startTime;
	//		System.err.println("number of words per second (train) = " + (numberOfWords/(double)computationTime) + "k");
	//		System.err.println("time to build word rep (train) = " +wordRepTime);
	//		System.err.println("total time (train) = " +computationTime);
	//		wordRepresentation.updateWeightsTimed(0, 0);
	//		wordRepresentation.printCommitTimeAndReset();
	//
	//	}

	public void trainRemainingWords(int threads){
		Set<String> wordsToRun = pairs.getWords();
		int numberOfWords = 0;
		long startTime = System.currentTimeMillis();		

		for(String word : wordsToRun){
			numberOfWords+=pairs.getCount(word);
		}

		long wordRepTime = wordRepresentation.fillCache(wordsToRun, threads, true);
		double norm = 1/(double)(numberOfWords*(setup.windowSize*2));

		StructuredSkipngramInstance[] trainInstances = new StructuredSkipngramInstance[threads];
		Thread[] trainThreads = new Thread[threads];

		Iterator<String> wordIterator = wordsToRun.iterator();

		for(int t = 0; t < threads; t++){
			final int tFinal = t;
			HashSet<String> wordsForThread = new HashSet<String>();
			for(int i = 0; i < wordsToRun.size()/numberOfWords; i++){
				wordsForThread.add(wordIterator.next());
			}
			if(t == threads-1){
				while(wordIterator.hasNext()){
					wordsForThread.add(wordIterator.next());
				}
			}
			//trainInstances[t] = new StructuredSkipngramInstance(t, batchesPerThread[t], outputVocab, wordRepresentation, softmax,softmaxFast, setup.windowSize, norm); 
			trainInstances[t] = new StructuredSkipngramInstance(t, pairs, wordsForThread, outputVocab, wordRepresentation, null,softmaxFast, setup.windowSize, norm);
			trainThreads[t] = new Thread(){
				@Override
				public void run() {
					trainInstances[tFinal].trainWindowFast();
				}
			};
			trainThreads[t].start();
		}
		for(int t = 0; t < threads; t++){
			try {
				trainThreads[t].join();
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			}			
		}

		//		long computationTime = System.currentTimeMillis() - startTime;
		//		System.err.println("number of words per second (train) = " + (numberOfWords/(double)computationTime) + "k");
		//		System.err.println("time to build word rep (train) = " +wordRepTime);
		//		System.err.println("total time (train) = " +computationTime);
		//		wordRepresentation.updateWeightsTimed(0, 0);
		//		wordRepresentation.printCommitTimeAndReset();
	}


	private void printRandomEntries() {
		PrintStream out = IOUtils.getPrintStream(setup.outputDir + "sim.gz");
		wordRepresentation.printSimilarityTable(10000, 1000, inputVocab,out);
		out.close();
	}

	private void printVectors() {
		PrintStream out = IOUtils.getPrintStream(setup.outputDir + "vectors.gz");
		wordRepresentation.printVectors(inputVocab,10000,out);
		out.close();
	}

	private void saveRepresentation(){
		System.err.println("saving word representations");
		String file = setup.outputDir + "rep.gz";
		PrintStream out = IOUtils.getPrintStream(file);
		wordRepresentation.save(out);
		out.close();
	}

	public void save(){
		System.err.println("saving model");
		String file = setup.outputDir + "model.gz";
		String tmpFile = setup.outputDir + "model.tmp.gz";
		PrintStream out = IOUtils.getPrintStream(tmpFile);
		inputVocab.saveVocab(out);
		outputVocab.saveVocab(out);
		wordRepresentation.save(out);
		for(int i = 0; i < setup.windowSize*2;i++){
			softmaxFast[i].save(out);			
		}
		//softmax.save(out);
		out.close();
		IOUtils.copyfile(tmpFile, file);
	}

	public void load(){
		String file = setup.outputDir + "model.gz";
		if(IOUtils.exists(file)){
			BufferedReader in = IOUtils.getReader(file);
			inputVocab = Vocab.loadVocab(in);
			outputVocab = Vocab.loadVocab(in);
			wordRepresentation = WordRepresentationLayer.load(in, wordSetup);
			softmaxFast = new FastNegativeSamplingLayer[setup.windowSize*2];
			for(int i = 0; i < setup.windowSize*2;i++){
				softmaxFast[i] = FastNegativeSamplingLayer.load(in, outputVocab);
			}
			//softmax = NegativeSamplingLayer.load(in);
			try {
				in.close();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}		
		}
	}

	public static WordRepresentationLayer loadRep(String file, StructuredSkipngramSpecification setup){
		BufferedReader in = IOUtils.getReader(file);
		Vocab.skipLoad(in);
		Vocab.skipLoad(in);
		WordRepresentationSetup wordSetup = new WordRepresentationSetup(null, setup.wordProjectionDim, setup.charProjectionDim, setup.charStateDim);
		wordSetup.loadFromString(setup.wordFeatures, setup.word2vecEmbeddings);
		WordRepresentationLayer ret = WordRepresentationLayer.load(in, wordSetup);
		try {	
			in.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return ret;
	}

	//	public void buildWindowCountFile(){
	//		String file = setup.outputDir + "countFile.gz";
	//		if(!IOUtils.exists(file)){
	//			String countRawFile = setup.outputDir + "countRaw.gz";
	//			PrintStream out = IOUtils.getPrintStream(countRawFile);
	//			setup.trainingData.reset();
	//			while(!setup.trainingData.isEnd()){
	//				InputSentence input = setup.trainingData.read();
	//				for(int i = 0; i < input.tokens.length; i++){
	//					String word = input.tokens[i];
	//					WordEntry wordEntry = inputVocab.getEntry(word);
	//					if(wordEntry != null){
	//						for(int w = -setup.windowSize; w <= setup.windowSize; w++){
	//							if(w!=0){
	//								int targetWordIndex = i + w;
	//								if(targetWordIndex > 0 && targetWordIndex < input.tokens.length){
	//									String targetWord = input.tokens[targetWordIndex].toLowerCase();
	//									WordEntry targetWordEntry = outputVocab.getEntry(targetWord);
	//									if(targetWordEntry != null){
	//										int windowIndex = w + setup.windowSize;
	//										if(w>0){
	//											windowIndex--;
	//										}
	//										out.println(word + "\t" + targetWord + "\t" + windowIndex);
	//									}
	//								}
	//							}
	//						}
	//					}
	//				}
	//			}
	//			out.close();
	//			SortingUtils.sortUniq(countRawFile, file, setup.outputDir);
	//		}
	//		typeReader = IOUtils.getReader(file);
	//	}

	public void trainByType(int batchSize, int threads){
		long startTime = System.currentTimeMillis();
		while(true){
			try {
				if(!typeReader.ready()){
					typeReader.close();
					typeReader = IOUtils.getReader(setup.outputDir + "countFile.gz");
				}
				String leftOver = pairs.readFromCountFile(typeReader);
				if(pairs.size() > batchSize){
					long wordRepTime = wordRepresentation.fillCache(currentPairs, threads, true);
					StructuredSkipngramInstance[] trainInstances = new StructuredSkipngramInstance[threads];
					Thread[] trainThreads = new Thread[threads];
					Iterator<String> wordIterator = currentPairs.iterator();

					int[] numberOfInstances = new int[threads];
					int sum = 0;
					for(int t = 0; t < threads; t++){
						final int tFinal = t;
						HashSet<String> wordsForThread = new HashSet<String>();
						for(int i = 0; i < currentPairs.size()/threads; i++){
							String word = wordIterator.next();
							numberOfInstances[t]+=pairs.getCount(word);
							sum+=pairs.getCount(word);
							wordsForThread.add(word);
						}
						if(t == threads-1){
							while(wordIterator.hasNext()){
								wordsForThread.add(wordIterator.next());
							}
						}
						//trainInstances[t] = new StructuredSkipngramInstance(t, batchesPerThread[t], outputVocab, wordRepresentation, softmax,softmaxFast, setup.windowSize, norm); 
						trainInstances[t] = new StructuredSkipngramInstance(t, pairs, wordsForThread, outputVocab, wordRepresentation, null,softmaxFast, setup.windowSize, numberOfInstances[t]/(double)sum);
						trainThreads[t] = new Thread(){
							@Override
							public void run() {
								trainInstances[tFinal].trainWindowFast();
							}
						};
						trainThreads[t].start();
					}
					for(int t = 0; t < threads; t++){
						try {
							trainThreads[t].join();
						} catch (InterruptedException e) {
							throw new RuntimeException(e);
						}			
					}

					long computationTime = System.currentTimeMillis() - startTime;
					System.err.println("number of words per second (train) = " + (numberOfWords/(double)computationTime) + "k");
					System.err.println("time to build word rep (train) = " +wordRepTime);
					System.err.println("total time (train) = " +computationTime);
					wordRepresentation.updateWeightsTimed(0, 0);
					wordRepresentation.printCommitTimeAndReset();

					currentPairs.clear();
					currentPairs.add(leftOver);
					break;
				}
				else{
					currentPairs.add(leftOver);
				}
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
	}

	public static void main(String[] args){
		Options options = new Options();
		options.addOption("train_file", true, "training file");
		options.addOption("dev_file", true, "development file");
		options.addOption("lr", true, "learning rate");
		options.addOption("batch_size", true, "batch size");
		options.addOption("threads", true, "number of threads");
		options.addOption("word2vec_embeddings", true, "word2vec embeddings");
		options.addOption("word_features", true, "features separated by commas (e.g. words,capitalization,characters)");
		options.addOption("epochs", true, "number of runs through the corpus");
		options.addOption("validation_interval", true, "batches till validation");
		options.addOption("output_dir", true, "output directory");
		options.addOption("training_type", true, "training type");
		options.addOption("use_short_term_memory", true, "use short term memory");
		options.addOption("min_count", true, "minCount");
		options.addOption("momentum", true, "use momentum");
		if(args.length == 0){
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "java -jar [this program]", options );
			System.exit(0);
		}
		CommandLineParser parser = new BasicParser();
		CommandLine cmd;
		try {
			cmd = parser.parse( options, args);
		} catch (ParseException e) {
			throw new RuntimeException(e);
		}

		int validationInterval = Integer.parseInt(cmd.getOptionValue("validation_interval"));

		if(Boolean.parseBoolean(cmd.getOptionValue("use_short_term_memory"))){
			GlobalParameters.useMomentumDefault = true;			
		}
		GlobalParameters.learningRateDefault = Double.parseDouble(cmd.getOptionValue("lr"));
		GlobalParameters.l2regularizerLambdaDefault = 0;
		int batchSize = Integer.parseInt(cmd.getOptionValue("batch_size"));
		int threads = Integer.parseInt(cmd.getOptionValue("threads"));
		String train = cmd.getOptionValue("train_file");
		String dev = cmd.getOptionValue("dev_file");

		StructuredSkipngramSpecification spec = new StructuredSkipngramSpecification();		
		spec.useShortTermMemory = Boolean.parseBoolean(cmd.getOptionValue("use_short_term_memory"));
		spec.word2vecEmbeddings = cmd.getOptionValue("word2vec_embeddings");
		spec.wordFeatures = cmd.getOptionValue("word_features");
		spec.outputDir = cmd.getOptionValue("output_dir");
		spec.minCount = Integer.parseInt(cmd.getOptionValue("min_count"));
		spec.setDataset(train);
		spec.setDevDataset(dev);
		spec.batchSize = batchSize;

		StructuredSkipngram skipngram = new StructuredSkipngram(spec);
		
		int epochs = Integer.parseInt(cmd.getOptionValue("epochs"));
		System.err.println("there are " + skipngram.numberOfSentences + " sentences");
		System.err.println("there are " + skipngram.numberOfWords + " words");

		String training_type = cmd.getOptionValue("training_type");

		//		if(training_type.equals("type-cache")){
		//			long numberOfIterations = skipngram.numberOfWords * epochs / batchSize;  
		//			for(long i = 0; i < numberOfIterations; i++){			
		//				skipngram.trainByWordPairs(5, batchSize, threads);
		//				System.err.println(i*100/numberOfIterations + "% completed");
		//				if(i % validationInterval == 0 && i > 1){
		//					skipngram.printRandomEntries();
		//					skipngram.printVectors();
		//					//skipngram.saveRepresentation();
		//					skipngram.save();
		//				}
		//			}
		//			skipngram.trainRemainingWords(threads);
		//			skipngram.printRandomEntries();
		//			skipngram.printVectors();
		//			skipngram.saveRepresentation();
		//			skipngram.save();
		//		}
		if(training_type.equals("text")){
			long numberOfIterations = skipngram.numberOfWords * epochs / batchSize;  
			for(long i = 0; i < numberOfIterations; i++){
				boolean printinfo = i % 10 == 0;
				skipngram.train(threads,printinfo);
				if(printinfo){
					System.err.println(i + " - "+ i*100/numberOfIterations + "% completed");
				}
				if(i % validationInterval == 0 && i > 1){
					skipngram.printRandomEntries();
					skipngram.printVectors();
					skipngram.saveRepresentation();
					skipngram.save();
					skipngram.devll(threads);
				}
			}
			skipngram.printRandomEntries();
			skipngram.printVectors();
			skipngram.saveRepresentation();
			skipngram.save();
			skipngram.devll(threads);
		}
		//		if(training_type.equals("type")){
		//			long numberOfIterations = skipngram.inputVocab.getTypes() * epochs / batchSize;
		//			skipngram.buildWindowCountFile();
		//			for(long i = 0; i < numberOfIterations; i++){			
		//				skipngram.trainByType(batchSize, threads);
		//				System.err.println(i*100/numberOfIterations + "% completed");
		//
		//				if(i % validationInterval == 0 && i > 1){
		//					skipngram.printRandomEntries();
		//					//skipngram.printVectors();
		//					skipngram.saveRepresentation();
		//					skipngram.save();
		//				}
		//			}
		//			skipngram.printRandomEntries();
		//			skipngram.printVectors();
		//			skipngram.saveRepresentation();
		//			skipngram.save();
		//		}
	}

}
