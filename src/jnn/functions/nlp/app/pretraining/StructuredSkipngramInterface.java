package jnn.functions.nlp.app.pretraining;

import java.io.BufferedReader;
import java.io.PrintStream;
import java.util.HashSet;

import jnn.functions.nlp.aux.input.LabelledData;
import jnn.functions.nlp.aux.input.LabelledSentence;
import jnn.functions.nlp.words.WordRepresentationLayer;
import jnn.functions.nlp.words.features.WordRepresentationSetup;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import util.IOUtils;
import vocab.Vocab;
import vocab.WordEntry;

public class StructuredSkipngramInterface {
	public static void main(String[] args){
		Options options = new Options();
		options.addOption("word_features", true, "features separated by commas (e.g. words,capitalization,characters)");
		options.addOption("model_file", true, "model file");
		options.addOption("model_type", true, "model file");
		options.addOption("vocab_file", true, "vocab file");
		options.addOption("min_count", true, "minimum count for words");
		options.addOption("threads", true, "number of threads");
		options.addOption("format", true, "file format");
		options.addOption("command", true, "command (interface if empty)");
		options.addOption("distance", true, "distance type (cosine or euclidean)");
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

		String file = cmd.getOptionValue("model_file");
		String modelType = cmd.getOptionValue("model_type");
		if(modelType == null){
			modelType = "";
		}

		int minCount = Integer.parseInt(cmd.getOptionValue("min_count"));
		String vocabFile = cmd.getOptionValue("vocab_file");
		final HashSet<String> vocab = new HashSet<String>();

		String firstLine = IOUtils.readLineFromFile(file);
		String[] firstLineWords = firstLine.split("\\s+");
		String mainCommand  = cmd.getOptionValue("command");

		int threads = Integer.parseInt(cmd.getOptionValue("threads"));
		String format = cmd.getOptionValue("format");
		if(format.equals("text")){
			Vocab vocabInText = new Vocab();
			IOUtils.iterateFiles(vocabFile, new IOUtils.iterateFilesCallback(){
				public void cb(String[] lines, int lineNumber) {
					String line = lines[0];
					String[] words = line.split("\\s+");
					for(String w : words){
						vocabInText.addWordToVocab(w);
					}
				};
			});
			for(int i = 0 ; i < vocabInText.getTypes(); i++){
				WordEntry entry = vocabInText.getEntryFromId(i);
				if(entry.count >= minCount){
					vocab.add(entry.getWord());
				}
			}
		}
		else if(format.equals("vocab")){
			IOUtils.iterateFiles(vocabFile, new IOUtils.iterateFilesCallback(){
				@Override
				public void cb(String[] lines, int lineNumber) {
					String line = lines[0];
					String[] lineArray = line.split("\\s+");
					String word = lineArray[1];
					int count = Integer.parseInt(lineArray[0]);
					if(count >= minCount) vocab.add(word);
				}
			});
		}	
		else if(format.startsWith("conll")){
			LabelledData dataset = new LabelledData(vocabFile,format);
			Vocab vocabInText = new Vocab();

			for(LabelledSentence s : dataset.sentences){
				for(String w : s.tokens){
					vocabInText.addWordToVocab(w);
				}
			}
			for(int i = 0 ; i < vocabInText.getTypes(); i++){
				WordEntry entry = vocabInText.getEntryFromId(i);
				if(entry.count >= minCount){
					vocab.add(entry.getWord());
				}
			}
		}

		String distance = cmd.getOptionValue("distance");
		
		WordRepresentationLayer model = null;
		if(modelType.equals("sskip")){
			StructuredSkipngramSpecification spec = new StructuredSkipngramSpecification();		
			spec.wordFeatures = cmd.getOptionValue("word_features");
			model = StructuredSkipngram.loadRep(file, spec);
		}
		else{
			StructuredSkipngramSpecification spec = new StructuredSkipngramSpecification();		
			spec.wordFeatures = cmd.getOptionValue("word_features");
			WordRepresentationSetup wordSetup = new WordRepresentationSetup(null, spec.wordProjectionDim, spec.charProjectionDim, spec.charStateDim);
			wordSetup.loadFromString(spec.wordFeatures, spec.word2vecEmbeddings);
			BufferedReader in = IOUtils.getReader(file);
			model = WordRepresentationLayer.load(in, wordSetup);
		}

		System.out.println("building vectors for contrast ("+vocab.size() + " words)");
		model.fillCache(vocab, threads, false);		
		
		while(true){
			System.out.println("waiting for command ( type sim <word to search>, type quit to exit)");
			String command = null;
			if(mainCommand != null){
				command = mainCommand;
			}
			else{
				command = IOUtils.getLineFromInput();
			}
			String[] commandArgs = command.split("\\s+");			
			if(commandArgs[0].equals("sim") || commandArgs[0].equals("similarity")){
				String pivot = commandArgs[1];
				if(vocab.contains(pivot)){
					System.out.println("sim for word " + pivot + "(in vocab)");
				}
				else{
					System.out.println("sim for word " + pivot + "(out of vocab)");					
				}
				model.printSimilarityTable(pivot, vocab, System.out,distance);
			}
			if(commandArgs[0].equals("act") || commandArgs[0].equals("similarity")){
				String pivot = commandArgs[1];
				if(vocab.contains(pivot)){
					System.out.println("activations for word " + pivot + "(in vocab)");
				}
				else{
					System.out.println("activations for word " + pivot + "(out of vocab)");					
				}
				model.printSimilarityTable(pivot, vocab, System.out,distance);
			}
			else if(commandArgs[0].equals("write")){
				System.err.println("writing to " + file + ".vector");
				PrintStream out = IOUtils.getPrintStream(file+".vector");
				model.printVectors(vocab, out);
				out.close();
				out = IOUtils.getPrintStream(file + ".sim.gz");
				model.printSimilarityTable(1000000, 1000000, vocab,out);
				out.close();
			}
			else if(commandArgs[0].equals("quit")){
				System.out.println("thank you, have a great day!");
				break;
			}
			else{
				System.out.println("unrecognized command");
			}
			if(mainCommand != null){
				break;
			}
		}
	}
}
