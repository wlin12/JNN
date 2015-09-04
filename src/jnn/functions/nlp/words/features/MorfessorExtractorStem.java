package jnn.functions.nlp.words.features;

import java.util.HashMap;

import util.IOUtils;
import vocab.Vocab;
import vocab.WordEntry;

public class MorfessorExtractorStem implements FeatureExtractor{
	
	String vocabFile;	
	
	Vocab segmentVocab = new Vocab();
	HashMap<String, String> wordToPrefix = new HashMap<String, String>();
	HashMap<String, String> wordToSuffix = new HashMap<String, String>();
	HashMap<String, String> wordToStem = new HashMap<String, String>();
	
	public MorfessorExtractorStem(String vocabFile) {
		super();
		
		IOUtils.iterateFiles(vocabFile, new IOUtils.iterateFilesCallback(){
			@Override
			public void cb(String[] lines, int lineNumber) {
				String segmented = lines[0];
				for(String s : segmented.split("\\s+")){
					segmentVocab.addWordToVocab(s);
				}
			}
		});
		
		IOUtils.iterateFiles(vocabFile, new IOUtils.iterateFilesCallback(){
			@Override
			public void cb(String[] lines, int lineNumber) {
				String segmented = lines[0];
				String word = segmented.replaceAll(" ","");
				String[] segments = segmented.split("\\s+");
				WordEntry[] segmentEntries = new WordEntry[segments.length];
				
				for(int i = 0; i < segments.length; i++){					
					segmentEntries[i] = segmentVocab.getEntry(segments[i]);
				}

				int stemIndex = 0;
				long stemCount = segmentEntries[0].count;
				for(int i = 1; i < segments.length; i++){					
					if(segmentEntries[i].count < stemCount){
						stemCount = segmentEntries[i].count;
						stemIndex = i;
					}
				}
				wordToStem.put(word, segments[stemIndex]);
				if(stemIndex > 0){					
					wordToPrefix.put(word, segments[0]);
				}
				if(stemIndex < segments.length-1){
					wordToSuffix.put(word, segments[segments.length-1]);
				}
			}
		});
	}


	@Override
	public String extract(String word) {
		if(!wordToStem.containsKey(word)){
			return word;
		}
		return wordToStem.get(word);
	}
}

