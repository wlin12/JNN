package jnn.functions.nlp.aux.readers;

import java.util.ArrayList;

import jnn.functions.nlp.aux.input.LabelledSentence;
import util.IOUtils;

public class ParallelFormatReader {
	public static void read(String file, LabelledReaderCallback cb) {
		IOUtils.iterateFiles(new String[]{file}, new IOUtils.iterateFilesCallback(){
			@Override
			public void cb(String[] lines, int lineNumber) {
				String line = lines[0];
				String[] sourceTarget = line.split("\\s+\\|\\|\\|\\s+");
				if(sourceTarget.length>0){
					String[] textTokens = sourceTarget[0].split("\\s+"); 
					String[] tagTokens = sourceTarget[1].split("\\s+");
					cb.cb(new LabelledSentence(line, textTokens, tagTokens));
				}
			}			
		});
	}
}
