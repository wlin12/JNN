package jnn.functions.nlp.aux.readers;

import jnn.functions.nlp.aux.input.LabelledSentence;
import util.IOUtils;

public class StanfordPosReader {
	
	public static void read(String file, LabelledReaderCallback cb) {
		IOUtils.iterateFiles(new String[]{file}, new IOUtils.iterateFilesCallback(){
			@Override
			public void cb(String[] lines, int lineNumber) {
				String line = lines[0];
				
				cb.cb(lineToPosTaggedSentence(line));
			}
		});
	}
	
	public static LabelledSentence lineToPosTaggedSentence(String line){
		String[] tokenAndPos = line.split("\\s+");
		String[] tokens = new String[tokenAndPos.length];
		String[] tags = new String[tokenAndPos.length];
		
		String original = "";
		for(int i = 0; i < tokens.length; i++){
			String[] split = tokenAndPos[i].split("_");
			String tag = split[split.length-1];
			String token = tokenAndPos[i].substring(0, tokenAndPos[i].length() - tag.length() - 1);
			if(token.equals("&lt;")){
				token = "&";
			}
			tags[i]=tag;
			tokens[i]=token;
			
			original += token+ " ";
		} 
		return new LabelledSentence(original.trim(), tokens, tags);
	}
}
