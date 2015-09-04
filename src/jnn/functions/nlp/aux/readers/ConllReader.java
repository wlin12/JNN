package jnn.functions.nlp.aux.readers;

import java.util.ArrayList;

import jnn.functions.nlp.aux.input.LabelledSentence;
import util.IOUtils;

public class ConllReader {
	public static void read(String file,int word, int pos, LabelledReaderCallback cb) {
		final ArrayList<String> curLines = new ArrayList<String>();
		IOUtils.iterateFiles(new String[]{file}, new IOUtils.iterateFilesCallback(){
			@Override
			public void cb(String[] lines, int lineNumber) {
				String line = lines[0];
				if (line.matches("^\\s*$")) {
					if (curLines.size() > 0) {
						// Flush
						cb.cb(getSentence(curLines,word, pos));
						curLines.clear();
					}
				} else {
					curLines.add(line);
				}

			}

			
		});
	}

	private static LabelledSentence getSentence(ArrayList<String> lines,int word, int pos) {
		ArrayList<String> tokens = new ArrayList<String>();
		ArrayList<String> tags = new ArrayList<String>();
		String original = "";
		for (String line : lines) {
			String[] parts = line.split("(\t|\\s)+");
			assert parts.length > pos;
			String token = parts[word].trim();
			String tag = parts[pos].trim();
			if(token.equals("&lt;")){
				token = "&";
			}
			tokens.add( token );
			tags.add( tag );
			original += token + " ";			
		}
		return new LabelledSentence(original.trim(), tokens.toArray(new String[0]), tags.toArray(new String[0]));
	}
}
