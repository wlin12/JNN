package jnn.functions.nlp.aux.readers;

import java.util.ArrayList;

import jnn.functions.nlp.aux.input.LabelledSentence;
import util.IOUtils;

public class StanfordOriginalFormatReader {
	
	public static void read(String file, LabelledReaderCallback cb) {
		String[] files = IOUtils.list(file);
		for(String annotationFile : files){
			String annotationFileFull = file + "/" + annotationFile;
			String[] lines = IOUtils.readFileLines(annotationFileFull);
			ArrayList<String> wordList = new ArrayList<String>();
			ArrayList<String> posList = new ArrayList<String>();
			String original = "";
			for(int i = 0; i < lines.length; i++){
				String line = lines[i].trim();
				if(!line.isEmpty()){
					if(line.equals("======================================")){
						if(wordList.size()>0){
							cb.cb(new LabelledSentence(original.trim(), wordList.toArray(new String[0]), posList.toArray(new String[0])));						
						}
						wordList = new ArrayList<String>();
						posList = new ArrayList<String>();	
						original = ""; 
					}
					else if(line.contains("/")){
						String[] lineParts = line.split("\\s+");
						for(String part : lineParts){
							if(part.contains("/")){
								addWordParts(part, wordList, posList);
								original+=wordList.get(wordList.size()-1) + " ";
							}
						}
					}	
					else if(line.length()>0){
						throw new RuntimeException("unknown line type " + line);
					}
				}
			}
			cb.cb(new LabelledSentence(original.trim(), wordList.toArray(new String[0]), posList.toArray(new String[0])));						
		}
	}

	public static void addWordParts(String part, ArrayList<String> words, ArrayList<String> tags){
		String[] partArray = part.split("/");
		String tag = partArray[partArray.length-1];
		String word = part.substring(0, part.length()-tag.length()-1);
		if(tag.contains("|")){
			tag=tag.split("\\|")[0];
		}
		if(word.contains("\\/")){
			word.replace("\\/", "/");
		}
		words.add(word);
		tags.add(tag);
	}
}
