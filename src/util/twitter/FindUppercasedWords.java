package util.twitter;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map.Entry;

import util.IOUtils;
import util.LangUtils;

public class FindUppercasedWords {
	static HashMap<String, int[]> counts = new HashMap<String, int[]>();
	public static void main(String[] args){
		String file = "/Users/lingwang/Documents/workspace/ContinuousVectors/twitter/context/twitter.txt";
		String output = "/Users/lingwang/Documents/workspace/ContinuousVectors/twitter/uppercase.txt";
		IOUtils.iterateFiles(new String[]{file}, new IOUtils.iterateFilesCallback(){
			@Override
			public void cb(String[] lines, int lineNumber) {
				String line = lines[0];
				String[] words = line.split("\\s+");
				for(String word : words){
					String lowercased = word.toLowerCase();
					if(!counts.containsKey(lowercased)){
						counts.put(lowercased, new int[2]);
					}
					int[] count = counts.get(lowercased);
					count[0]++;
					if(LangUtils.isCapitalized(word)){
						count[1]++;
					}
				}
			}
		});
		PrintStream out = IOUtils.getPrintStream(output);
		for(Entry<String, int[]> entry : counts.entrySet()){
			int[] count = entry.getValue();
			if(count[0]<5){
				continue;
			}
			out.println(count[1]/(double)count[0] + " ||| " + entry.getKey());
		}
		out.close();
	}
}
