package jnn.functions.nlp.aux.readers;

import java.io.File;
import java.util.ArrayList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import jnn.functions.nlp.aux.input.LabelledSentence;

import org.w3c.dom.Document;
import org.w3c.dom.NodeList;

import util.IOUtils;

public class NpsReader{

	public static void read(String file, LabelledReaderCallback cb) {
		String[] files = IOUtils.list(file);
		for(String annotationFile : files){
			if(annotationFile.endsWith(".xml")){
				String annotationFileFull = file + "/" + annotationFile;
				File fXmlFile = new File(annotationFileFull);
				DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
				DocumentBuilder dBuilder;
				try {
					dBuilder = dbFactory.newDocumentBuilder();
					Document doc = dBuilder.parse(fXmlFile);
					doc.getDocumentElement().normalize();
					NodeList posts = doc.getElementsByTagName("Post");
					for(int p = 0; p < posts.getLength(); p++){
						ArrayList<String> wordList = new ArrayList<String>();
						ArrayList<String> posList = new ArrayList<String>();
						NodeList words = posts.item(p).getChildNodes().item(1).getChildNodes();
						for(int w = 1; w < words.getLength(); w+=2){
							String pos = words.item(w).getAttributes().getNamedItem("pos").getTextContent();
							if(pos.startsWith("^")){
								pos = pos.substring(1,pos.length());
							}
							String word = words.item(w).getAttributes().getNamedItem("word").getTextContent();
							if(word.matches("[0-9][0-9]-[0-9][0-9]-[0-9][0-9]sUser[0-9]+")){
								word="<NPS-USER>";
							}
							if(pos.equals("")){
								System.err.println(word);
								System.err.println(annotationFile);
							}
							if(word.equals("&lt;")){
								word = "&";
							}
							wordList.add(word);
							posList.add(pos);
						}
						String original = posts.item(p).getTextContent().trim();
						cb.cb(new LabelledSentence(original, wordList.toArray(new String[0]), posList.toArray(new String[0])));
					}
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
			}
		}
	}
}
