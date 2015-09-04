package util;

import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.management.RuntimeErrorException;

import util.twitter.Twokenize;

public class LangUtils {
	public static String CHINESETAG = "CN";
	public static String ENGLISHTAG = "EN";
	public static String ARABICTAG = "AR";
	public static String JAPANESETAG = "JP";

	static Pattern punct = Pattern.compile("\\p{P}|\\p{Punct}");
	static Pattern han = Pattern.compile("[\\u4e00-\\u9fa5]");
	static Pattern words = Pattern.compile("\\p{Alnum}+s'|\\p{Alnum}+|[‚Äô']([dmst]|re|ve|ll)");  
	static Pattern latin = Pattern.compile("(\\p{Alpha}|[\u00C0-\u024F])+");  
	static Pattern numbers = Pattern.compile("\\p{Digit}+");
	static Pattern hangul = Pattern.compile("[\u3131-\u3163]|[\uac00-\ud7a3]");
	static Pattern hiraKata = Pattern.compile("[\u3041-\u30FC]");
	static Pattern arabic = Pattern.compile("[\u0600-\u0700]+");
	static Pattern emoticons = Pattern.compile("[\u1F600]-[\u1F64F]");
	static Pattern emoticonsComplex = Pattern.compile(Twokenize.emoticon);
	static Pattern chinesePunct = Pattern.compile("。|「|」|﹁|﹂|“|”|、|‧| 《|》|…|—|～|？|；|！|（|）|【|】|［|］");
	static Pattern cyrillic = Pattern.compile("([\u0400-\u052F])+");
	static Pattern armenian = Pattern.compile("[\u0530-\u058F]+");

	static Pattern punctString = Pattern.compile("\\S*(\\p{P}|\\p{Punct})\\S*");
	static Pattern numberString = Pattern.compile("\\S*\\p{Digit}\\S*");
	static Pattern vowelString = Pattern.compile("\\S*[aeiouAEIOU]\\S*");
	static Pattern link = Pattern.compile("[0-9a-zA-Z:/]+(\\.[0-9a-zA-Z/]+)(\\.[0-9a-zA-Z/]+)+");
	static Pattern link2 = Pattern.compile("(http:)[0-9a-zA-Z/]+(\\.[0-9a-zA-Z/]+)+");
	static Pattern email = Pattern.compile(Twokenize.Email);
	static Pattern url = Pattern.compile(Twokenize.url);

	public static boolean isHan(String s){
		Matcher m = han.matcher(s);
		return m.matches();
	}

	public static boolean isChinesePunct(String s){
		Matcher m = chinesePunct.matcher(s);
		return m.matches();
	}

	public static boolean isNonChinesePunct(String s){
		return isPunct(s) && !isChinesePunct(s);
	}

	public static boolean isEng(String s){
		Matcher m = latin.matcher(s);
		return m.matches();
	}

	public static boolean isLatin(String s){
		Matcher m = latin.matcher(s);
		return m.matches();
	}

	public static boolean isPunct(String s){
		Matcher m = punct.matcher(s);
		return m.matches();
	}

	public static boolean isHangul(String s){
		Matcher m = hangul.matcher(s);
		return m.matches();
	}

	public static boolean isArabic(String s) {
		Matcher m = arabic.matcher(s);
		return m.matches();
	}

	public static boolean isKana(String s){
		Matcher m = hiraKata.matcher(s);
		return m.matches();
	}

	public static boolean isCyrillic(String s){
		Matcher m = cyrillic.matcher(s);
		return m.matches();		 
	}

	public static boolean isCapitalized(String s){
		if(s.length() == 0) return false;
		return Character.isUpperCase(s.charAt(0));
	}

	public static boolean isUppercased(String s) {
		int uppercase = 0;
		int letters = 0;
		if(s.length() == 1){
			return false;
		}
		for(int i = 0; i < s.length(); i++){
			char c = s.charAt(i);
			if(Character.isAlphabetic(c)){
				letters++;
				if(Character.isUpperCase(c)){
					uppercase++;
				}
			}
		}
		return uppercase/(double)letters > 0.8;		
	}

	public static boolean isNumber(String s) {
		Matcher m = numbers.matcher(s);
		return m.matches();
	}

	public static boolean isEmoticon(String s){
		Matcher m = emoticons.matcher(s);
		return m.matches();
	}

	public static boolean isEmoticonTobi(String s){		
		Matcher m = emoticonsComplex.matcher(s);
		return m.matches();
	}

	public static boolean isHashTag(String string) {
		return string.startsWith("#") && string.length()>1;
	}

	public static boolean isAtMention(String string) {
		return string.startsWith("@") && string.length()>1;
	}

	public static boolean isLink(String string) {
		Matcher mtobi = url.matcher(string);
		Matcher m = link.matcher(string);
		Matcher m2 = link2.matcher(string);
		return m.matches() || m2.matches() || mtobi.matches();
	}

	public static boolean isEmail(String string){
		Matcher emailTobi = email.matcher(string);
		return emailTobi.matches();
	}

	public static boolean isWord(String string){
		return !isHashTag(string) && !isEmoticon(string) && !isPunct(string) && !isNumber(string) && !isLink(string);
	}

	public static boolean isArmenian(String s) {
		Matcher m = armenian.matcher(s);
		return m.matches();
	}

	public static boolean containsPunct(String s) {
		Matcher m = punctString.matcher(s);
		return m.matches();
	}

	public static boolean containApos(String original) {
		original = original.replaceAll("[‘’´`]", "'").replaceAll("[“”]", "\"");
		if(original.length() > 2 && (original.contains("‘"))){
			return true;
		}
		return false;
	}
	
	public static boolean containsValSymbol(String original) {
		if(original.contains("£") || original.contains("$") || original.contains("%")){
			return true;			
		}
		return false;
	}

	public static boolean containsNumber(String s) {
		Matcher m = numberString.matcher(s);
		return m.matches();
	}
	
	public static boolean containsVowel(String s){
		Matcher m = vowelString.matcher(s);
		return m.matches();		
	}

	public static String capitalizeWord(String lowercase) {
		char c = lowercase.charAt(0);
		char capC = Character.toUpperCase(c);
		return capC + lowercase.substring(1, lowercase.length());
	}


	public static String noRepeat(String form) {
		String ret = "";
		for(int i = 0; i < form.length(); i++){
			if(i==0 || form.charAt(i) != form.charAt(i-1)){
				ret+=form.charAt(i);
			}
		}
		return ret;
	}

	public static interface FuzzyMatchingCheck{
		public boolean check(String form);
		public void found(String form);
		public void notFound();
	}
	
	public static void fuzzyMatching(String word, FuzzyMatchingCheck check){
		HashSet<String> checkedForms = new HashSet<String>();
		String form = word.replaceAll("[‘’´`]", "'").replaceAll("[“”]", "\"");
		
		if(!checkedForms.contains(form) && check.check(form)){check.found(form); return;}else{checkedForms.add(form);}
		form = word.toLowerCase();
		if(!checkedForms.contains(form) && check.check(form)){check.found(form); return;}else{checkedForms.add(form);}
		if(form.length()>5){
			//remove multiple repetitions
			String start = "";
			for(int i = 0; i < form.length(); i++){
				if(i == 0 || form.charAt(i) != form.charAt(i-1)){
					start += form.charAt(i);					
				}
				String end = "";
				if(i!=form.length()-1){
					end = form.substring(i+1, form.length());
				}
				String fuzzyWord = start + end; 
				if(!checkedForms.contains(fuzzyWord) && check.check(fuzzyWord)){check.found(fuzzyWord); return;}else{checkedForms.add(fuzzyWord);}
			}
			String end = "";
			for(int i = form.length()-1; i >= 0; i--){
				if(i == form.length()-1 || form.charAt(i) != form.charAt(i+1)){
					end = form.charAt(i) + end;					
				}
				start = "";
				if(i!=0){
					start = form.substring(0, i);
				}
				String fuzzyWord = start + end; 
				if(!checkedForms.contains(fuzzyWord) && check.check(fuzzyWord)){check.found(fuzzyWord); return;}else{checkedForms.add(fuzzyWord);}
			}	
			
			//no punct
			String[] splitedByPunct = form.split("\\p{Punct}");
			int longest = 0;
			String longestSplit = "";
			for(String split : splitedByPunct){
				if(split.length() > longest){
					longestSplit = split;
					longest = split.length();
				}
			}
			if(longest > 0){
				if(!checkedForms.contains(longestSplit) && check.check(longestSplit)){check.found(longestSplit); return;}else{checkedForms.add(longestSplit);}
			}
		}		
		if(form.length()>5){
			// remove one letter
//			for(int i = 0; i < form.length(); i++){
//				String start = "";
//				if(i>0){
//					start = form.substring(0, i);
//				}
//				String end = "";
//				if(i < form.length()-1){
//					end = form.substring(i+1, form.length());
//				}
//				String fuzzyWord = start + end;
//				if(!checkedForms.contains(fuzzyWord) && check.check(fuzzyWord)){check.found(fuzzyWord); return;}else{checkedForms.add(fuzzyWord);}
//			}
		}
		check.notFound();
	}

	public static String[] splitWordIntoParts(String form){
		ArrayList<String> sequence = new ArrayList<String>();
		String prev = String.valueOf(form.charAt(0));
		String word = prev;
		for(int i = 1; i < form.length(); i++){
			String c = String.valueOf(form.charAt(i));
			boolean split = false;
			if(LangUtils.isPunct(c) && !LangUtils.isPunct(prev)){
				split =true;
			}
			else if(LangUtils.isNumber(c) && !LangUtils.isNumber(prev)){
				split = true;
			}
			else if(LangUtils.isLatin(c) && !LangUtils.isLatin(prev)){
				split = true;
			}

			if(split){
				sequence.add(word);
				word = "";
			}
			word+=c;
			prev = c;
		}
		if(word.length()!=0){
			sequence.add(word);
		}
		return sequence.toArray(new String[]{});
	}
	
	public static String[] lowercase(String[] words){
		for(int i = 0; i < words.length; i++){
			words[i] = words[i].toLowerCase();
		}
		return words;
	}
	
	public static boolean isStringEmpty(String str){
		return str.split("\\s+").length == 0 || str.split("\\s+")[0].length() == 0;
	}
	
	public static String[] splitWord(String inputWord){
		ArrayList<String> letters = new ArrayList<String>();
		int prev = 2;
		for(int i = 0; i < inputWord.length(); i++){
			int len = 2;
			String word = null;
			if(i < inputWord.length()-1){				
				word = inputWord.substring(i, i+2);
				len = word.getBytes().length;
			}
			if(len == 2 || len == 3){
				if(prev == 2 || prev == 3){
					letters.add(String.valueOf(inputWord.charAt(i)));
				}
			}
			else if(len >= 4){
			    try {
					String out = new String(String.valueOf(inputWord.charAt(i)).getBytes(), "UTF-8");
					if(out.equals(String.valueOf(inputWord.charAt(i)))){
						letters.add(String.valueOf(inputWord.charAt(i)));
						len=2;
					}
					else{
						letters.add(word);				
					}

				} catch (UnsupportedEncodingException e) {
					throw new RuntimeException(e);
				}
			}
			else{
				throw new RuntimeException("problem in split with word " + inputWord);
			}
			prev = len;
		}
		String[] letterArray = new String[letters.size()];
		for(int i = 0; i < letters.size(); i++){
			letterArray[i] = letters.get(i);
		}
		if(letterArray.length == 0){
			throw new RuntimeException("returned empty array for input " + inputWord);
		}

		return letterArray;
	}
	
	public static void main(String[] args){
		String i = "Định 😳 wtffffffff";
		System.err.println(StringUtils.arrayToString(splitWord(i)));
		
	}
}
