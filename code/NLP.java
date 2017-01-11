import java.io.*;
import java.lang.*;
import java.util.*;

public class NLP
{
	//Debug level -- 1 means test train.txt; 2 means some procedure just take only ONE line
	private static final int Debug = 0;

	//Feature description -- unigram/bigram/trigram
	private static final int[][] feature = {{-1}, {0}, {1}, {-1, 0}, {-1, 1}, {0, 1}, {-1, 0, 1}};
	private static final int NofFea = feature.length;

	//File path & Encoding type
	private static final String trainFileName = "./train.txt"; // train text
	private static final String testFileName = "./test.txt"; // test text
	private static final String logFileName = "./log.txt"; // save the theta vector
	private static final String outFileName = "./out.txt"; // output the test result
	private static final String encoding = "UTF-8";

	//B-M-E-S tags. '$' for separation tag
	private static final char sepTag = '$';
	private static final char bgnTag = 'B';
	private static final char midTag = 'M';
	private static final char endTag = 'E';
	private static final char sglTag = 'S';

	//the number of iteration rounds. the reliability of this is proved in the report
	private static final int NofRound = 150;

	//trainX is the sequence of all sentences. Sentences are separated by '$'
	//trainY is the sequence of tags for all characters.
	private static String trainX = null;
	private static String trainY = null;

	// N for the length of trainX&trainY.
	// Dim for how many features there are
	// theta is the weighted vector we trained. Initial vector is a zero vector.
	private static int N, Dim;
	private static int[] theta;

	// vec[i][j] for whether the i-th character has the j-th feature
	private static String[][] vec = null;
	//hash save the serial number of features. e.g. 3_触_猪 is the third feature
	private static HashMap<String, Integer> hash = null;

	//Read from file and add tag to it. implemented by LIYANHAO
	private static void readTrainFile(){
		try{
			File file = new File(trainFileName);
			if(file.isFile() && file.exists()){
				StringBuffer tmpTrainX = new StringBuffer();
				StringBuffer tmpTrainY = new StringBuffer();
				BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(file), encoding));
				String line = null;
				while((line = bufferedReader.readLine()) != null){
					//append sepTag
					tmpTrainX.append(sepTag);
					tmpTrainY.append(sepTag);
					String[] token = line.split(" ");
					for (int i = 0; i < token.length; ++i){
						int len = token[i].length();
						if (len == 0)
							continue;
						//append the word to trainX
						tmpTrainX.append(token[i]);
						//append sglTag to trainY if it's a single-character word
						if (len == 1)
							tmpTrainY.append(sglTag);
						else{
							// append bgnTag to trainY for its beginning character
							tmpTrainY.append(bgnTag);
							// append bgnTag to trainY for characters within the word
							for (int  j = 1; j < len-1; ++j)
								tmpTrainY.append(midTag);
							// append bgnTag to trainY for its ending character
							tmpTrainY.append(endTag);
						}
					}
					if (Debug >= 2) break;
				}
				bufferedReader.close();
				tmpTrainX.append(sepTag);
				tmpTrainY.append(sepTag);
				// transfer them into real string
				trainX = new String(tmpTrainX);
				trainY = new String(tmpTrainY);
				// panic if the length of trainX is NOT equal to the length of trainY
				assert(trainX.length() == trainY.length());
				N = trainY.length();
			}
			else{
				System.out.println("File Not Found");
				assert(false);
			}
		}catch (Exception e){
			System.out.println("Error: readTrainFile()");
			e.printStackTrace();
		}
	}

	//Build Feature Vector. Implemented by LICAIHUA
	private static void buildFeatureVector()
	{
		vec = new String[N][NofFea];
		hash = new HashMap<String, Integer>();
		// traverse all characters
		for (int i = 0; i < N; ++i)
		{	
			//ignore the sepTag
			if (sepTag == trainY.charAt(i))
			{
				for (int j = 0; j < NofFea; ++j)
					vec[i][j] = null;
				continue;
			}
			//else traverse all kind of features
			for (int j = 0; j < NofFea; ++j)
			{
				StringBuffer buffer = new StringBuffer();
				buffer.append(j);
				int tmp = feature[j].length;
				// extract features and name them
				for (int k = 0; k < tmp; ++k)
				{
					buffer.append('_');
					buffer.append(trainX.charAt(i + feature[j][k]));
				}
				vec[i][j] = new String(buffer);
				hash.put(vec[i][j], new Integer(0));
			}
		}
		// number all the feature we extracted
		Dim = 0;
		Set<Map.Entry<String,Integer>> sets = hash.entrySet();
		for (Map.Entry<String,Integer> entry : sets)
			hash.put(entry.getKey(), Dim++);
	}

	// Train the weighted vector. Implemented by LICAIHUA
	private static void trainClassifier()
	{
		//Init the vector as zero vector
		theta = new int[Dim];
		for (int i = 0; i < Dim; ++i)
			theta[i] = 0;
		//Iterate for NofRound times
		for (int round = 0; round < NofRound; ++round)
		{
			// Printing iteration info
			System.out.println("Iterating: ROUND: " + round);
			int wrongNumber = 0;
			// traverse all characters in the sequence
			for (int i = 0; i < N; ++i)
			{
				char ch = trainY.charAt(i);
				if (ch == sepTag) continue;
				//separate the word if ch is the beginning of the word OR itself forms a word.
				int belong = (ch == bgnTag || ch == sglTag)? 1:-1;
				int score = 0;
				//calc the score
				for (int j = 0; j < NofFea; ++j)
				{
					int index = hash.get(vec[i][j]);
					score += theta[index];
				}
				// see if it is classified correctly
				score = (score > 0)? 1:-1;
				if (score != belong)
				{
					// if not, adjust the vector
					wrongNumber++;
					for (int j = 0; j < NofFea; ++j)
					{
						int index = hash.get(vec[i][j]);
						theta[index] += belong;
					}
				}
			}
			// After a number of iteration, the wrong point percentage will be on a stable stage.
			System.out.printf("Wrong Percentage: %f\n", (double)wrongNumber/N);
		}

		try
		{
			//save the weighted vector we got
			File file = new File(logFileName);
			if (!file.exists()) file.createNewFile();
			PrintWriter fout = new PrintWriter(file);
			Set<Map.Entry<String,Integer>> sets = hash.entrySet();
			for (Map.Entry<String,Integer> entry : sets)
			{
				String token = entry.getKey();
				int index = hash.get(token);
				fout.print(token + " ");
				fout.println(theta[index]);
			}
			fout.close();
		} catch (Exception e)
		{
			System.out.println("Error: trainClassifier()");
			e.printStackTrace();
		}
	}

	// Called when using train.txt as test text
	private static String removeSpace(String line)
	{
		String[] token = line.split(" ");
		StringBuffer result = new StringBuffer();
		for (int i = 0; i < token.length; ++i)
			if (token[i].length() > 0)
				result.append(token[i]);
		return new String(result);
	}

	// Get the separation result as a string.
	private static String testResult(String line)
	{
		//remove the space in the train.txt when debugging
		if (Debug >= 1)
		{
			line = removeSpace(line);
			if (Debug >= 2) System.out.println(line);
		}
		StringBuffer result = new StringBuffer();
		String testX = sepTag + line + sepTag;
		int N = testX.length();
		result.append(testX.charAt(1));
		// see if the word in test.txt is the point to cut
		for (int i = 2; i < N-1; ++i)
		{
			int score = 0;
			for (int j = 0; j < NofFea; ++j)
			{
				StringBuffer buffer = new StringBuffer();
				buffer.append(j);
				int tmp = feature[j].length;
				//extract features
				for (int k = 0; k < tmp; ++k)
				{
					buffer.append('_');
					buffer.append(testX.charAt(i + feature[j][k]));
				}
				String token = new String(buffer);
				// calc score
				if (hash.containsKey(token))
					score += theta[hash.get(token)];
			}
			// predict. if score > 0 then append space in front of it, which means it should be separated
			if (score > 0)
				result.append("  ");
			result.append(testX.charAt(i));
		}
		return new String(result);
	}

	//Classify text.txt. Implemented by LICAIHUA
	private static void testClassifier()
	{
		try
		{
			//open a file to write result
			File oFile = new File(outFileName);
			if (!oFile.exists()) oFile.createNewFile();
			PrintWriter fout = new PrintWriter(new OutputStreamWriter(new FileOutputStream(oFile), encoding));
			File iFile = null;
			//test.txt or train.txt to predict
			if (Debug >= 1)
				iFile = new File(trainFileName);
			else
				iFile = new File(testFileName);
			if(iFile.isFile() && iFile.exists())
			{
				BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(iFile), encoding));
				String line = null;
				while((line = bufferedReader.readLine()) != null)
				{
					//predict
					fout.println(testResult(line));
					if (Debug >= 2) break;
				}
				bufferedReader.close();
			}
			else
			{
				System.out.println("File Not Found");
				assert(false);
			}
			fout.close();
		} catch (Exception e)
		{
			System.out.println("Error: testClassifier()");
			e.printStackTrace();
		}
	}

	//Main entrance that describe the experiment procedure.
	public static void main(String[] args)
	{
		System.out.println("Reading from files");
		readTrainFile();
		System.out.println("Building Feature Vector");
		buildFeatureVector();
		System.out.println("Training");
		trainClassifier();
		System.out.println("Testing");
		testClassifier();
	}
}