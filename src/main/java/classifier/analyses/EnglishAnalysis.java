package classifier.analyses;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.io.StringReader;
import java.util.Map;

public class EnglishAnalysis implements Analysis<String> {

    private final Map<String, Integer> dictionary;
    private final Multiset<String> words;
    private final Vector instanceVector;

    public EnglishAnalysis(String instance, Map<String, Integer> dictionary, Map<Integer, Long> frequencies) throws IOException {
        this.dictionary = dictionary;
        this.words = extractWords(instance);
        this.instanceVector = calculateInstanceVector(dictionary, frequencies);
    }

	/**
	 * Analyzes the instance text and returns a set of "significant" words for classification
	 * @param text the instance text
	 * @return a set of {@link java.lang.String}s words for classification
	 * @throws IOException
	 */
    private Multiset<String> extractWords(String text) throws IOException {
        Multiset<String> resultWords = ConcurrentHashMultiset.create();
        Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_46);
        TokenStream tokenStream = analyzer.tokenStream("text", new StringReader(text));
        CharTermAttribute termAttribute = tokenStream.addAttribute(CharTermAttribute.class);
        tokenStream.reset();
        while (tokenStream.incrementToken()) {
            if (termAttribute.length() > 0) {
                String word = tokenStream.getAttribute(CharTermAttribute.class).toString();
                Integer id = dictionary.get(word);
                if (id != null) {
                    resultWords.add(word);
                }
            }
        }
        analyzer.close();
        tokenStream.end();
        tokenStream.close();
        return resultWords;
    }

	/**
	 * Calculates the {@link org.apache.mahout.math.Vector} for an instance
	 * @param dictionary map from word to word id
	 * @param frequencies map from word id to frequency count
	 * @return resulting {@link org.apache.mahout.math.Vector} for instance
	 */
	private Vector calculateInstanceVector(Map<String, Integer> dictionary, Map<Integer, Long> frequencies) {
		InstanceVectorCalculator calculator = new InstanceVectorCalculator(words, dictionary, frequencies);
		return calculator.vector();
	}

	/**
	 * Returns the instance vector for the target instance
	 * @return instance vector
	 */
    public Vector instanceVector() {
        return this.instanceVector;
    }

}
