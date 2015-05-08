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

public class EnglishAnalysis implements Analysis {

    private final Map<String, Integer> dictionary;
    private final Multiset<String> words;
    private final Vector instanceVector;

    public EnglishAnalysis(String instance, Map<String, Integer> dictionary, Map<Integer, Long> frequencies) throws IOException {
        this.dictionary = dictionary;
        this.words = words(instance);

        InstanceVectorCalculator calculator = new InstanceVectorCalculator(words, dictionary, frequencies);
        this.instanceVector = calculator.vector();
    }

    public Multiset<String> words(String text) throws IOException {
        Multiset<String> words = ConcurrentHashMultiset.create();
        Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_46);
        TokenStream tokenStream = analyzer.tokenStream("text", new StringReader(text));
        CharTermAttribute termAttribute = tokenStream.addAttribute(CharTermAttribute.class);
        tokenStream.reset();
        while (tokenStream.incrementToken()) {
            if (termAttribute.length() > 0) {
                String word = tokenStream.getAttribute(CharTermAttribute.class).toString();
                Integer id = dictionary.get(word);
                if (id != null) {
                    words.add(word);
                }
            }
        }
        analyzer.close();
        tokenStream.end();
        tokenStream.close();
        return words;
    }

    public Vector instanceVector() {
        return this.instanceVector;
    }

}
