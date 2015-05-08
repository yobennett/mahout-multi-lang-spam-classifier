package classifier.analyses;

import com.google.common.collect.Multiset;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.TFIDF;

import java.util.Map;

public class InstanceVectorCalculator {

    private static final int TFIDF_CARDINALITY = 10000;

    private final Vector vector;

    public InstanceVectorCalculator(Multiset<String> words, Map<String, Integer> dictionary, Map<Integer, Long> frequencies) {
        this.vector = calculateVector(words, dictionary, frequencies);
    }

    private Vector calculateVector(Multiset<String> words, Map<String, Integer> dictionary, Map<Integer, Long> frequencies) {
        int documentCount = frequencies.get(-1).intValue();

        // vector for term frequency inverse document frequency
        Vector v = new RandomAccessSparseVector(TFIDF_CARDINALITY);
        TFIDF tfidf = new TFIDF();
        for (Multiset.Entry<String> entry : words.entrySet()) {
            String word = entry.getElement();
            int count = entry.getCount();
            Integer id = dictionary.get(word);
            Long frequency = frequencies.get(id);
            double tfIdfValue = tfidf.calculate(count, frequency.intValue(), words.size(), documentCount);
            v.setQuick(id, tfIdfValue);
        }

        return v;
    }

    public Vector vector() {
        return vector;
    }


}
