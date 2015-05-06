import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.TFIDF;

import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class SpamClassifier {

    private static final String WORK_DIR = "/tmp/mahout-work-bennett";
    private static final String MODEL_PATH = WORK_DIR + "/model";
    private static final String LABEL_INDEX_PATH = WORK_DIR + "/labelindex";
    private static final String DICTIONARY_PATH = WORK_DIR + "/spamassassin-vectors/dictionary.file-0";
    private static final String FREQUENCIES_PATH = WORK_DIR + "/spamassassin-vectors/df-count/part-r-00000";

    private final Configuration conf;
    private final NaiveBayesModel model;
    private final StandardNaiveBayesClassifier classifier;
    private final Map<Integer, String> labels;
    private final Map<String, Integer> dictionary;
    private final Map<Integer, Long> frequencies;

    public SpamClassifier() throws IOException {
        this.conf = new Configuration();
        this.model = NaiveBayesModel.materialize(new Path(MODEL_PATH), conf);
        this.classifier = new StandardNaiveBayesClassifier(model);
        this.labels = parseLabels(LABEL_INDEX_PATH);
        this.dictionary = parseDictionary(DICTIONARY_PATH);
        this.frequencies = parseFrequencies(FREQUENCIES_PATH);
    }

    private Map<Integer, String> parseLabels(String path) {
        return BayesUtils.readLabelIndex(conf, new Path(path));
    }

    private Map<String, Integer> parseDictionary(String path) {
        Map<String, Integer> dict = new HashMap<String, Integer>();
        String key;
        int value;
        for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(new Path(path), true, conf)) {
            key = pair.getFirst().toString();
            value = pair.getSecond().get();
            dict.put(key, value);
        }
        return dict;
    }

    private Map<Integer, Long> parseFrequencies(String path) {
        Map<Integer, Long> dict = new HashMap<Integer, Long>();
        int key;
        long value;
        for (Pair<IntWritable, LongWritable> pair : new SequenceFileIterable<IntWritable, LongWritable>(new Path(path), true, conf)) {
            key = pair.getFirst().get();
            value = pair.getSecond().get();
            dict.put(key, value);
        }
        return dict;
    }

    private String classify(String text) throws IOException {

        int documentCount = frequencies.get(-1).intValue();

        // extract words
        Multiset<String> words = ConcurrentHashMultiset.create();
        Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_46);
        TokenStream tokenStream = analyzer.tokenStream("text", new StringReader(text));
        CharTermAttribute termAttribute = tokenStream.addAttribute(CharTermAttribute.class);
        tokenStream.reset();
        int wordCount = 0;
        while (tokenStream.incrementToken()) {
            if (termAttribute.length() > 0) {
                String word = tokenStream.getAttribute(CharTermAttribute.class).toString();
                Integer id = dictionary.get(word);
                if (id != null) {
                    words.add(word);
                    wordCount++;
                }
            }
        }
        analyzer.close();
        tokenStream.end();
        tokenStream.close();

        // vector for term frequency inverse document frequency
        Vector vector = new RandomAccessSparseVector(10000);
        TFIDF tfidf = new TFIDF();
        for (Multiset.Entry<String> entry : words.entrySet()) {
            String word = entry.getElement();
            int count = entry.getCount();
            Integer id = dictionary.get(word);
            Long frequency = frequencies.get(id);
            double tfIdfValue = tfidf.calculate(count, frequency.intValue(), wordCount, documentCount);
            vector.setQuick(id, tfIdfValue);
        }

        Vector resultVector = classifier.classifyFull(vector);
        double bestScore = -Double.MAX_VALUE;
        int bestLabelId = -1;
        for (Vector.Element e : resultVector.all()) {
            int labelId = e.index();
            double score = e.get();
            if (bestScore < score) {
                bestScore = score;
                bestLabelId = labelId;
            }
            System.out.println(labels.get(labelId) + ": " + score);
        }
        System.out.println("* best label: " + labels.get(bestLabelId));
        return labels.get(bestLabelId);
    }

    private String classify(java.nio.file.Path path) throws IOException {
        byte[] encoded = Files.readAllBytes(path);
        String text = new String(encoded, StandardCharsets.UTF_8);
        return classify(text);
    }

    public static void main(String[] args) {
        try {
            SpamClassifier sc = new SpamClassifier();

            String[] paths = new String[] {
                "/tmp/mahout-work-bennett/spamassassin-all/ham/2551.3b1f94418de5bd544c977b44bcc7e740",
                "/tmp/mahout-work-bennett/spamassassin-all/spam/0210.285d263b1a18e67c68ec9fe005253dd0",
                "/Users/bennett/Code/src/github.com/yobennett/mahout-spam-classifier/src/main/resources/ham/aphyr-jepsen-aerospike.txt",
                "/Users/bennett/Code/src/github.com/yobennett/mahout-spam-classifier/src/main/resources/spam/britney-and-friends.txt",
                "/Users/bennett/Code/src/github.com/yobennett/mahout-spam-classifier/src/main/resources/spam/court-notice.txt",
                "/Users/bennett/Code/src/github.com/yobennett/mahout-spam-classifier/src/main/resources/spam/hearthstone-kr-spam-01.txt",
                "/Users/bennett/Code/src/github.com/yobennett/mahout-spam-classifier/src/main/resources/spam/hearthstone-kr-spam-02.txt"
            };

            for (String path : paths) {
                System.out.println("Classifying " + path + "...");
                sc.classify(Paths.get(path));
            }

        } catch (IOException e) {
            System.out.println(e);
        }
    }

}
