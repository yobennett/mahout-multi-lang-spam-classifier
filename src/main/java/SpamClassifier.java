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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class SpamClassifier {

    private static final Logger LOGGER = LoggerFactory.getLogger(SpamClassifier.class);
    private static final String PROPERTIES_NAME = "classifier.properties";

    private final Configuration conf;
    private final Properties prop;
    private final NaiveBayesModel model;
    private final StandardNaiveBayesClassifier classifier;
    private final Map<Integer, String> labels;
    private final Map<String, Integer> dictionary;
    private final Map<Integer, Long> frequencies;

    public SpamClassifier() {
        this.conf = new Configuration();

        try {
            this.prop = loadProperties(PROPERTIES_NAME);
            this.model = NaiveBayesModel.materialize(new Path(prop.getProperty("classifier.model.dir")), conf);
            this.classifier = new StandardNaiveBayesClassifier(model);
            this.labels = readLabelIndex(new Path(prop.getProperty("classifier.model.labelindex")));
            this.dictionary = readDictionary(new Path(prop.getProperty("classifier.model.dictionary.path")));
            this.frequencies = readFrequencies(new Path(prop.getProperty("classifier.model.frequencies.path")));
        } catch (IOException e) {
            throw new IllegalStateException("Missing or invalid classifier properties at " + PROPERTIES_NAME + ".");
        }
    }

    private Properties loadProperties(String name) throws IOException {
        Properties properties = new Properties();
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        try (InputStream stream = loader.getResourceAsStream(name)) {
            properties.load(stream);
            return properties;
        }
    }

    private Map<Integer, String> readLabelIndex(Path path) {
        return BayesUtils.readLabelIndex(conf, path);
    }

    private Map<String, Integer> readDictionary(Path path) {
        Map<String, Integer> dict = new HashMap<>();
        String key;
        int value;
        for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(path, true, conf)) {
            key = pair.getFirst().toString();
            value = pair.getSecond().get();
            dict.put(key, value);
        }
        return dict;
    }

    private Map<Integer, Long> readFrequencies(Path path) {
        Map<Integer, Long> dict = new HashMap<>();
        int key;
        long value;
        for (Pair<IntWritable, LongWritable> pair : new SequenceFileIterable<IntWritable, LongWritable>(path, true, conf)) {
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
        }
        return labels.get(bestLabelId);
    }

    private String classifyFile(java.nio.file.Path path) throws IOException {
        byte[] encoded = Files.readAllBytes(path);
        String text = new String(encoded, StandardCharsets.UTF_8);
        String label = classify(text);
        LOGGER.info(label + ", " + path.toAbsolutePath());
        return label;
    }

    private void classifyDir(java.nio.file.Path dir) throws IOException {
        // try-with-resources automatically closes the DirectoryStream upon exit
        try (DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(dir)) {
            for (java.nio.file.Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    classifyDir(entry);
                } else {
                    classifyFile(entry);
                }
            }
        }
    }

    public static void main(String[] args) {
        try {
            SpamClassifier sc = new SpamClassifier();
            sc.classifyDir(Paths.get("data", "spamassassin"));
        } catch (IOException e) {
            LOGGER.error(e.getMessage());
        }
    }

}
