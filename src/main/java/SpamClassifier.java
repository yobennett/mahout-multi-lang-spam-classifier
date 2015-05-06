import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class SpamClassifier {

    private static final String WORK_DIR = "/tmp/mahout-work-bennett";
    private static final String MODEL_PATH = WORK_DIR + "/model";
    private static final String LABEL_INDEX_PATH = WORK_DIR + "/labelindex";
    private static final String DICTIONARY_PATH = WORK_DIR + "/spamassassin-vectors/dictionary.file-0";
    private static final String FREQUENCIES_PATH = WORK_DIR + "/spamassassin-vectors/frequency.file-0";

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

    public static void main(String[] args) {
        try {
            SpamClassifier sc = new SpamClassifier();
        } catch (IOException e) {
            System.out.println(e);
        }
    }

}
