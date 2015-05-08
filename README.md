# Train Naive Bayes model with Spam Assassin corpus using Mahout

1. Install and compile Mahout

```
brew install mahout
mvn -DskipTests clean install
```

2. Set environment variables:

```
export HADOOP_HOME = /path/to/hadoop
export MAHOUT_HOME = /path/to/mahout
export MAHOUT_LOCAL = true
```

3. Copy SpamAssassin corpus to work directory

```
cp -R /path/to/corpus /tmp/mahout-work-{your username}/spamassassin-all
cp -R /path/to/corpus /tmp/mahout-work-{your username}/spamassassin-bayesinput
```

4. Run script to classify (from Mahout project source root directory)

```
examples/bin/classify-spamassassin.sh
```
