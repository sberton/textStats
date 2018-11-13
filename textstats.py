from pyspark import SparkContext
import argparse
import re
import logging as lg

sc = SparkContext()

def filter_stop_words(partition):
    from nltk.corpus import stopwords
    english_stop_words = set(stopwords.words("english"))
    for word in partition:
        if word not in english_stop_words:
            yield word

def load_text(text_path):
    # Split text in words
    # Remove empty word artefacts
    # Remove stop words ('I', 'you', 'a', 'the', ...)
    pattern = re.compile(r"^((?!(@|\\)).)*$")

    vocabulary = sc.textFile(text_path, minPartitions=4)\
        .flatMap(lambda lines: lines.lower().split())\
        .flatMap(lambda word: word.split("."))\
        .flatMap(lambda word: word.split(","))\
        .flatMap(lambda word: word.split("!"))\
        .flatMap(lambda word: word.split("?"))\
        .flatMap(lambda word: word.split("'"))\
        .flatMap(lambda word: word.split("\""))\
        .filter(lambda word: word is not None and len(word) > 0)\
        .filter(lambda word : pattern.match(word))\
        .persist()
        #.mapPartitions(filter_stop_words)

    #compute length of each word
    most_freq_word_4 = vocabulary.filter(lambda word: len(word)==4).map(lambda word:(word,1)).reduceByKey(lambda cnt1, cnt2: cnt1 + cnt2)
    most_freq_word_10 = vocabulary.filter(lambda word: len(word)==10).map(lambda word:(word,1)).reduceByKey(lambda cnt1, cnt2: cnt1 + cnt2)
    #different_words = vocabulary.distinct()
    word_len = vocabulary.map(lambda word: (word,len(word)))

    return word_len,most_freq_word_4,most_freq_word_10

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--datafile",help="""Text File to analyze""")
    return parser.parse_args()

def main():
    args = parse_arguments()
    try:
        datafile = args.datafile
        if datafile == None:
            raise Warning('You must indicate a datafile!')
    except Warning as no_datafile:
        lg.warning(no_datafile)
    else:
        text_len,most_freq_word_4,most_freq_word_10 = load_text(datafile)
         # 10 words that get a decrease in frequency in the sequel
        longuest_word = text_len.takeOrdered(1,  key = lambda x: -x[1])
        word_4 = most_freq_word_4.takeOrdered(1,lambda (word, occ): -occ)
        word_10 = most_freq_word_10.takeOrdered(1,lambda (word, occ): -occ)
        # Print results
        for word, word_len in longuest_word:
            print("Longest word: ", word)
        for word, word_occ in word_4:
            print("Most frequent 4-letter word: ", word)
        for word, word_occ in word_10:
            print("Most frequent 10-letter word: ", word)
    finally:
        lg.info('#################### Analysis is over ######################')
    
    input("press ctrl+c to exit")

if __name__ == "__main__":
    main()