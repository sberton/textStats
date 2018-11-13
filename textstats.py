from pyspark import SparkContext
import argparse
import logging as lg

sc = SparkContext()

def filter_stop_words(word):
    from nltk.corpus import stopwords
    english_stop_words = stopwords.words("english")
    return word not in english_stop_words

def load_text(text_path):
    # Split text in words
    # Remove empty word artefacts
    # Remove stop words ('I', 'you', 'a', 'the', ...)
    vocabulary = sc.textFile(text_path)\
        .flatMap(lambda lines: lines.lower().split())\
        .flatMap(lambda word: word.split("."))\
        .flatMap(lambda word: word.split(","))\
        .flatMap(lambda word: word.split("!"))\
        .flatMap(lambda word: word.split("?"))\
        .flatMap(lambda word: word.split("'"))\
        .flatMap(lambda word: word.split("\""))\
        .filter(lambda word: word is not None and len(word) > 0)\
        .filter(filter_stop_words)

    # Count the total number of words in the text
    word_count = vocabulary.count()

    # Compute the frequency of each word: frequency = #appearances/#word_count
    word_freq = vocabulary.map(lambda word: (word, 1))\
        .reduceByKey(lambda count1, count2: count1 + count2)\
        .map(lambda (word, count): (word, count/float(word_count)))\

    return word_freq

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
        load_text(datafile)
    finally:
        lg.info('#################### Analysis is over ######################')
    
    input("press ctrl+c to exit")

if __name__ == "__main__":
    main()