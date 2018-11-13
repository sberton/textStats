from pyspark import SparkContext
import argparse
import re
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
    pattern = re.compile(r"^((?!(@|\\)).)*$")

    vocabulary = sc.textFile(text_path)\
        .flatMap(lambda lines: lines.lower().split())\
        .flatMap(lambda word: word.split("."))\
        .flatMap(lambda word: word.split(","))\
        .flatMap(lambda word: word.split("!"))\
        .flatMap(lambda word: word.split("?"))\
        .flatMap(lambda word: word.split("'"))\
        .flatMap(lambda word: word.split("\""))\
        .filter(lambda word: word is not None and len(word) > 0)\
        .filter(lambda word : pattern.match(word))\
        .filter(filter_stop_words)

    #compute length of each word
    different_words = vocabulary.distinct()
    word_len = different_words.map(lambda word: (word,len(word)))

    return word_len

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
        text_len = load_text(datafile)
         # 10 words that get a decrease in frequency in the sequel
        longuest_word = text_len.takeOrdered(10,  key = lambda x: -x[1])

        # Print results
        for word, word_len in longuest_word:
            print("Longest word: ", word)
    finally:
        lg.info('#################### Analysis is over ######################')
    
    input("press ctrl+c to exit")

if __name__ == "__main__":
    main()