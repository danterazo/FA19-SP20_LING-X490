# LING-X 490
# This file builds and exports data
# Dante Razo, drazo
import pandas as pd
from kaggle_preprocessing import read_data, boost_data, sample_data

""" GLOBAL VARIABLES """
sample_size = 20000
verbose = True


# only import once
def get_train():
    dataset = "train.target+comments.tsv"  # 'test' for classification problem
    return read_data(dataset, delimiter="tab")


# Gets 'n' posts, randomly selected, from the dataset. Then save to `.csv`
def build_random(data):
    # sample + export
    random1 = sample_data(data, sample_size)
    random2 = sample_data(data, sample_size)
    random3 = sample_data(data, sample_size)
    export_data("random", [random1, random2, random3])
    pass


def build_boosted(data, manual_boost):
    data_file = "train.target+comments.tsv"  # name for verbose prints

    # sample + export, topic
    boosted_topic_data = boost_data(data, data_file, manual_boost)
    boosted_topic1 = sample_data(boosted_topic_data, sample_size)
    boosted_topic2 = sample_data(boosted_topic_data, sample_size)
    boosted_topic3 = sample_data(boosted_topic_data, sample_size)
    export_data("topic", [boosted_topic1, boosted_topic2, boosted_topic3])

    # boost + sample + export, wordbank
    boosted_wordbank_data = boost_data(data, data_file)
    boosted_wordbank1 = sample_data(boosted_wordbank_data, sample_size)
    boosted_wordbank2 = sample_data(boosted_wordbank_data, sample_size)
    boosted_wordbank3 = sample_data(boosted_wordbank_data, sample_size)
    export_data("wordbank", [boosted_wordbank1, boosted_wordbank2, boosted_wordbank3])

    # print(f"Verbose ")
    pass


# save data to `.tsv`, `.csv`, etc.
def export_data(source, data, extension=".csv"):
    i = 1

    for d in data:
        d.to_csv(f"train.{source}{i}{extension}", index=False, header=False)
        i += 1

    pass


# builds one or both
def build_main(choice, topic):
    """
    choice: choose which sample types to build. "random", "boosted", or "all"
    topic: topic for manual boosting
    """
    train = get_train()

    build_random(train) if choice is "random" or "all" else None
    build_boosted(train, topic) if choice is "boosted" or "all" else None
    print(f"\nDatasets built.") if verbose else None
    pass


""" MAIN """
# configuration
topic = ["trump"]  # [str]
to_build = "all"  # "all", "random", or "boosted"

# run
build_main(to_build, topic)
