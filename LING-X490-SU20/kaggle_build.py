# LING-X 490
# This file builds and exports data
# Dante Razo, drazo
import os
from datetime import date
from kaggle_preprocessing import read_data, boost_data, sample_data


# data headers: [y, X]

# only import once
def get_train():
    dataset = "train.target+comments.tsv"  # 'test' for classification problem
    return read_data(dataset, delimiter="tab")


# Gets 'n' posts, randomly selected, from the dataset. Then save to `.csv`
def build_random(data, sample_size, repeats=3):
    to_export = []
    # sample + export
    for i in range(0, repeats):
        to_export.append(sample_data(data, sample_size))

    export_data("random", to_export)


def build_boosted(data, manual_boost, sample_size, repeats=3):
    data_file = "train.target+comments.tsv"  # name for verbose prints
    to_export = []

    # sample + export, topic
    boosted_topic_data = boost_data(data, data_file, manual_boost)
    for i in range(0, repeats):
        to_export.append(sample_data(boosted_topic_data, sample_size))

    export_data("topic", to_export)

    # boost + sample + export, wordbank
    boosted_wordbank_data = boost_data(data, data_file)

    for i in range(0, repeats):
        to_export.append(sample_data(boosted_wordbank_data, sample_size))

    export_data("wordbank", to_export)


# save data to `.tsv`, `.csv`, etc.
def export_data(source, data, extension=".csv"):
    i = 1

    for d in data:
        filepath = os.path.join("../data/kaggle_data", f"train.{source}{i}{extension}")
        d.to_csv(filepath, index=False, header=False)
        i += 1


# generalized version of the above. save data to `.tsv`, `.csv`, etc.
def export_df(data, sample, index, extension=".csv"):
    filepath = os.path.join(f"output/{date.today()}/", f"report.{sample}{index}{extension}")
    data.to_csv(filepath, index=False, header=False)


# builds one or both
def build_main(choice, topic, repeats, sample_size, verbose):
    """
    choice: choose which sample types to build. "random", "boosted", or "all"
    topic: topic for manual boosting
    """
    train = get_train()

    build_random(train, sample_size, repeats) if choice is "random" or "all" else None
    build_boosted(train, topic, sample_size, repeats) if choice is "boosted" or "all" else None
    print(f"Datasets built.") if verbose else None


""" MAIN """
# configuration
topic = ["trump"]  # [str]
to_build = "all"  # "all", "random", or "boosted"

# manually run:
# build_main(to_build, topic)
