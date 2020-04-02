# Documentation
I tried to make this as automated and modular as possible so I could run it on the NLP server
with `nohup`. This requires hardcoded inputs. I iterate over lists/arrays of inputs so that the
multiple models can be trained consecutively. This means that I effectively "set it and
forget it." With the size of the Kaggle dataset it's important to check `nohup` output for progress often!



**REMEMBER: It's okay to fail in research!**

## Legend
- `[ ]`: TODO, not yet completed
- `[/]`: In-progress
- `[x]`: Completed

# Work
## For 2/4/2020
- [x] Establish baseline
- [x] Feed **just** the *X* (`comment_text`) to CountVectorizer
- [x] Use only two columns for fits: `comment_text` for *X* and `class` for *y*
- [/] Split training data into `train` and `dev`
    - [x] Baseline:
        - [x] Split in `get_data()`. 80% `train`, 20% `dev`
        - [x] Make it easy to switch from `dev` to `train` when it's time to use the latter
    - [] If you have time:
        - [] Implement 5-fold cross validation (time-consuming!)
- [/] Get results! Train on **NLP** server
- [] Extra time? Do parameter optimization with GridSearchCV
    
## 2/11/2020
- [x] hard-code n_grams to `n=3` for this project
- [x] GridSearchCV for parameters, don't worry about Pipeline for now (or n_grams)
    - kernels
    - gamma
- []/ Get results! Train on **NLP** server
- Research paper
    - [x] email papers to Sandra
    - Two boosted datasets: get original papers, read them, figure out what they mean by "boosting"
    - Boosting method
    
## For ~~2/18/2020~~ 2/25/2020
- [] Create biased dataset from Kaggle by filtering on topics that cause abuse
    - [] First, find abusive topics. Filter 
    - [] Extract abusive, scan them, and find non-abusive hashtags (pick 3) specific to those topics
    - Then, sample on all tweets that contain that hashtag (save as "biased" dataset)
        - [] find a way to save dataset so you don't have to continually rebuild
        - [] print kaggle["comment_text"] to file too
        - [] file: "tweet1 \t label1" format to make access easier
    - Finally, randomly sample on Kaggle to get a dataset the same size ("boosted" b/c it's originally boosted)

## For 3/3/2020
- Not Tweets! Wikipedia talk pages!
- Undergrad conference: can report WIP papers
- [] look at abusive examples (save as CSV for ease), then describe them with a broad topic
- [] Save output of countvectorizer to .csv
    - If CV returns compressed format, then collect tweets with collected labels and save _that_ as a .csv
        - [] Save comment_text and class in separate .csv
- [] Shoot for ~15000 results
    - [x] Try "#metoo" / other topics for more results
    - [x] Combine topics if necessary!
- [x] make sure that randomly-sampled part works (i.e. shuffle, then pick first `n`)

# For 3/10/2020
- [x] Slack: @Sandra
    - Email Sandra
- [x] Uncomment `hate_data` step, save as CSV, save to Box and ping Sandra
    - [] Create frequency dictionary. Look for words that look like topics, but not abuse. MANUAL step
- Figure out which Wikipedia pages this dataset is from, then get topic(s) from that
    - Otherwise, will have to rethink problem
    - PING Sandra and have her look at Kaggle dataset and look for terms to use
- [] Get ~15000 results; Waseem dataset size

# For after SPR BREAK
- [] #1 goal: get terms to 15000
- [] Split Hate Speech List into three:
    1. Hate
    2. Not Hate
    3. Unsure
    - Don't add to wordbank!! Using afterwards to determine implicit/explicit abusive language
- [] Come up with unoffensive terms that will find offensive tweets
    - e.g., Islam. It's not offensive, but when people talk about it it tends to be offensive.
    - [] Sample from Kaggle. Very manual
    - Good distribution of offensive and non-offensive after filtering
- [] start with unigrams as features from `train.csv.`
- [] Two experiments to determine which words are implicit / explicit offensive
    - [] Their experiment (Hate Speech List.csv), where words like vomit are offensive
        - Paper hypothesis: % of explicit offense makes it easier for classifier
        - e.g. "Women belong in the kitchen"
    - [] Their experiment but we remove unoffensive words (e.g. vomit)
    
# TODO, after SPR break
- [] baseline. create train/test
    - 80/20 or 90/10
    - shuffle before
    - train on 80, test on 20. file away, then go back to testing on small samples.

# TODO, future
- [] FUTURE: potentially sample only implicit tweets
- [] python regex module: compiles to Finite-state automaton to search
    - CONSIDER if searching/boosting takes too long
- [] CSV stuff from last time and the time before
- [] GridSearchCV parameter optimization
    - Use [Pipeline](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#building-a-pipeline)
    - [] Replace `ngram_range` list iteration functionality with GS-CV
- [] 5-fold Cross Validation
- [] Clean up code, make it easy to follow and read
    - [] Consider keeping only common code in `get_data()`
        - [] Fix `boost_data()`, rename to `get_boosted_data()`
        - [] Create `get_random_data()`
        - [] Call both from `get_data()`. This should clean up the code considerably
- [] Work on `dev` exclusively, find the best settings / parameters, then use `train` at the very end
- [] Remove the following from the dataset: `http://t.co/*` links, `:NEWLINE_TOKEN:`, quotes `""`
- [] Don't return accuracy in the end. Instead, use macro average / f (harmonic mean). See Ken's code
- [] Filtering for boosted sampling
    - [] Prerequisites
        - [] Understand what boosting is. Refer to paper: Kaggle, Kumar boost abusive language. Others use topic sampling.
        - [] Kaggle: boost again
        - [] Kaggle: remove non-English language tweets (if any)
    - [] Filtering implementation
        - [] Filter on events. Sift through to identify some.
            - For example, using **Islam**, manually create a list of Islam-related hashtags (they don't have to be
            abusive). Focus on non-abusive hashtags if possible.
            - See how large you can make the set. Trim down to the same size as the randomly-sampled set if possible.
                - Idea: get boosted first, then trim random-sample set instead of vice versa?
            - Can an Islam-trained model detect abusive language in tweets related to a cooking show?
- Paper discoveries
        - One method might work better than the other due to explicit / overt abusive language.
            - Easy for machine learner to detect explicit examples
            - Implicit is hard to detect using the bag-of-words method (e.g. Hillary White House example)
            - The more explicit abusive, the easier the task
        - Wiegand's paper features multiple sampling types on different datasets. We want to try those sampling types
        on one dataset (Kaggle)
- Paper updates
    - [] S2
        - [] Sampling theory
            - [] Boosted sampling method
            - [] Random sampling method
        - [] Condition corresponding to sampling used in other datasets. i.e. find hashtags in data, sample using that
        - [] Describe 7 columns of numerical data
    - [] Research question: want to look at what charactistics of the dataset net better results (might not be as simple 
    as boosted vs. random sampling)
    - [] Look up how Kaggle dataset was created
    - [] 