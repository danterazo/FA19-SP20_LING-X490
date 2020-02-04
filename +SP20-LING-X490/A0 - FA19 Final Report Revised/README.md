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
    - Didn't have time but did research! (see below)
    
## 2/11/2020
- [] hard-code n_grams to `n=3` for this project
- [] GridSearchCV for parameters, don't worry about Pipeline for now (or n_grams)
    - kernels
    - gamma
- [] Get results! Train on **NLP** server
- Research paper
    - [] email papers to Sandra
    - Two boosted datasets: get original papers, read them, figure out what they mean by "boosting"
    - Boosting method

# TODO, future
- [] GridSearchCV parameter optimization
    - Use [Pipeline](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#building-a-pipeline)
    - [] Replace `ngram_range` list iteration functionality with GS-CV
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