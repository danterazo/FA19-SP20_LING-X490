# Documentation
I tried to make this as automated and modular as possible so I could run it on the NLP server
with nohup. This requires hardcoded inputs. I iterate over lists/arrays of inputs so that the
multiple models can be trained consecutively. This means that I effectively "set it and
forget it." 

Remember!! It's okay to fail in research!

# Work
## For 2/4/2020
- [x] Establish baseline
- [x] Feed **just** the *X* (`comment_text`) to CountVectorizer
- [] Use only two columns in data: `comment_text` for *X* and `class` for *y*
- Split training data into `train` and `dev`
    - [] Split in `get_data()`
    - [] Make it easy to switch from `dev` to `train` when it's time to use it

# TODO, future
- [] Work on `dev` exclusively, find the best settings / parameters, then use `train` at the very end
- [] Remove the following from the dataset: `http://t.co/*` links, `:NEWLINE_TOKEN:`, quotes `""`
- [] Don't return accuracy in the end. Instead, use macro average / f (harmonic mean). See Ken's code
- Filtering for boosted sampling
    - Prerequisites
        - [] Understand what boosting is. Refer to paper: Kaggle, Kumar boost abusive language. Others use topic sampling.
        - [] Kaggle: boost again
        - [] Kaggle: remove non-English language tweets (if any)
    - Filtering implementation
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
    - S2
        - [] Sampling theory
            - [] Boosted sampling method
            - [] Random sampling method
        - [] Condition corresponding to sampling used in other datasets. i.e. find hashtags in data, sample using that
        - [] Describe 7 columns of numerical data