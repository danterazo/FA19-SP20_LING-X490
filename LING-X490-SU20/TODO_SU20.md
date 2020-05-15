# TODO, after SPR break III
- [] URGENT: finish hate speech list
    - look over manual tags again
    - 4 classes: not hate, mildly abusive, very abusive, ?
      - 0, 1, 2, n
    - abusive = you say if you want to hurt someone
- [] Figure out differences between R-imported and Python data
    - Unix `diff`. Save train from R, use `diff`, then compare two
- [] Ask Sandra to change delimiter (?)
- [] count how often people agree
    - consider collapsing mildly + very abusive
    - look at ones with low agreement
    - consider agreement rate, that determines how far we go
    - might have to do one last round
    - Wiegand was automatic, hence uncommon false flags
- [] add other TODO items from unsorted list


# 4-23 TODO
- email whenever you upload stuff to Box
- UPLOAD hatespeech lexicon
- [] compare datasets: 3-4 abusive, take
    - 2 abusive, remove and have people decide later
- New lexicon files: `lex.tox`
- up wordbank to get 10000 posts
- [x] write a script to get 10000 posts randomly from training set
    - run ML on 10000 random and 10000 filtered to compare
    - [x] save 10000 posts to file + share. don't get 10000 random each time (one fixed random dataset)
- [] share python scripts/files on Box, send email to both
- Let Sandra know if you're busy over the summer
    - Could postpone to FA20

# 4-30, 5-15 TODO
- `lex.general` most common words + appearances
- [] might be an issue with my code
- should be 176048 trump examples. if so, just use those
    - `grep -i -c "trump" train.target+comments.tsv`
    - 10K randomly sampled trump examples (?)
- [] workshop, july 23, see email for details
- [] compare manually-annotated lexicons (?)

## Questions/comments for 2020-05-15
- For Sandra: issue was I cut dataset down to sample size too early. so it was filtering on only the first 10000 comments
- Define "boosting" and modify boost_data() appropriately (right now, it just filters)
    - Q: "Is it just filtering or something more?"
    - Refer to Wiegand, Sec. 3 (pg. 2)