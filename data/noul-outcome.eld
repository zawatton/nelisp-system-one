;; -*- lisp-data -*-
;;
;; P1's Noul task: "Does this sentence report a successful outcome?"
;;
;; Built as MINIMAL PAIRS.  Every entry has a partner with almost the same
;; words and the opposite label, which matters for three reasons:
;;
;;   1. The set is exactly balanced, so the majority-class baseline is 0.500
;;      with no rounding to argue about.
;;   2. Sentence length and unigram frequency are matched across labels by
;;      construction, so a probe cannot win on "negative sentences are longer"
;;      or any similar artefact of how the set was written.
;;   3. A pair must not be split across train and held-out -- the partner is a
;;      near-duplicate, and seeing one member during training would make the
;;      other a memorisation test rather than a generalisation one.  The
;;      splitter keys on :pair for exactly this reason.
;;
;; Two difficulties, marked by :hard.
;;
;;   :hard nil -- an antonym swap (passed/failed, rose/fell).  Lexically
;;      separable: a unigram baseline should do well here, and that is why the
;;      baseline is reported rather than the encoder being given the credit.
;;
;;   :hard t -- the polarity words are the SAME on both sides and the
;;      composition flips the meaning: "failed to lose" against "failed to
;;      win", "never missed" against "never met", "nothing went wrong" against
;;      "nothing went right".  A bag of words cannot separate these.  If the
;;      pooled encoder carries sentence meaning rather than word identity,
;;      this is where it shows, so accuracy is reported for the two subsets
;;      separately.
;;
;; Written for this repository rather than drawn from a corpus, which is a
;; limitation and is stated as one: the labels are the author's, the sentences
;; are short and clean, and none of that resembles the messy program state a
;; System One model would see in use.  It is enough to answer P1's question --
;; does a frozen 0.6B causal donor pool into something a linear head can use --
;; and not enough to answer anything about deployment.

(:question "Does this sentence report a successful outcome?"
 :template "%s Verdict:"
 :examples
 [
  ;; --- antonym pairs -----------------------------------------------------
  (:pair  1 :label 1 :hard nil :text "The repair finished ahead of schedule.")
  (:pair  1 :label 0 :hard nil :text "The repair finished behind schedule.")
  (:pair  2 :label 1 :hard nil :text "She managed to fix the leak.")
  (:pair  2 :label 0 :hard nil :text "She failed to fix the leak.")
  (:pair  3 :label 1 :hard nil :text "The backup restored every file.")
  (:pair  3 :label 0 :hard nil :text "The backup restored no files.")
  (:pair  4 :label 1 :hard nil :text "Nobody complained about the result.")
  (:pair  4 :label 0 :hard nil :text "Everybody complained about the result.")
  (:pair  5 :label 1 :hard nil :text "The test passed on the first run.")
  (:pair  5 :label 0 :hard nil :text "The test failed on the first run.")
  (:pair  6 :label 1 :hard nil :text "They avoided the outage entirely.")
  (:pair  6 :label 0 :hard nil :text "They caused the outage entirely.")
  (:pair  7 :label 1 :hard nil :text "The delivery arrived undamaged.")
  (:pair  7 :label 0 :hard nil :text "The delivery arrived damaged.")
  (:pair  8 :label 1 :hard nil :text "It was not a disappointment.")
  (:pair  8 :label 0 :hard nil :text "It was a disappointment.")
  (:pair  9 :label 1 :hard nil :text "The meeting ended with an agreement.")
  (:pair  9 :label 0 :hard nil :text "The meeting ended without an agreement.")
  (:pair 10 :label 1 :hard nil :text "The patch stopped the crashes.")
  (:pair 10 :label 0 :hard nil :text "The patch started the crashes.")
  (:pair 11 :label 1 :hard nil :text "Sales rose for the third quarter.")
  (:pair 11 :label 0 :hard nil :text "Sales fell for the third quarter.")
  (:pair 12 :label 1 :hard nil :text "The audit found nothing wrong.")
  (:pair 12 :label 0 :hard nil :text "The audit found something wrong.")
  (:pair 13 :label 1 :hard nil :text "We finished under budget.")
  (:pair 13 :label 0 :hard nil :text "We finished over budget.")
  (:pair 14 :label 1 :hard nil :text "The bridge held during the storm.")
  (:pair 14 :label 0 :hard nil :text "The bridge collapsed during the storm.")
  (:pair 15 :label 1 :hard nil :text "Her proposal was accepted.")
  (:pair 15 :label 0 :hard nil :text "Her proposal was rejected.")
  (:pair 16 :label 1 :hard nil :text "The engine restarted immediately.")
  (:pair 16 :label 0 :hard nil :text "The engine stalled immediately.")
  (:pair 17 :label 1 :hard nil :text "No one was injured.")
  (:pair 17 :label 0 :hard nil :text "Several were injured.")
  (:pair 18 :label 1 :hard nil :text "The vaccine prevented the outbreak.")
  (:pair 18 :label 0 :hard nil :text "The vaccine worsened the outbreak.")
  (:pair 19 :label 1 :hard nil :text "He remembered to lock the door.")
  (:pair 19 :label 0 :hard nil :text "He forgot to lock the door.")
  (:pair 20 :label 1 :hard nil :text "The team beat the deadline.")
  (:pair 20 :label 0 :hard nil :text "The team missed the deadline.")
  (:pair 21 :label 1 :hard nil :text "Nothing broke during transport.")
  (:pair 21 :label 0 :hard nil :text "Everything broke during transport.")
  (:pair 22 :label 1 :hard nil :text "The recipe turned out well.")
  (:pair 22 :label 0 :hard nil :text "The recipe turned out badly.")
  (:pair 23 :label 1 :hard nil :text "The alarm worked as intended.")
  (:pair 23 :label 0 :hard nil :text "The alarm did not work as intended.")
  (:pair 24 :label 1 :hard nil :text "Costs came down this year.")
  (:pair 24 :label 0 :hard nil :text "Costs went up this year.")
  (:pair 25 :label 1 :hard nil :text "The wound healed cleanly.")
  (:pair 25 :label 0 :hard nil :text "The wound became infected.")
  (:pair 26 :label 1 :hard nil :text "She passed the certification exam.")
  (:pair 26 :label 0 :hard nil :text "She failed the certification exam.")
  (:pair 27 :label 1 :hard nil :text "The roof survived the winter.")
  (:pair 27 :label 0 :hard nil :text "The roof leaked all winter.")
  (:pair 28 :label 1 :hard nil :text "We recovered the lost data.")
  (:pair 28 :label 0 :hard nil :text "We lost the recovered data.")
  (:pair 29 :label 1 :hard nil :text "The translation was accurate.")
  (:pair 29 :label 0 :hard nil :text "The translation was inaccurate.")
  (:pair 30 :label 1 :hard nil :text "He stopped smoking last year.")
  (:pair 30 :label 0 :hard nil :text "He started smoking last year.")
  (:pair 31 :label 1 :hard nil :text "The flight landed on time.")
  (:pair 31 :label 0 :hard nil :text "The flight landed three hours late.")
  (:pair 32 :label 1 :hard nil :text "The negotiation succeeded.")
  (:pair 32 :label 0 :hard nil :text "The negotiation collapsed.")
  (:pair 33 :label 1 :hard nil :text "Quality improved after the change.")
  (:pair 33 :label 0 :hard nil :text "Quality worsened after the change.")
  (:pair 34 :label 1 :hard nil :text "The battery lasted all day.")
  (:pair 34 :label 0 :hard nil :text "The battery died by noon.")
  (:pair 35 :label 1 :hard nil :text "They found the missing parcel.")
  (:pair 35 :label 0 :hard nil :text "They never found the parcel.")
  (:pair 36 :label 1 :hard nil :text "The dog recovered fully.")
  (:pair 36 :label 0 :hard nil :text "The dog never recovered.")
  (:pair 37 :label 1 :hard nil :text "Nothing was stolen.")
  (:pair 37 :label 0 :hard nil :text "Everything was stolen.")
  (:pair 38 :label 1 :hard nil :text "The seal held under pressure.")
  (:pair 38 :label 0 :hard nil :text "The seal burst under pressure.")
  (:pair 39 :label 1 :hard nil :text "The student answered correctly.")
  (:pair 39 :label 0 :hard nil :text "The student answered incorrectly.")
  (:pair 40 :label 1 :hard nil :text "The contract was signed today.")
  (:pair 40 :label 0 :hard nil :text "The contract fell through today.")
  (:pair 41 :label 1 :hard nil :text "The treatment relieved the pain.")
  (:pair 41 :label 0 :hard nil :text "The treatment increased the pain.")
  (:pair 42 :label 1 :hard nil :text "The company avoided bankruptcy.")
  (:pair 42 :label 0 :hard nil :text "The company declared bankruptcy.")
  (:pair 43 :label 1 :hard nil :text "He kept his promise.")
  (:pair 43 :label 0 :hard nil :text "He broke his promise.")
  (:pair 44 :label 1 :hard nil :text "The software installed without errors.")
  (:pair 44 :label 0 :hard nil :text "The software installed with errors.")
  (:pair 45 :label 1 :hard nil :text "Attendance exceeded expectations.")
  (:pair 45 :label 0 :hard nil :text "Attendance fell short of expectations.")
  (:pair 46 :label 1 :hard nil :text "The rescue reached them in time.")
  (:pair 46 :label 0 :hard nil :text "The rescue arrived too late.")
  (:pair 47 :label 1 :hard nil :text "We were refunded in full.")
  (:pair 47 :label 0 :hard nil :text "We were refunded nothing.")
  (:pair 48 :label 1 :hard nil :text "The pipeline ran to completion.")
  (:pair 48 :label 0 :hard nil :text "The pipeline aborted halfway.")
  (:pair 49 :label 1 :hard nil :text "The forecast was right.")
  (:pair 49 :label 0 :hard nil :text "The forecast was wrong.")
  (:pair 50 :label 1 :hard nil :text "Nobody had to work overtime.")
  (:pair 50 :label 0 :hard nil :text "Everybody had to work overtime.")
  (:pair 51 :label 1 :hard nil :text "The plants survived the frost.")
  (:pair 51 :label 0 :hard nil :text "The plants died in the frost.")
  (:pair 52 :label 1 :hard nil :text "The applicant was hired.")
  (:pair 52 :label 0 :hard nil :text "The applicant was turned down.")
  (:pair 53 :label 1 :hard nil :text "The problem disappeared after the reboot.")
  (:pair 53 :label 0 :hard nil :text "The problem persisted after the reboot.")
  (:pair 54 :label 1 :hard nil :text "The ship reached port safely.")
  (:pair 54 :label 0 :hard nil :text "The ship ran aground.")

  ;; --- compositional pairs (same polarity words on both sides) -----------
  (:pair 55 :label 1 :hard t :text "He failed to lose the contract.")
  (:pair 55 :label 0 :hard t :text "He failed to win the contract.")
  (:pair 56 :label 1 :hard t :text "We lost nothing.")
  (:pair 56 :label 0 :hard t :text "We gained nothing.")
  (:pair 57 :label 1 :hard t :text "She never missed a deadline.")
  (:pair 57 :label 0 :hard t :text "She never met a deadline.")
  (:pair 58 :label 1 :hard t :text "Nothing went wrong.")
  (:pair 58 :label 0 :hard t :text "Nothing went right.")
  (:pair 59 :label 1 :hard t :text "They stopped the damage.")
  (:pair 59 :label 0 :hard t :text "They stopped the repairs.")
  (:pair 60 :label 1 :hard t :text "The delay was avoided.")
  (:pair 60 :label 0 :hard t :text "The repair was avoided.")
  (:pair 61 :label 1 :hard t :text "He was not blamed.")
  (:pair 61 :label 0 :hard t :text "He was not praised.")
  (:pair 62 :label 1 :hard t :text "No one doubted the result.")
  (:pair 62 :label 0 :hard t :text "No one trusted the result.")
  (:pair 63 :label 1 :hard t :text "The fault was found and fixed.")
  (:pair 63 :label 0 :hard t :text "The fix was found to be faulty.")
  (:pair 64 :label 1 :hard t :text "They refused to give up.")
  (:pair 64 :label 0 :hard t :text "They refused to help.")
  (:pair 65 :label 1 :hard t :text "It could not have gone better.")
  (:pair 65 :label 0 :hard t :text "It could not have gone worse.")
  (:pair 66 :label 1 :hard t :text "The warning turned out to be false.")
  (:pair 66 :label 0 :hard t :text "The warning turned out to be true.")
  (:pair 67 :label 1 :hard t :text "Few parts failed.")
  (:pair 67 :label 0 :hard t :text "Few parts worked.")
  (:pair 68 :label 1 :hard t :text "The leak stopped by itself.")
  (:pair 68 :label 0 :hard t :text "The pump stopped by itself.")
  (:pair 69 :label 1 :hard t :text "Nobody was left behind.")
  (:pair 69 :label 0 :hard t :text "Nobody was picked up.")
  (:pair 70 :label 1 :hard t :text "The inspection missed nothing.")
  (:pair 70 :label 0 :hard t :text "The inspection missed everything.")
  ])
