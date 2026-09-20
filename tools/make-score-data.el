;;; make-score-data.el --- build the Score dataset, and refuse to build a bad one -*- lexical-binding: t; -*-

;; The Noul and Choice sets were written by hand and then audited by script.
;; This one is generated, and that is a deliberate difference rather than
;; laziness: an ordinal label is an assertion about ORDER, and an order the
;; author assigned sentence by sentence is an order the author can drift on.
;; Here the level is a property of the template, so every claim the data makes
;; is auditable by reading sixteen scenarios and ten templates instead of two
;; hundred and forty sentences.
;;
;; The cost is stated rather than hidden: the sentences are regular in a way
;; real program state is not, so a result here bounds what the encoder CAN do
;; and says nothing about what it would do on messy input.
;;
;; The generator refuses to emit a file that fails its own construction
;; checks -- level balance, no duplicate sentences, and the negation-signature
;; check described below, which is the one that decides whether the hard
;; subset is worth measuring.
;;
;; Run:  emacs -Q --batch -l tools/make-score-data.el

(defvar nso-mk--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(defvar nso-mk--out (expand-file-name "../data/score-commitment.eld" nso-mk--here))

(defvar nso-mk--levels
  ["ruled out" "unlikely" "uncertain" "likely" "certain"])

;; Each scenario supplies the pieces the templates need.  BE is carried
;; because "the results is unlikely" would be an artefact of the generator
;; showing up as a property of the data.
(defvar nso-mk--scenarios
  '((:subj "the shipment" :be "is" :vp "arrive today")
    (:subj "the meeting" :be "is" :vp "start on time")
    (:subj "the payment" :be "is" :vp "clear this week")
    (:subj "the report" :be "is" :vp "be ready by Friday")
    (:subj "the engine" :be "is" :vp "start in the cold")
    (:subj "the flight" :be "is" :vp "depart on schedule")
    (:subj "the contract" :be "is" :vp "be signed this month")
    (:subj "the team" :be "is" :vp "finish the migration")
    (:subj "the power" :be "is" :vp "come back before dark")
    (:subj "the parcel" :be "is" :vp "fit through the letterbox")
    (:subj "the shop" :be "is" :vp "open on Sunday")
    (:subj "the alarm" :be "is" :vp "sound during the test")
    (:subj "the bridge" :be "is" :vp "reopen in spring")
    (:subj "the deposit" :be "is" :vp "be refunded in full")
    (:subj "the roof" :be "is" :vp "leak again")
    (:subj "the results" :be "are" :vp "come back on Monday")))

(defun nso-mk--cap (s)
  (concat (upcase (substring s 0 1)) (substring s 1)))

(defun nso-mk--forms (sc)
  "The surface forms SC's templates are built from."
  (let ((subj (plist-get sc :subj))
        (be (plist-get sc :be))
        (vp (plist-get sc :vp)))
    (list :will (format "%s will %s" subj vp)
          :will-not (format "%s will not %s" subj vp)
          :likely (format "%s %s likely to %s" subj be vp)
          :unlikely (format "%s %s unlikely to %s" subj be vp)
          :maybe (format "%s may or may not %s" subj vp)
          :certain (format "%s will certainly %s" subj vp))))


;; The two halves of the set, and the whole argument for building it this way.
;;
;; EASY -- the level is carried by one surface word: "will not", "unlikely",
;; "may or may not", "likely", "certainly".  A bag of words should do well
;; here, and it is reported so that the encoder is not credited with what a
;; lookup does.
;;
;; HARD -- built so that a bag of words CANNOT do well, and provably so rather
;; than plausibly.  Every hard sentence has a partner in the same scenario
;; whose BINARY BAG OF WORDS IS IDENTICAL and whose level is different.  The
;; trick is a matrix clause that already contains "not", so flipping the
;; embedded clause's polarity adds a second "not" that a set-valued bag cannot
;; see:
;;
;;   "I do not doubt that E will arrive"        certain      |  same
;;   "I do not doubt that E will not arrive"    ruled out    |  bag
;;
;;   "It is not true that E will arrive"        ruled out    |  same
;;   "It is not true that E will not arrive"    certain      |  bag
;;
;;   "It is not unlikely that E will arrive"        likely   |  same
;;   "It is not unlikely that E will not arrive"    unlikely |  bag
;;
;; and so on.  Because the two members are indistinguishable to any model
;; whose input is the set of words, such a model must answer both the same
;; way, and therefore gets at most one of each unequal pair right.  That is a
;; bound on ACCURACY and MAE that holds for a bag model of any capacity -- not
;; a claim about linear models, and not a claim that has to be re-argued when
;; someone fits something bigger.  `nso-mk--check-collisions' computes the
;; bound from the emitted file and refuses to write one whose hard half is not
;; fully paired.
;;
;; This replaced a first version whose hard half swapped "no doubt" against
;; "no chance" and asserted that a bag of words could not tell them apart.
;; The check written for it looked only at which NEGATION words were present,
;; ignoring that "doubt" and "chance" are themselves features, and a five-way
;; softmax -- one weight vector per level rather than one shared score --
;; scored 0.800 on that subset.  The check fired on what it measured; it was
;; measuring the wrong thing.  `tools/score-lexical-check.el' found it before
;; any GPU time was spent, which is what it is for.
;;
;; One pair is deliberately level-EQUAL ("It is not certain that E will
;; arrive" and "... will not arrive" are both uncertain), so a bag model gets
;; that one right and the accuracy ceiling on the hard subset is 0.60 rather
;; than 0.50.  The subset is a bound, not a trap.

(defun nso-mk--rows (sc idx)
  "The fifteen examples for scenario SC, numbered IDX."
  (let* ((f (nso-mk--forms sc))
         (g (lambda (k) (plist-get f k)))
         (row (lambda (level hard family text)
                (list :scenario idx :level level :hard hard
                      :family family :text text))))
    (list
     ;; --- easy: one modal word carries the level ---------------------------
     (funcall row 0 nil 'easy (concat (nso-mk--cap (funcall g :will-not)) "."))
     (funcall row 1 nil 'easy (concat (nso-mk--cap (funcall g :unlikely)) "."))
     (funcall row 2 nil 'easy (concat (nso-mk--cap (funcall g :maybe)) "."))
     (funcall row 3 nil 'easy (concat (nso-mk--cap (funcall g :likely)) "."))
     (funcall row 4 nil 'easy (concat (nso-mk--cap (funcall g :certain)) "."))
     ;; --- hard: five bag-identical pairs, one per level pair ---------------
     (funcall row 4 t 'doubt
              (format "I do not doubt that %s." (funcall g :will)))
     (funcall row 0 t 'doubt
              (format "I do not doubt that %s." (funcall g :will-not)))
     (funcall row 0 t 'true
              (format "It is not true that %s." (funcall g :will)))
     (funcall row 4 t 'true
              (format "It is not true that %s." (funcall g :will-not)))
     (funcall row 3 t 'litotes
              (format "It is not unlikely that %s." (funcall g :will)))
     (funcall row 1 t 'litotes
              (format "It is not unlikely that %s." (funcall g :will-not)))
     (funcall row 3 t 'expect
              (format "It is not unreasonable to expect that %s." (funcall g :will)))
     (funcall row 1 t 'expect
              (format "It is not unreasonable to expect that %s." (funcall g :will-not)))
     (funcall row 2 t 'certain
              (format "It is not certain that %s." (funcall g :will)))
     (funcall row 2 t 'certain
              (format "It is not certain that %s." (funcall g :will-not))))))

;;; Construction checks

(defun nso-mk--bag (text)
  "The set of lower-case word types in TEXT, sorted.
A SET, not a multiset, because that is what `nso-probe-unigram-features'
builds: it writes 1.0 into a slot and never counts.  The pairs below exploit
exactly that, so the check has to model the feature map the baseline actually
uses rather than the one the phrase \"bag of words\" suggests."
  (let ((seen (make-hash-table :test 'equal)) (out nil))
    (dolist (w (split-string (downcase text) "[^a-z]+" t))
      (unless (gethash w seen) (puthash w t seen) (push w out)))
    (sort out #'string<)))

(defun nso-mk--check-collisions (rows)
  "Signal unless every hard row has a bag-identical partner in its scenario.
Returns (:pairs N :accuracy-ceiling A :mae-floor M) for the hard subset: the
best any model whose input is the set of words can do on it."
  (let ((groups (make-hash-table :test 'equal))
        (pairs 0) (correct-ceiling 0.0) (mae-floor 0.0) (n 0))
    (dolist (r rows)
      (when (plist-get r :hard)
        (let ((key (list (plist-get r :scenario) (nso-mk--bag (plist-get r :text)))))
          (puthash key (cons r (gethash key groups)) groups))))
    (maphash
     (lambda (key members)
       (unless (= (length members) 2)
         (error (concat "make-score-data: %d hard sentences share bag %S in "
                        "scenario %S.  The subset is built from PAIRS; a "
                        "singleton is a sentence a bag of words can answer "
                        "freely, and a triple makes the ceiling below wrong")
                (length members) (cadr key) (car key)))
       (let* ((a (plist-get (nth 0 members) :level))
              (b (plist-get (nth 1 members) :level)))
         (setq pairs (1+ pairs) n (+ n 2))
         ;; Indistinguishable inputs must get the same answer, so the pair
         ;; yields two correct only when the two levels agree, and otherwise
         ;; one.  The best single answer for the pair on MAE is anything
         ;; between them, which costs |a-b| over the two.
         (setq correct-ceiling (+ correct-ceiling (if (= a b) 2.0 1.0)))
         (setq mae-floor (+ mae-floor (abs (- a b))))))
     groups)
    (when (= pairs 0) (error "make-score-data: the hard subset is empty"))
    (list :pairs pairs
          :accuracy-ceiling (/ correct-ceiling n)
          :mae-floor (/ mae-floor n))))

(defun nso-mk--check-balance (rows k)
  "Signal unless every level appears equally often, overall and per scenario."
  (let ((overall (make-vector k 0))
        (per (make-hash-table :test 'equal)))
    (dolist (r rows)
      (aset overall (plist-get r :level) (1+ (aref overall (plist-get r :level))))
      (let* ((key (cons (plist-get r :scenario) (plist-get r :level)))
             (n (or (gethash key per) 0)))
        (puthash key (1+ n) per)))
    (let ((first (aref overall 0)))
      (dotimes (i k)
        (unless (= (aref overall i) first)
          (error (concat "make-score-data: level %d appears %d times against "
                         "level 0's %d.  The majority baseline would not be "
                         "1/K and every comparison against it would be wrong")
                 i (aref overall i) first))))
    (maphash (lambda (key n)
               (unless (= n 3)
                 (error "make-score-data: scenario %S has %d examples at level %S"
                        (car key) n (cdr key))))
             per)
    (aref overall 0)))

(defun nso-mk--check-unique (rows)
  "Signal on a repeated sentence, which would put the same text on both sides."
  (let ((seen (make-hash-table :test 'equal)))
    (dolist (r rows)
      (when (gethash (plist-get r :text) seen)
        (error "make-score-data: duplicate sentence %S" (plist-get r :text)))
      (puthash (plist-get r :text) t seen))
    (hash-table-count seen)))

;;; Emit

(let* ((rows nil) (idx 0))
  (dolist (sc nso-mk--scenarios)
    (setq idx (1+ idx))
    (setq rows (append rows (nso-mk--rows sc idx))))
  (let* ((k (length nso-mk--levels))
         (bound nil))
    (nso-mk--check-unique rows)
    (nso-mk--check-balance rows k)
    (setq bound (nso-mk--check-collisions rows))
    (with-temp-file nso-mk--out
      (insert ";; -*- lisp-data -*-\n;;\n")
      (insert ";; GENERATED by tools/make-score-data.el -- edit that, not this.\n;;\n")
      (insert ";; Score's task: \"how strongly does this sentence commit to the event\n")
      (insert ";; happening?\", on five ordered levels\n;;\n")
      (insert ";;   0 ruled out  <  1 unlikely  <  2 uncertain  <  3 likely  <  4 certain\n;;\n")
      (insert ";; Sixteen scenarios, fifteen sentences each, 240 examples, exactly 48 at\n")
      (insert ";; every level and exactly three per level per scenario.  So the\n")
      (insert ";; majority-class baseline is 0.200 with nothing to round, and the best\n")
      (insert ";; constant answer is the middle level at an MAE of 1.200.  Both numbers\n")
      (insert ";; are what the head has to beat to have used the sentence at all.\n;;\n")
      (insert ";; SPLIT BY :scenario, never by example.  Fifteen sentences share a subject\n")
      (insert ";; and a verb phrase; splitting inside one would put near-duplicates on both\n")
      (insert ";; sides and report memorisation as generalisation.\n;;\n")
      (insert ";; Two difficulties, marked by :hard.\n;;\n")
      (insert ";;   :hard nil -- one modal word carries the level (\"unlikely\", \"certainly\").\n")
      (insert ";;      Lexically separable, and the unigram baseline is reported on it so\n")
      (insert ";;      the encoder is not credited with what a lookup table does.\n;;\n")
      (insert ";;   :hard t -- every sentence has a partner in the same scenario with an\n")
      (insert ";;      IDENTICAL SET OF WORDS and, in four pairs out of five, a different\n")
      (insert ";;      level.  The matrix clause already carries \"not\", so flipping the\n")
      (insert ";;      embedded clause's polarity adds a second \"not\" that a set-valued\n")
      (insert ";;      bag cannot see:\n")
      (insert ";;\n")
      (insert ";;        \"I do not doubt that E will arrive\"       certain\n")
      (insert ";;        \"I do not doubt that E will not arrive\"   ruled out\n")
      (insert ";;\n")
      (insert ";;      Two inputs a model cannot distinguish must get the same answer, so\n")
      (insert (format ";;      ANY bag-of-words model scores at most %.3f accuracy and at least\n"
                      (plist-get bound :accuracy-ceiling)))
      (insert (format ";;      %.3f MAE on this subset, whatever its capacity.  That is a bound,\n"
                      (plist-get bound :mae-floor)))
      (insert ";;      not an expectation, and the generator recomputes it from the\n")
      (insert ";;      sentences rather than restating it from here.\n;;\n")
      (insert ";; Limitation, stated rather than left to be discovered: these sentences are\n")
      (insert ";; template-regular.  A result here bounds what a frozen donor CAN carry on\n")
      (insert ";; an ordinal scale; it says nothing about messy input, and the levels are\n")
      (insert ";; assigned by template rather than annotated, so they test whether the\n")
      (insert ";; encoder recovers a rule rather than whether it agrees with a human.\n\n")
      (let ((print-level nil) (print-length nil))
        (insert "(:question \"How strongly does this sentence commit to the event happening?\"\n")
        (insert " :template \"%s Likelihood:\"\n")
        (insert " :levels ")
        (prin1 nso-mk--levels (current-buffer))
        (insert "\n :examples\n [\n")
        (let ((last nil))
          (dolist (r rows)
            (when (and last (/= last (plist-get r :scenario))) (insert "\n"))
            (setq last (plist-get r :scenario))
            (insert "  ")
            (prin1 r (current-buffer))
            (insert "\n")))
        (insert " ])\n")))
    (message "wrote %d examples over %d scenarios to %s"
             (length rows) idx nso-mk--out)
    (message "hard subset: %d bag-identical pairs, bag ceiling acc %.3f, MAE floor %.3f"
             (plist-get bound :pairs)
             (plist-get bound :accuracy-ceiling)
             (plist-get bound :mae-floor))))

;;; make-score-data.el ends here
