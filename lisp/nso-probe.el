;;; nso-probe.el --- splitting, baselines and scoring for the P1 probe -*- lexical-binding: t; -*-

;;; Commentary:

;; The parts of P1 that do not need the donor: loading the dataset, splitting
;; it without leaking, the baselines the encoder has to beat, and the scoring.
;;
;; Kept donor-free on purpose.  One encoder pass costs 19 seconds on this
;; hardware, so the whole 140-example set is about an hour and a half of GPU
;; time; discovering afterwards that the split leaked, or that the "hard"
;; subset was solvable by counting words, would mean spending that hour again.
;; Everything here runs in `make test' against the real dataset, with no
;; weights present.
;;
;; Two baselines, because "beats chance" is a weak claim:
;;
;;   majority -- predict the commoner label.  Exactly 0.500 on this set, since
;;      it is built from minimal pairs.
;;   unigram  -- a logistic probe on binary bag-of-words features, fitted the
;;      same way the encoder probe is.  This is the one that matters: on
;;      antonym pairs it should do well, and any credit the encoder takes on
;;      that subset has to be credit *over* it.

;;; Code:

(require 'nso-head)
(require 'nso-metrics)

;;; Dataset

(defun nso-probe-load (path)
  "Read the dataset at PATH.  Returns (:question Q :template T :examples LIST)."
  (with-temp-buffer
    (insert-file-contents path)
    (let* ((form (read (buffer-string)))
           (ex (append (plist-get form :examples) nil)))
      (list :question (plist-get form :question)
            :template (plist-get form :template)
            :examples ex))))

(defun nso-probe-prompt (template example)
  "Render EXAMPLE's text through TEMPLATE."
  (format template (plist-get example :text)))

;;; Splitting
;;
;; By :pair, never by example.  The two members of a pair differ by a word or
;; two, so putting one in train and the other in held-out would measure
;; memorisation and report it as generalisation -- and it would report a
;; flattering number, which is the kind of bug that does not look like one.

(defun nso-probe-split (examples &optional every)
  "Split EXAMPLES into train and held-out, keeping each :pair whole.
Every EVERY-th pair (default 3) goes to held-out."
  (let ((every (or every 3))
        (train nil) (test nil))
    (dolist (e examples)
      (if (= 0 (mod (plist-get e :pair) every))
          (push e test)
        (push e train)))
    (list :train (nreverse train) :test (nreverse test))))

(defun nso-probe-labels (examples)
  "Labels of EXAMPLES as floats."
  (mapcar (lambda (e) (float (plist-get e :label))) examples))

(defun nso-probe-majority-accuracy (examples)
  "Accuracy of always predicting the commoner label in EXAMPLES."
  (let ((n (length examples)) (ones 0))
    (dolist (e examples)
      (when (= 1 (plist-get e :label)) (setq ones (1+ ones))))
    (/ (float (max ones (- n ones))) n)))

;;; Unigram baseline

(defun nso-probe-words (text)
  "Lower-case alphabetic tokens of TEXT."
  (let ((start 0) (out nil))
    (while (string-match "[A-Za-z]+" text start)
      (push (downcase (match-string 0 text)) out)
      (setq start (match-end 0)))
    (nreverse out)))

(defun nso-probe-vocab (examples)
  "Sorted vocabulary of EXAMPLES.
Built from the training split only; a vocabulary that has seen the held-out
sentences is a leak, even though the labels never enter it."
  (let ((seen (make-hash-table :test 'equal)) (out nil))
    (dolist (e examples)
      (dolist (w (nso-probe-words (plist-get e :text)))
        (unless (gethash w seen) (puthash w t seen) (push w out))))
    (sort out #'string<)))

(defun nso-probe-unigram-features (examples vocab)
  "Binary bag-of-words vectors for EXAMPLES over VOCAB."
  (let ((index (make-hash-table :test 'equal))
        (i 0))
    (dolist (w vocab) (puthash w i index) (setq i (1+ i)))
    (mapcar (lambda (e)
              (let ((v (make-vector (length vocab) 0.0)))
                (dolist (w (nso-probe-words (plist-get e :text)))
                  (let ((j (gethash w index)))
                    (when j (aset v j 1.0))))
                v))
            examples)))

;;; Scoring

(defun nso-probe-wilson (k n &optional z)
  "Wilson score interval for K successes in N trials.  Returns (LO . HI)."
  (if (= n 0) (cons 0.0 1.0)
    (let* ((z (or z 1.959963984540054))
           (p (/ (float k) n))
           (zz (* z z))
           (den (+ 1.0 (/ zz n)))
           (centre (/ (+ p (/ zz (* 2.0 n))) den))
           (half (/ (* (/ z den)
                       (sqrt (+ (/ (* p (- 1.0 p)) n) (/ zz (* 4.0 n n)))))
                    1.0)))
      (cons (max 0.0 (- centre half)) (min 1.0 (+ centre half))))))

(defun nso-probe-samples (probs labels)
  "Build `nso-metrics' samples from P(yes) values PROBS and LABELS."
  (let ((out nil) (rest labels))
    (dolist (p probs)
      (push (nso-sample (list (- 1.0 p) p) (if (= 1.0 (car rest)) 1 0)) out)
      (setq rest (cdr rest)))
    (nreverse out)))

(defun nso-probe-score (probs labels &optional bins)
  "Score P(yes) values PROBS against LABELS.
Returns (:n :correct :accuracy :ci-lo :ci-hi :ece :bins :brier :nll), or
(:n 0 :empty t) when PROBS is empty.

An empty subset is a real thing to report -- a small run can hold out no hard
examples at all -- and it is not the same as a metric over nothing.  So the
emptiness is handled here and `nso-ece' keeps refusing an empty sample set,
rather than the guard being loosened to let this case through.  A metric that
returns a number for no data is how a report ends up with a 0.000 that reads
like a measurement."
  (if (null probs)
      (list :n 0 :empty t :correct 0 :accuracy 0.0 :ci-lo 0.0 :ci-hi 1.0
            :ece 0.0 :bins (or bins 5) :brier 0.0 :nll 0.0)
  (let* ((n (length probs))
         (correct 0)
         (rest labels))
    (dolist (p probs)
      (when (eq (>= p 0.5) (= 1.0 (car rest))) (setq correct (1+ correct)))
      (setq rest (cdr rest)))
    (let* ((samples (nso-probe-samples probs labels))
           (ece (nso-ece samples (or bins 5)))
           (ci (nso-probe-wilson correct n)))
      (list :n n :correct correct :accuracy (/ (float correct) n)
            :ci-lo (car ci) :ci-hi (cdr ci)
            :ece (plist-get ece :ece) :bins (plist-get ece :bins)
            :brier (nso-brier samples) :nll (nso-nll samples))))))

(defun nso-probe-subset (probs labels examples pred)
  "Return (PROBS LABELS) restricted to EXAMPLES satisfying PRED."
  (let ((ps nil) (ls nil) (rp probs) (rl labels))
    (dolist (e examples)
      (when (funcall pred e)
        (push (car rp) ps)
        (push (car rl) ls))
      (setq rp (cdr rp) rl (cdr rl)))
    (list (nreverse ps) (nreverse ls))))

(defun nso-probe-format-score (label s)
  "One line for score plist S under LABEL."
  (if (plist-get s :empty)
      (format "  %-22s (no examples in this subset)" label)
    (format "  %-22s acc %.3f [%.3f,%.3f] n=%-3d  ECE %.3f  NLL %.3f  Brier %.3f"
            label (plist-get s :accuracy) (plist-get s :ci-lo) (plist-get s :ci-hi)
            (plist-get s :n) (plist-get s :ece) (plist-get s :nll)
            (plist-get s :brier))))

;;; Out-of-fold logits, for fitting a temperature honestly
;;
;; A temperature has to be fitted on logits from a head that never saw the
;; example, and there are three candidate sets, two of which are wrong:
;;
;;   held-out  -- leaks.  It is the set the result is reported on.
;;   training  -- worse, and not obviously so.  The head has already separated
;;                its training split, so the temperature that minimises NLL
;;                there is a SHARPENING one, and applying it to held-out data
;;                makes the calibration worse rather than better.  Measured on
;;                noise features, where the honest answer is a flat
;;                distribution: T came out at 0.13 and held-out ECE went from
;;                0.169 to 0.451.
;;   out of fold -- what is below.  Every training example is scored by a head
;;                fitted without its pair, so the logits are unbiased, and all
;;                of the training split is still used.
;;
;; The middle one was this repository's first implementation, with a comment
;; explaining why it was not the held-out set.  Avoiding one error is not the
;; same as being right.

(defun nso-probe-fold-map (pairs folds)
  "Assign each distinct pair in PAIRS to one of FOLDS, round-robin.
Folding on the pair id modulo FOLDS would collapse here: the training split is
already the pairs NOT divisible by three, so a further modulo-three fold would
leave one fold empty."
  (let ((map (make-hash-table :test 'eql)) (i 0))
    (dolist (p pairs)
      (unless (gethash p map)
        (puthash p (mod i folds) map)
        (setq i (1+ i))))
    map))

(defun nso-probe-oof-logits (items ys pairs featurizer &optional folds steps lr l2)
  "Out-of-fold logits for ITEMS/YS, folded by PAIRS so no pair scores itself.

FEATURIZER is called with the fold's fit ITEMS and labels and must return a
function from an item to a feature vector, *refit for that fold*.  For a fixed
pooling it ignores its arguments and returns the pooling.  For a learned one --
the attention pool -- it has to retrain, and that is the whole reason this
argument exists: with a pool direction fitted on the entire training split, the
out-of-fold head is scored on features that already saw the fold, the logits
come back separated, and the temperature fitted on them sharpens again.

Measured on noise features, where the honest temperature is a flattening one:
with the pool refit per fold the `last' and `mean' variants reached T=20 and
16 and drove ECE to 0.011 and 0.073, while `attn' with a pool fitted once on
all of train came back with T=0.21 and ECE 0.346.  Moving a leak is not the
same as removing one.

Returns a list aligned with ITEMS."
  (let* ((npairs (let ((h (make-hash-table :test 'eql)))
                   (dolist (p pairs) (puthash p t h))
                   (hash-table-count h)))
         ;; More folds than pairs leaves a fold with nothing in it.  Clamping
         ;; is the right answer rather than signalling: the caller asked for
         ;; out-of-fold logits, and with three pairs the honest answer is
         ;; three-fold degenerating to leave-one-pair-out, not a failure.  Two
         ;; is the floor, because one fold is not out of anything.
         (folds (max 2 (min (or folds 3) npairs)))
         (n (length items))
         (map (nso-probe-fold-map pairs folds))
         (out (make-vector n 0.0))
         (f 0))
    (while (< f folds)
      (let ((fit nil) (fy nil) (hold nil) (hi nil) (i 0) (rp pairs) (ry ys))
        (dolist (x items)
          (if (= f (gethash (car rp) map))
              (progn (push x hold) (push i hi))
            (push x fit) (push (car ry) fy))
          (setq i (1+ i) rp (cdr rp) ry (cdr ry)))
        (setq fit (nreverse fit) fy (nreverse fy)
              hold (nreverse hold) hi (nreverse hi))
        (unless (and fit hold)
          (error "nso-probe-oof-logits: fold %d left a side empty" f))
        (let* ((feat (funcall featurizer fit fy))
               (fx (mapcar feat fit))
               (std (nso-standardizer fx))
               (head (nso-head-train
                      (mapcar (lambda (v) (nso-standardize std v)) fx)
                      fy (or steps 600) (or lr 0.5) (or l2 0.05)))
               (rt hi))
          (dolist (x hold)
            (aset out (car rt)
                  (nso-head-logit head (nso-standardize std (funcall feat x))))
            (setq rt (cdr rt)))))
      (setq f (1+ f)))
    (append out nil)))

;;; Fitting a probe on cached features

(defun nso-probe-fit-and-score (train-x train-y test-x test-y
                                        &optional steps lr l2 bins)
  "Standardise on TRAIN-X, fit a head, and score it on both splits.
Returns (:train S :test S :logits LIST :head H :std STD)."
  (let* ((std (nso-standardizer train-x))
         (trx (mapcar (lambda (x) (nso-standardize std x)) train-x))
         (tex (mapcar (lambda (x) (nso-standardize std x)) test-x))
         (head (nso-head-train trx train-y (or steps 600) (or lr 0.5) (or l2 0.05)))
         (tr-logits (mapcar (lambda (x) (nso-head-logit head x)) trx))
         (te-logits (mapcar (lambda (x) (nso-head-logit head x)) tex)))
    (list :train (nso-probe-score (mapcar #'nso-sigmoid tr-logits) train-y bins)
          :test (nso-probe-score (mapcar #'nso-sigmoid te-logits) test-y bins)
          :logits te-logits
          :train-logits tr-logits
          :head head :std std)))

(provide 'nso-probe)
;;; nso-probe.el ends here
