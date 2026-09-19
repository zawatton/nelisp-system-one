;;; probe-test.el --- the dataset and the split, before any GPU time -*- lexical-binding: t; -*-

;;; Commentary:

;; One encoder pass costs 19 seconds, so the full set is about an hour and a
;; half of GPU time.  This suite is what runs first, and it exists to catch
;; the three ways that hour gets wasted:
;;
;;   - the split leaks, so held-out accuracy measures memorisation;
;;   - the set is not balanced, so "beats the majority baseline" means less
;;     than it appears to;
;;   - the "hard" subset is not hard, so the one measurement that would
;;     distinguish sentence meaning from word identity distinguishes nothing.
;;
;; The third is checked by running the unigram baseline itself, two-fold by
;; pair parity so every hard pair is scored out of sample.  The baseline must
;; do well on the antonym pairs -- otherwise it is broken and its later
;; failure would prove nothing -- and must sit near chance on the
;; compositional ones.  Both directions, as usual; a control that only ever
;; fails is as uninformative as one that only ever passes.

;;; Code:

(require 'nso-probe)
(require 'nso-stub)                     ; the seeded generator
(load (expand-file-name "nso-test-helper.el"
                        (file-name-directory (or load-file-name buffer-file-name))))

(message "== probe ==")

(defvar pt--data
  (nso-probe-load
   (expand-file-name "../data/noul-outcome.eld"
                     (file-name-directory (or load-file-name buffer-file-name)))))
(defvar pt--ex (plist-get pt--data :examples))

;;; --- shape ---------------------------------------------------------------

(nso-t "the dataset loads" (and pt--ex (listp pt--ex)))
(nso-t-num "140 examples" (float (length pt--ex)) 140.0 0.5)

(let ((by-pair (make-hash-table :test 'eql)))
  (dolist (e pt--ex)
    (puthash (plist-get e :pair)
             (cons (plist-get e :label) (gethash (plist-get e :pair) by-pair))
             by-pair))
  (let ((pairs 0) (bad 0))
    (maphash (lambda (_k v)
               (setq pairs (1+ pairs))
               (unless (and (= 2 (length v)) (member 0 v) (member 1 v))
                 (setq bad (1+ bad))))
             by-pair)
    (nso-t-num "70 pairs" (float pairs) 70.0 0.5)
    (nso-t "every pair is one yes and one no" (= 0 bad))))

(let ((seen (make-hash-table :test 'equal)) (dups 0))
  (dolist (e pt--ex)
    (if (gethash (plist-get e :text) seen)
        (setq dups (1+ dups))
      (puthash (plist-get e :text) t seen)))
  (nso-t "no sentence appears twice" (= 0 dups)))

(nso-t-num "the majority baseline is exactly one half"
           (nso-probe-majority-accuracy pt--ex) 0.5 1e-12)

;;; --- the split does not leak ---------------------------------------------

(let* ((sp (nso-probe-split pt--ex))
       (train (plist-get sp :train))
       (test (plist-get sp :test))
       (tr-pairs (make-hash-table :test 'eql))
       (shared 0))
  (dolist (e train) (puthash (plist-get e :pair) t tr-pairs))
  (dolist (e test) (when (gethash (plist-get e :pair) tr-pairs)
                     (setq shared (1+ shared))))
  (message "  split: %d train, %d held-out" (length train) (length test))
  (nso-t "no pair straddles the split" (= 0 shared))
  (nso-t "the split is non-trivial in both directions"
         (and (> (length train) 40) (> (length test) 20)))
  (nso-t-num "the majority baseline is one half on the training split"
             (nso-probe-majority-accuracy train) 0.5 1e-12)
  (nso-t-num "and on the held-out split"
             (nso-probe-majority-accuracy test) 0.5 1e-12)
  (let ((h-tr 0) (h-te 0) (e-tr 0) (e-te 0))
    (dolist (e train) (if (plist-get e :hard) (setq h-tr (1+ h-tr)) (setq e-tr (1+ e-tr))))
    (dolist (e test) (if (plist-get e :hard) (setq h-te (1+ h-te)) (setq e-te (1+ e-te))))
    (message "  hard: %d train / %d held-out;  antonym: %d / %d" h-tr h-te e-tr e-te)
    (nso-t "both difficulties reach both splits"
           (and (> h-tr 0) (> h-te 0) (> e-tr 0) (> e-te 0)))))

;;; --- Wilson interval -----------------------------------------------------

(let ((ci (nso-probe-wilson 30 40)))
  (nso-t "the interval brackets the point estimate"
         (and (< (car ci) 0.75) (> (cdr ci) 0.75)))
  (nso-t "and is not the whole unit interval at n=40"
         (> (car ci) 0.55)))
(let ((ci (nso-probe-wilson 5 10)))
  (nso-t "a small sample gives a wide interval"
         (and (< (car ci) 0.3) (> (cdr ci) 0.7))))

;;; --- empty subsets ------------------------------------------------------
;;
;; A small run can hold out no hard examples at all.  That is a fact to
;; report, not a metric to compute, so `nso-probe-score' answers it and
;; `nso-ece' goes on refusing -- a metric that returns a number for no data
;; puts a 0.000 in a table where it reads like a measurement.

(let ((s (nso-probe-score nil nil 5)))
  (nso-t "an empty subset scores as empty rather than signalling"
         (and (plist-get s :empty) (= 0 (plist-get s :n))))
  (nso-t "and formats as absent rather than as a number"
         (string-match-p "no examples" (nso-probe-format-score "hard" s))))

(nso-t-signals "but the ECE itself still refuses an empty sample set"
               (lambda () (nso-ece nil)))

(let ((s (nso-probe-score '(0.9 0.2) '(1.0 0.0) 5)))
  (nso-t "a non-empty subset is not marked empty" (null (plist-get s :empty)))
  (nso-t-num "and is scored normally" (plist-get s :accuracy) 1.0 1e-12))

;;; --- the unigram baseline, two-fold by pair parity -----------------------
;;
;; Every pair is held out exactly once, so each subset is scored on all of its
;; examples rather than on the handful the fixed split happens to hold back.
;;
;; The result was not what this suite was first written to assert, and the
;; correction is worth stating rather than quietly editing away.  The first
;; version demanded that the baseline separate the antonym pairs, on the
;; reasoning that "passed" against "failed" is a lexical difference.  It does
;; not, and it cannot: splitting by pair means the discriminating token of a
;; held-out pair -- "collapsed", "aground", "undamaged" -- was never seen with
;; a label during training.  A bag of words has nothing to carry across.
;;
;; So the two assertions below are: the baseline MEMORISES its training split,
;; which proves the features and the fitting work and that a chance result is
;; not a broken implementation; and it lands at chance out of fold on both
;; subsets, which is a property of the design rather than a defect.
;;
;; The consequence for P1 is that the unigram baseline is not the interesting
;; comparison here -- it is at chance by construction, so any encoder above
;; chance beats it and the win would mean little.  The majority baseline at
;; 0.500 is the one the acceptance criterion names, and the antonym-versus-
;; compositional split remains informative because both are semantic
;; questions for anything that generalises at all.

(let ((fold-probs (make-hash-table :test 'equal))
      (train-accs nil)
      (fold-of (lambda (e) (mod (plist-get e :pair) 2))))
  (dotimes (held 2)
    (let* ((train (let (out) (dolist (e pt--ex)
                               (unless (= held (funcall fold-of e)) (push e out)))
                       (nreverse out)))
           (test (let (out) (dolist (e pt--ex)
                              (when (= held (funcall fold-of e)) (push e out)))
                      (nreverse out)))
           (vocab (nso-probe-vocab train))
           (trx (nso-probe-unigram-features train vocab))
           (tex (nso-probe-unigram-features test vocab))
           (fit (nso-probe-fit-and-score trx (nso-probe-labels train)
                                         tex (nso-probe-labels test)
                                         400 0.5 0.05 5)))
      (push (plist-get (plist-get fit :train) :accuracy) train-accs)
      (message "  fold %d: vocab %d, train acc %.3f"
               held (length vocab)
               (plist-get (plist-get fit :train) :accuracy))
      (let ((rest (plist-get fit :logits)))
        (dolist (e test)
          (puthash (plist-get e :text) (nso-sigmoid (car rest)) fold-probs)
          (setq rest (cdr rest))))))
  (let* ((probs (mapcar (lambda (e) (gethash (plist-get e :text) fold-probs)) pt--ex))
         (labels (nso-probe-labels pt--ex))
         (easy (nso-probe-subset probs labels pt--ex
                                 (lambda (e) (not (plist-get e :hard)))))
         (hard (nso-probe-subset probs labels pt--ex
                                 (lambda (e) (plist-get e :hard))))
         (s-all (nso-probe-score probs labels 5))
         (s-easy (nso-probe-score (nth 0 easy) (nth 1 easy) 5))
         (s-hard (nso-probe-score (nth 0 hard) (nth 1 hard) 5))
         (worst-train (apply #'min train-accs)))
    (message "  unigram baseline, out of fold:")
    (message "%s" (nso-probe-format-score "all" s-all))
    (message "%s" (nso-probe-format-score "antonym pairs" s-easy))
    (message "%s" (nso-probe-format-score "compositional pairs" s-hard))
    (nso-t "every example got an out-of-fold prediction"
           (= 140 (length (delq nil (copy-sequence probs)))))
    ;; The baseline works: it fits what it is shown.  Without this row, the
    ;; chance results below would be indistinguishable from a broken probe.
    (nso-t-gt "the unigram baseline memorises its own training split"
              worst-train 0.90)
    ;; And cannot carry any of it across a pair-level split.
    (nso-t-lt "yet lands at chance out of fold on the antonym pairs"
              (plist-get s-easy :accuracy) 0.68)
    (nso-t-lt "and at chance on the compositional pairs"
              (plist-get s-hard :accuracy) 0.70)))

;;; --- the featurizer is refit per fold ------------------------------------
;;
;; The second defect the noise control found.  Moving the head out of fold
;; while leaving a LEARNED pooling fitted on the whole training split does not
;; remove the leak, it relocates it: the out-of-fold head is scored on features
;; that already saw the fold, the logits come back separated, and the
;; temperature sharpens again.  Measured on noise, with the pool fitted once:
;; T=0.21 and ECE 0.346, against 0.011 and 0.073 for the two fixed poolings.
;;
;; So the contract is that the featurizer is called once per fold, with that
;; fold's fit items only, and never with the whole set.  Pinned directly,
;; because the symptom is a plausible-looking number rather than an error.

(let* ((items '(a b c d e f g h i))
       (ys '(1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0))
       (pairs '(1 1 2 2 3 3 4 4 5))
       (calls nil)
       (leaked nil))
  (nso-probe-oof-logits
   items ys pairs
   (lambda (fit-items _fit-ys)
     (push fit-items calls)
     (lambda (_x) (vector 1.0 0.0)))
   3 20 0.5 0.05)
  (nso-t-num "the featurizer is called once per fold"
             (float (length calls)) 3.0 0.5)
  (nso-t "and never with the whole set"
         (progn (dolist (c calls)
                  (when (= (length c) (length items)) (setq leaked t)))
                (not leaked)))
  ;; Each fold's fit set must be disjoint from the items that fold scores.
  (let* ((map (nso-probe-fold-map pairs 3))
         (bad 0))
    (dolist (c calls)
      (let ((folds-present nil))
        (dolist (x c)
          (let ((i (- (length items) (length (memq x items)))))
            (push (gethash (nth i pairs) map) folds-present)))
        ;; a fit set spans exactly the two folds it is not holding out
        (unless (= 2 (length (delete-dups (copy-sequence folds-present))))
          (setq bad (1+ bad)))))
    (nso-t "each fit set spans exactly the folds it does not hold out"
           (= 0 bad))))

;;; --- where a temperature may be fitted -----------------------------------
;;
;; Pins the correction described in `nso-probe-oof-logits'.  The setup is a
;; head that overfits: 128 features, 40 training examples, a weak signal.  On
;; such a head the training split's own logits are already separated, so the
;; temperature that minimises NLL there is a sharpening one and it makes
;; held-out calibration worse.  Out-of-fold logits do not have that property.
;;
;; Without this check the wrong method is invisible: both produce a
;; temperature, both report an ECE, and the table looks the same either way.

(let* ((rng (nso-rng 97))
       (dim 128) (ntr 40) (nte 60)
       (mk (lambda (n)
             (let ((xs nil) (ys nil) (ps nil) (i 0))
               (while (< i n)
                 (let ((y (if (= 0 (mod i 2)) 1.0 0.0))
                       (v (make-vector dim 0.0)))
                   (dotimes (j dim) (aset v j (- (nso-rng-float rng) 0.5)))
                   (aset v 3 (+ (aref v 3) (if (= y 1.0) 0.25 -0.25)))
                   (push v xs) (push y ys) (push (/ i 2) ps))
                 (setq i (1+ i)))
               (list (nreverse xs) (nreverse ys) (nreverse ps)))))
       (tr (funcall mk ntr))
       (te (funcall mk nte))
       (fit (nso-probe-fit-and-score (nth 0 tr) (nth 1 tr)
                                     (nth 0 te) (nth 1 te) 600 0.5 0.05 5))
       (te-logits (plist-get fit :logits))
       (t-train (plist-get (nso-temperature-fit (plist-get fit :train-logits)
                                                (nth 1 tr))
                           :temperature))
       (t-oof (plist-get (nso-temperature-fit
                          (nso-probe-oof-logits (nth 0 tr) (nth 1 tr) (nth 2 tr)
                                                (lambda (_rows _ys) #'identity)
                                                3 600 0.5 0.05)
                          (nth 1 tr))
                         :temperature))
       (ece (lambda (tt)
              (plist-get (nso-probe-score
                          (mapcar (lambda (z) (nso-sigmoid (/ z tt))) te-logits)
                          (nth 1 te) 5)
                         :ece))))
  (message "  train acc %.3f, held-out acc %.3f"
           (plist-get (plist-get fit :train) :accuracy)
           (plist-get (plist-get fit :test) :accuracy))
  (message "  T fitted on train logits %.3f -> held-out ECE %.3f (was %.3f)"
           t-train (funcall ece t-train) (funcall ece 1.0))
  (message "  T fitted out of fold     %.3f -> held-out ECE %.3f"
           t-oof (funcall ece t-oof))
  (nso-t-gt "the head overfits, which is the precondition for the trap"
            (plist-get (plist-get fit :train) :accuracy) 0.95)
  (nso-t-lt "a temperature fitted on training logits sharpens (T<1)" t-train 1.0)
  (nso-t-gt "and makes held-out calibration worse"
            (funcall ece t-train) (funcall ece 1.0))
  (nso-t-gt "the out-of-fold temperature is larger" t-oof t-train)
  (nso-t-lt "and does not make held-out calibration worse"
            (funcall ece t-oof) (+ 1e-9 (funcall ece 1.0))))

(nso-t-done "probe")

;;; probe-test.el ends here
