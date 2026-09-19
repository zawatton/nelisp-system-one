;;; nso-metrics.el --- calibration and accuracy metrics, and their bins -*- lexical-binding: t; -*-

;;; Commentary:

;; The measuring instruments.  Section 4.3 of `docs/design/01-system-one.org'
;; is the contract: expected calibration error is a *binned* statistic, so a
;; bare ECE number is not a result.  Every function here that returns an ECE
;; returns the bin count, the binning scheme and the sample count with it, in
;; the same plist, because those three decide what the number means.
;;
;; A note on which confidence this is.  ECE below is top-label ECE (Guo et
;; al. 2017): confidence is the largest reported probability and an answer is
;; correct when its argmax is the true label.  That is deliberately *not* the
;; `:confidence' field of an answer, which section 2 of the design doc argues
;; needs its own definition and does not get one here.  Conflating them is the
;; trap that section names; keeping them in separate files is the cheapest way
;; not to fall into it.  Calibrating the reported confidence field is P3's
;; problem, not P0's.
;;
;; NaN policy: a NaN entering a metric produces a quiet wrong number rather
;; than an error, so every entry point checks and signals instead.

;;; Code:

(require 'nso-types)

(defconst nso-default-bins 10
  "Bin count used when none is given.  Always reported alongside an ECE.")

(defconst nso-default-ece-threshold 0.05
  "ECE at or above which `nso-calibration-gate' reports a failure.")

;;; Samples
;;
;; A sample is (:probs LIST-OF-NUMBERS :label INDEX-INTO-PROBS).

(defun nso-sample (probs label)
  "Build a sample from PROBS and the index LABEL of the true option."
  (list :probs probs :label label))

(defun nso--check-sample (s)
  "Signal unless S is a usable sample."
  (let ((probs (plist-get s :probs))
        (label (plist-get s :label)))
    (unless (and (listp probs) probs)
      (error "nso: sample carries no probability list: %S" s))
    (dolist (p probs)
      (unless (numberp p)
        (error "nso: non-numeric probability in sample: %S" p))
      (when (nso-nan-p p)
        (error "nso: NaN probability in sample -- refusing to report a number")))
    (unless (and (integerp label) (>= label 0) (< label (length probs)))
      (error "nso: sample label %S out of range for %d options"
             label (length probs)))))

(defun nso-argmax (probs)
  "Index of the largest element of PROBS, first one on a tie."
  (let ((best -1.0) (idx 0) (i 0))
    (dolist (p probs)
      (when (> p best) (setq best p idx i))
      (setq i (1+ i)))
    idx))

(defun nso-top-confidence (probs)
  "Largest element of PROBS."
  (apply #'max probs))

(defun nso--conf-correct (samples)
  "Return a list of (CONFIDENCE . CORRECT-P), one per sample in SAMPLES."
  (mapcar (lambda (s)
            (nso--check-sample s)
            (let ((probs (plist-get s :probs)))
              (cons (nso-top-confidence probs)
                    (= (nso-argmax probs) (plist-get s :label)))))
          samples))

;;; Binning
;;
;; Equal-width and equal-mass give different answers on the same predictions,
;; which is exactly why the scheme travels with the number.  Equal-width is
;; the gate's scheme; equal-mass is kept as a diagnostic and as the thing that
;; makes the disagreement visible in the suite.

(defun nso--bins-equal-width (pairs m)
  "Group PAIRS into M bins of equal width over [0,1]."
  (let ((buckets (make-vector m nil))
        (out nil)
        (k 0))
    (dolist (p pairs)
      (let ((i (min (1- m) (floor (* (car p) m)))))
        (when (< i 0) (setq i 0))
        (aset buckets i (cons p (aref buckets i)))))
    (while (< k m)
      (push (list :lo (/ (float k) m) :hi (/ (float (1+ k)) m)
                  :pairs (aref buckets k))
            out)
      (setq k (1+ k)))
    (nreverse out)))

(defun nso--bins-equal-mass (pairs m)
  "Group PAIRS into M bins holding as near as possible the same count."
  (let* ((vec (vconcat (sort (copy-sequence pairs)
                             (lambda (a b) (< (car a) (car b))))))
         (n (length vec))
         (out nil)
         (k 0))
    (while (< k m)
      (let* ((start (/ (* k n) m))
             (end (/ (* (1+ k) n) m))
             (group nil)
             (j start))
        (while (< j end)
          (push (aref vec j) group)
          (setq j (1+ j)))
        (push (list :lo (if (< start end) (car (aref vec start)) 0.0)
                    :hi (if (< start end) (car (aref vec (1- end))) 0.0)
                    :pairs group)
              out))
      (setq k (1+ k)))
    (nreverse out)))

(defun nso--bin-row (bin)
  "Summarise one BIN as (:lo :hi :n :conf :acc :gap)."
  (let* ((pairs (plist-get bin :pairs))
         (n (length pairs))
         (sum-conf 0.0)
         (n-correct 0))
    (dolist (p pairs)
      (setq sum-conf (+ sum-conf (car p)))
      (when (cdr p) (setq n-correct (1+ n-correct))))
    (if (= n 0)
        (list :lo (plist-get bin :lo) :hi (plist-get bin :hi)
              :n 0 :conf 0.0 :acc 0.0 :gap 0.0)
      (let ((conf (/ sum-conf n))
            (acc (/ (float n-correct) n)))
        (list :lo (plist-get bin :lo) :hi (plist-get bin :hi)
              :n n :conf conf :acc acc :gap (abs (- acc conf)))))))

;;; Metrics

(defun nso-ece (samples &optional bins scheme)
  "Top-label expected calibration error of SAMPLES.

BINS defaults to `nso-default-bins' and SCHEME to `equal-width'.  Returns
(:ece E :bins M :scheme S :n N :table ROWS); the first four travel together
because E alone does not identify a quantity."
  (let* ((m (or bins nso-default-bins))
         (sch (or scheme 'equal-width))
         (pairs (nso--conf-correct samples))
         (n (length pairs))
         (ece 0.0)
         groups rows)
    (when (= n 0)
      (error "nso: cannot compute an ECE over an empty sample set"))
    (setq groups (cond ((eq sch 'equal-width) (nso--bins-equal-width pairs m))
                       ((eq sch 'equal-mass) (nso--bins-equal-mass pairs m))
                       (t (error "nso: unknown binning scheme %S" sch))))
    (setq rows (mapcar #'nso--bin-row groups))
    (dolist (r rows)
      (setq ece (+ ece (* (/ (float (plist-get r :n)) n) (plist-get r :gap)))))
    (list :ece ece :bins m :scheme sch :n n :table rows)))

(defun nso-calibration-gate (samples &optional threshold bins scheme)
  "Run `nso-ece' on SAMPLES and judge it against THRESHOLD.
Returns the ECE plist with :pass and :threshold prepended."
  (let* ((thr (or threshold nso-default-ece-threshold))
         (res (nso-ece samples bins scheme)))
    (append (list :pass (< (plist-get res :ece) thr) :threshold thr) res)))

(defun nso-brier (samples)
  "Multiclass Brier score of SAMPLES.  Lower is better; it is a proper score."
  (let ((n (length samples)) (sum 0.0))
    (when (= n 0) (error "nso: empty sample set"))
    (dolist (s samples)
      (nso--check-sample s)
      (let ((label (plist-get s :label)) (i 0))
        (dolist (p (plist-get s :probs))
          (let ((d (- p (if (= i label) 1.0 0.0))))
            (setq sum (+ sum (* d d))))
          (setq i (1+ i)))))
    (/ sum n)))

(defun nso-nll (samples)
  "Mean negative log-likelihood of SAMPLES.  Lower is better; proper."
  (let ((n (length samples)) (sum 0.0))
    (when (= n 0) (error "nso: empty sample set"))
    (dolist (s samples)
      (nso--check-sample s)
      (let ((p (nth (plist-get s :label) (plist-get s :probs))))
        (setq sum (+ sum (- (log (max p 1e-15)))))))
    (/ sum n)))

(defun nso-accuracy (samples)
  "Top-label accuracy of SAMPLES."
  (let ((n (length samples)) (k 0))
    (when (= n 0) (error "nso: empty sample set"))
    (dolist (s samples)
      (nso--check-sample s)
      (when (= (nso-argmax (plist-get s :probs)) (plist-get s :label))
        (setq k (1+ k))))
    (/ (float k) n)))

;;; The diagram a human actually reads

(defun nso-format-reliability (result &optional label)
  "Render RESULT, an `nso-ece' plist, as a reliability table under LABEL."
  (let ((s (format "reliability%s\n  n=%d  bins=%d  scheme=%s  ECE=%.4f\n"
                   (if label (format " -- %s" label) "")
                   (plist-get result :n)
                   (plist-get result :bins)
                   (plist-get result :scheme)
                   (plist-get result :ece))))
    (setq s (concat s
                    "  | range       |    n | conf   | acc    | gap    |\n"
                    "  |-------------+------+--------+--------+--------|\n"))
    (dolist (r (plist-get result :table))
      (let ((empty (= 0 (plist-get r :n))))
        (setq s (concat s (format "  | %.2f - %.2f | %4d | %6s | %6s | %6s |\n"
                                  (plist-get r :lo) (plist-get r :hi)
                                  (plist-get r :n)
                                  (if empty "--" (format "%.4f" (plist-get r :conf)))
                                  (if empty "--" (format "%.4f" (plist-get r :acc)))
                                  (if empty "--" (format "%.4f" (plist-get r :gap))))))))
    s))

(provide 'nso-metrics)
;;; nso-metrics.el ends here
