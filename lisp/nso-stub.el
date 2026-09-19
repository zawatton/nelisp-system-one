;;; nso-stub.el --- synthetic models with known calibration -*- lexical-binding: t; -*-

;;; Commentary:

;; P0 builds the instruments before the thing they measure, which means the
;; instruments need subjects whose answer is known in advance.  These are
;; those subjects: models with no encoder, no weights and no learning, whose
;; calibration is a property of how their labels are drawn.
;;
;; `nso-stub-calibrated' is calibrated by construction -- the label is drawn
;; *from* the distribution the stub reports, so "among the answers it gave
;; probability p, a fraction p are right" holds by definition rather than by
;; training.  `nso-stub-tempered' then reports a sharpened or flattened
;; distribution over those same labels, which is miscalibration with a dial
;; on it.
;;
;; The random numbers come from a Park-Miller generator written out here
;; rather than from `random', for two reasons: the sequence has to be
;; identical on every host for the suite to pin numbers at all, and the
;; arithmetic stays far below the 2^61 fixnum ceiling the NeLisp standalone
;; reader imposes.

;;; Code:

(require 'nso-metrics)

;;; Deterministic uniforms

(defun nso-rng (seed)
  "A generator state for SEED.  Park-Miller: x <- 16807 x mod 2^31-1."
  (list (1+ (mod (abs seed) 2147483645))))

(defun nso-rng-next (rng)
  "Advance RNG and return a raw integer in [1, 2147483646]."
  (let ((x (mod (* 16807 (car rng)) 2147483647)))
    (setcar rng x)
    x))

(defun nso-rng-float (rng)
  "Advance RNG and return a uniform float in [0,1)."
  (/ (float (1- (nso-rng-next rng))) 2147483646.0))

;;; Distributions

(defun nso-softmax (logits)
  "Softmax of LOGITS, shifted by the max before exponentiating.
The shift is clamped at -700 because the NeLisp standalone reader returns
NaN for `exp' of a large negative argument and hangs near -1e6; see
`nelisp-llm/lisp/nl-llm-compat.el'."
  (let* ((mx (apply #'max logits))
         (ex (mapcar (lambda (l) (exp (max -700.0 (- l mx)))) logits))
         (sum (apply #'+ ex)))
    (mapcar (lambda (e) (/ e sum)) ex)))

(defun nso-sample-index (rng probs)
  "Draw an index from the categorical distribution PROBS using RNG."
  (let ((u (nso-rng-float rng))
        (acc 0.0)
        (i 0)
        (res nil))
    (dolist (p probs)
      (when (null res)
        (setq acc (+ acc p))
        (when (< u acc) (setq res i)))
      (setq i (1+ i)))
    (or res (1- (length probs)))))

;;; The stubs

(defun nso-stub-calibrated (seed n k)
  "N samples over K options that are calibrated by construction.

Each sample gets a random distribution, spread over a range of sharpness so
the reliability bins are not all one column, and its label is then drawn
from that distribution.  Any deviation from perfect calibration in the
result is finite-sample noise, which is itself something the suite measures."
  (let ((rng (nso-rng seed))
        (out nil))
    (dotimes (_ n)
      (let* ((scale (+ 0.3 (* 5.0 (nso-rng-float rng))))
             (logits nil))
        (dotimes (_ k)
          (push (* scale (- (nso-rng-float rng) 0.5)) logits))
        (let* ((probs (nso-softmax logits))
               (label (nso-sample-index rng probs)))
          (push (nso-sample probs label) out))))
    (nreverse out)))

(defun nso-stub-tempered (samples temp)
  "Report SAMPLES' distributions raised to the power 1/TEMP and renormalised.

The labels are untouched, so the reported probabilities no longer describe
how often the stub is right.  TEMP below 1 sharpens (overconfident); above 1
flattens (underconfident).  TEMP 0.25 is the 4x-overconfident control that
section 4.4 of the design doc requires the calibration gate to catch."
  (mapcar (lambda (s)
            (nso-sample
             (nso-softmax (mapcar (lambda (p) (/ (log (max p 1e-15)) temp))
                                  (plist-get s :probs)))
             (plist-get s :label)))
          samples))

(defun nso-stub-cancelling (seed n)
  "N samples whose miscalibration cancels inside a coarse bin.

Half report confidence 0.55 and are right 85% of the time; half report 0.95
and are right 65%.  Pooled, mean confidence and mean accuracy are both 0.75,
so a single bin sees a perfectly calibrated model.  Ten bins see two gaps of
0.30.  This stub is aimed at the metric, not at a model: it is the control
that proves the bin count is load-bearing rather than cosmetic."
  (let ((rng (nso-rng seed))
        (out nil)
        (i 0))
    (while (< i n)
      (let* ((low (= 0 (mod i 2)))
             (conf (if low 0.55 0.95))
             (acc (if low 0.85 0.65))
             (label (if (< (nso-rng-float rng) acc) 0 1)))
        (push (nso-sample (list conf (- 1.0 conf)) label) out))
      (setq i (1+ i)))
    (nreverse out)))

;;; Ill-typed answers, for the type gate

(defun nso-stub-answer-valid (question)
  "A well-formed answer to QUESTION, for the gate's happy path."
  (let* ((options (plist-get question :options))
         (k (length options))
         (head (- 1.0 (* 0.1 (1- k))))
         (probs nil)
         (rest (cdr options)))
    (push (cons (car options) head) probs)
    (dolist (o rest) (push (cons o 0.1) probs))
    (list :choice (car options)
          :probabilities (nreverse probs)
          :confidence head)))

(defun nso-stub-answer-out-of-set (question)
  "An answer to QUESTION naming an option that was never declared.
The headline negative control for the type gate."
  (let ((a (nso-stub-answer-valid question)))
    (plist-put (copy-sequence a) :choice 'nso--never-declared)))

(provide 'nso-stub)
;;; nso-stub.el ends here
