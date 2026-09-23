;;; calibration-test.el --- the calibration gate, and controls on the gate itself -*- lexical-binding: t; -*-

;;; Commentary:

;; Two layers of control here, and the second is the one that is usually
;; missing.
;;
;; The first layer aims at the model: a stub that is calibrated by
;; construction must come out green, and stubs sharpened and flattened by a
;; factor of four must come out red.  A gate that reports "well calibrated"
;; for all three is measuring nothing, and its passing output is identical to
;; a working gate's.
;;
;; The second layer aims at the metric.  ECE is a binned statistic and the
;; binning is not a display choice: `nso-stub-cancelling' is genuinely
;; miscalibrated, by 0.30, and one bin cannot see it at all.  That row is the
;; reason this file reports a bin count and a sample count next to every
;; number rather than a bare ECE.

;;; Code:

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-metrics)
(require 'nso-stub)
(require 'nso-head)                    ; nso-sigmoid, the temperature fit
(load (expand-file-name "nso-test-helper.el"
                        (file-name-directory (or load-file-name buffer-file-name))))

(message "== calibration ==")

(defconst nso-test-n 4000)
(defconst nso-test-k 4)
(defconst nso-test-seed 20260919)

(let* ((calibrated (nso-stub-calibrated nso-test-seed nso-test-n nso-test-k))
       (over (nso-stub-tempered calibrated 0.25))
       (under (nso-stub-tempered calibrated 4.0))
       (e-cal (nso-ece calibrated))
       (e-over (nso-ece over))
       (e-under (nso-ece under)))

  (message "%s" (nso-format-reliability e-cal "calibrated by construction"))
  (message "%s" (nso-format-reliability e-over "4x overconfident (T=0.25)"))
  (message "%s" (nso-format-reliability e-under "4x underconfident (T=4.0)"))

  ;; --- layer one: the gate judges the model -------------------------------

  (nso-t-green "the calibrated stub passes the gate"
               (nso-calibration-gate calibrated))
  (nso-t-red "the 4x-overconfident stub fails the gate"
             (nso-calibration-gate over))
  (nso-t-red "the 4x-underconfident stub fails the gate"
             (nso-calibration-gate under))

  (nso-t-lt "calibrated ECE is below overconfident ECE"
            (plist-get e-cal :ece) (plist-get e-over :ece))
  (nso-t-lt "calibrated ECE is below underconfident ECE"
            (plist-get e-cal :ece) (plist-get e-under :ece))

  ;; A calibrated model's mean confidence matches its accuracy globally.  This
  ;; is weaker than ECE -- it is ECE with one bin -- and it is here because a
  ;; stub that failed it would be broken at construction, not miscalibrated.
  (nso-t-num "calibrated stub: mean confidence tracks accuracy"
             (nso-accuracy calibrated)
             (let ((s 0.0))
               (dolist (x calibrated)
                 (setq s (+ s (nso-top-confidence (plist-get x :probs)))))
               (/ s (length calibrated)))
             0.03)

  ;; --- proper scoring rules see it too ------------------------------------
  ;;
  ;; Section 4.2 rests on NLL and Brier being minimised by the true
  ;; probability.  If that did not hold here, the training objective proposed
  ;; for P3 would be unsupported.

  (nso-t-lt "NLL prefers the calibrated stub over the overconfident one"
            (nso-nll calibrated) (nso-nll over))
  (nso-t-lt "NLL prefers the calibrated stub over the underconfident one"
            (nso-nll calibrated) (nso-nll under))
  (nso-t-lt "Brier prefers the calibrated stub over the overconfident one"
            (nso-brier calibrated) (nso-brier over))
  (nso-t-lt "Brier prefers the calibrated stub over the underconfident one"
            (nso-brier calibrated) (nso-brier under))

  ;; Sharpening does not reorder anything, so accuracy must be untouched.  A
  ;; difference here would mean the temperature control is changing the model
  ;; rather than only its confidence.
  (nso-t-num "temperature does not change accuracy"
             (nso-accuracy over) (nso-accuracy calibrated) 1e-12)

  ;; --- layer two: controls on the metric ----------------------------------

  ;; Equal-width and equal-mass agree here to floating-point noise, and that
  ;; is not luck.  When the gap keeps one sign across the whole confidence
  ;; range -- as it does for a uniformly overconfident model -- the absolute
  ;; value can be pulled outside the weighted sum, and what is left telescopes
  ;; to |mean confidence - mean accuracy| whatever the partition was.  So the
  ;; binning is not always a choice that matters.  It matters exactly where
  ;; the sign of the gap changes, which is what the cancelling stub below is
  ;; built to do, and that is a statement about the data rather than about
  ;; ECE's parameters.
  (let* ((w (nso-ece over 10 'equal-width))
         (m (nso-ece over 10 'equal-mass))
         (m20 (nso-ece over 20 'equal-mass))
         (w50 (nso-ece over 50 'equal-width))
         (mean-conf (let ((s 0.0))
                      (dolist (x over)
                        (setq s (+ s (nso-top-confidence (plist-get x :probs)))))
                      (/ s (length over))))
         (collapse (abs (- mean-conf (nso-accuracy over)))))
    (message (concat "  uniformly overconfident stub: width/10 %.9f  mass/10 %.9f"
                     "  mass/20 %.9f  width/50 %.9f  |conf-acc| %.9f")
             (plist-get w :ece) (plist-get m :ece) (plist-get m20 :ece)
             (plist-get w50 :ece) collapse)
    (nso-t-num "a constant-sign gap makes the binning scheme irrelevant"
               (plist-get m :ece) (plist-get w :ece) 1e-9)
    (nso-t-num "and ECE then collapses to |mean confidence - mean accuracy|"
               (plist-get w :ece) collapse 1e-9)
    ;; True for any partition, by the triangle inequality: the binned estimate
    ;; can only ever be at or above the global gap, never below it.  A bin
    ;; count can therefore hide miscalibration but never invent it.
    (nso-t "no partition can report less than the global gap"
           (and (>= (+ 1e-12 (plist-get w :ece)) collapse)
                (>= (+ 1e-12 (plist-get m :ece)) collapse)
                (>= (+ 1e-12 (plist-get m20 :ece)) collapse)
                (>= (+ 1e-12 (plist-get w50 :ece)) collapse)))))

;; The cancelling stub: miscalibrated by 0.30, invisible to a single bin.
(let* ((cancel (nso-stub-cancelling 7 nso-test-n))
       (e1 (nso-ece cancel 1))
       (e10 (nso-ece cancel 10)))
  (message "%s" (nso-format-reliability e1 "cancelling stub, 1 bin"))
  (message "%s" (nso-format-reliability e10 "cancelling stub, 10 bins"))

  (nso-t-num "one bin sees a perfectly calibrated model" (plist-get e1 :ece) 0.0 0.03)
  (nso-t-green "and the gate therefore passes it -- this is the gate lying"
               (nso-calibration-gate cancel nil 1))
  (nso-t-num "ten bins see the real 0.30 miscalibration"
             (plist-get e10 :ece) 0.30 0.03)
  (nso-t-red "and the gate at ten bins rejects it"
             (nso-calibration-gate cancel nil 10)))

;; Finite-sample behaviour of the estimator.  With few samples per bin the
;; per-bin accuracy is a noisy estimate of the per-bin confidence, and |gap|
;; cannot be negative, so the noise has nowhere to cancel: a perfectly
;; calibrated model scores further from zero the less data it is given.
(let ((small 0.0) (large 0.0) (seeds 6) (i 0))
  (while (< i seeds)
    (setq small (+ small (plist-get (nso-ece (nso-stub-calibrated (+ 100 i) 100 4)) :ece)))
    (setq large (+ large (plist-get (nso-ece (nso-stub-calibrated (+ 100 i) 4000 4)) :ece)))
    (setq i (1+ i)))
  (setq small (/ small seeds) large (/ large seeds))
  (message "  mean ECE of a CALIBRATED stub over %d seeds: n=100 %.4f, n=4000 %.4f"
           seeds small large)
  (nso-t-gt "a calibrated model scores worse on less data (bias away from zero)"
            small large))

;; --- refusals ------------------------------------------------------------
;;
;; A NaN must stop the run.  The alternative is a number that looks like a
;; result.

(nso-t-signals "a NaN probability signals instead of returning a number"
               (lambda () (nso-ece (list (nso-sample (list (/ 0.0 0.0) 1.0) 0)))))
(nso-t-signals "an empty sample set signals"
               (lambda () (nso-ece nil)))
(nso-t-signals "an out-of-range label signals"
               (lambda () (nso-ece (list (nso-sample (list 0.5 0.5) 7)))))
(nso-t-signals "an unknown binning scheme signals"
               (lambda () (nso-ece (list (nso-sample (list 0.5 0.5) 0)) 10 'quartiles)))

;; --- the plist carries what the number needs to be read ------------------

(let ((r (nso-ece (nso-stub-calibrated 1 200 3) 15 'equal-mass)))
  (nso-t "ECE reports its bin count" (= 15 (plist-get r :bins)))
  (nso-t "ECE reports its sample count" (= 200 (plist-get r :n)))
  (nso-t "ECE reports its binning scheme" (eq 'equal-mass (plist-get r :scheme)))
  (nso-t "the reliability table has one row per bin"
         (= 15 (length (plist-get r :table))))
  (nso-t "the table's counts sum to n"
         (= 200 (let ((s 0))
                  (dolist (row (plist-get r :table))
                    (setq s (+ s (plist-get row :n))))
                  s))))


;;; --- a calibration statistic that works at small n -----------------------
;;
;; ECE could not be measured on the P1 held-out split: at n=84 a stub that is
;; calibrated BY CONSTRUCTION reports 0.043 to 0.082 depending on bin count,
;; against a gate of 0.05.  The binning is the floor.
;;
;; The recalibration gain has no bins:
;;
;;   gain = NLL(uncorrected) - NLL(recalibrated)
;;
;; with the recalibrator fitted on a separate split.  A model that needs no
;; correction has nothing to gain, and in fact loses a little, because a
;; temperature fitted on finite data is noisy and applying a noisy correction
;; to fresh data costs something.  A miscalibrated one gains in proportion to
;; how wrong it was.  These checks pin both directions at the n where ECE
;; failed.

(defun ct--gain-draw (rng n sharpen)
  "N logit/label pairs; labels from the UNSHARPENED logit."
  (let ((zs nil) (ys nil))
    (dotimes (_ n)
      (let* ((z (* 4.0 (- (nso-rng-float rng) 0.5)))
             (y (if (< (nso-rng-float rng) (nso-sigmoid z)) 1.0 0.0)))
        (push (* sharpen z) zs) (push y ys)))
    (cons (nreverse zs) (nreverse ys))))

(defun ct--gain (rng sharpen)
  (let* ((tr (ct--gain-draw rng 168 sharpen))
         (te (ct--gain-draw rng 84 sharpen))
         (temp (plist-get (nso-temperature-fit (car tr) (cdr tr)) :temperature)))
    (- (nso-platt-nll (car te) (cdr te) 1.0 0.0)
       (nso-platt-nll (car te) (cdr te) (/ 1.0 temp) 0.0))))

(let* ((rng (nso-rng 4242))
       (mean (lambda (sharpen draws)
               (let ((s 0.0) (i 0))
                 (while (< i draws) (setq s (+ s (ct--gain rng sharpen)) i (1+ i)))
                 (/ s draws))))
       (g-cal (funcall mean 1.0 120))
       (g-over (funcall mean 4.0 120)))
  (message "  recalibration gain at n=84: calibrated %+.4f, 4x overconfident %+.4f"
           g-cal g-over)
  ;; The sign is the point: a correction fitted for a model that needs none is
  ;; noise, and noise applied to fresh data costs rather than pays.
  (nso-t-lt "a calibrated model gains nothing from recalibration" g-cal 0.01)
  (nso-t-gt "a 4x-overconfident one gains a great deal" g-over 0.2)
  (nso-t-gt "and the two are separated by more than an order of magnitude"
            g-over (* 10.0 (max 0.001 (abs g-cal))))
  ;; The contrast with ECE at the same n, which is the reason this exists.
  (let ((ece-floor (let ((s 0.0) (i 0))
                     (while (< i 40)
                       (setq s (+ s (plist-get (nso-ece (nso-stub-calibrated
                                                         (+ 500 i) 84 2) 10)
                                               :ece))
                             i (1+ i)))
                     (/ s 40))))
    (message "  for contrast, ECE at n=84 / 10 bins on a calibrated stub: %.4f" ece-floor)
    (nso-t-gt "while ECE at this n cannot even reach its own gate" ece-floor 0.05)))

;; argmax on values that are not probabilities.  The function is named for
;; probabilities and was seeded with a constant below all of them; these are
;; the inputs that seeding could not handle, and one of them cost an afternoon
;; of believing a monotone transformation had reversed an order.
(nso-t-num "argmax of all-negative values picks the largest, not the first"
           (nso-argmax '(-5.0 -2.0 -9.0)) 1 0.5)
(nso-t-num "argmax of values below the old -1.0 seed" (nso-argmax '(-3.0 -4.0)) 0 0.5)
(nso-t-num "argmax still works on probabilities" (nso-argmax '(0.1 0.7 0.2)) 1 0.5)
(nso-t-signals "argmax refuses an empty list" (lambda () (nso-argmax nil)))

(nso-t-done "calibration")

;;; calibration-test.el ends here
