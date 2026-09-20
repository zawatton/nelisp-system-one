;;; p3-gain.el --- calibration without bins, and its noise floor -*- lexical-binding: t; -*-

;; ECE turned out to be unmeasurable at n=84: on a stub that is calibrated by
;; construction, the estimator reports 0.043 to 0.082 depending on bin count,
;; and the gate was 0.05.  The instrument's own noise exceeded the quantity.
;;
;; A proper scoring rule has no bins, so it has no such floor from binning --
;; but NLL alone does not isolate calibration, because it mixes calibration
;; with sharpness.  The standard way to separate them needs no bins either:
;;
;;   CALIBRATION GAIN = NLL(uncorrected) - NLL(recalibrated)
;;
;; with the recalibrator fitted OUT OF FOLD on the training split, as
;; everywhere else here.  If a model is already calibrated there is nothing
;; for the recalibrator to find and the gain is zero or slightly negative --
;; slightly, because a temperature fitted on finite data is noisy and applying
;; a noisy correction to fresh data costs a little.  If the model is
;; miscalibrated the gain is positive and measures how much.
;;
;; *This is not a pre-registered test.*  The real model's gain was already
;; computed in the P3 run -- 0.3830 to 0.3610, a gain of 0.0220 -- before this
;; analysis existed, so a threshold set now could be set around it.  What this
;; file can honestly supply is the floor and the spread, measured on stubs
;; that never touch the real predictions, so the reader can judge 0.0220
;; against them.  Presented as an interpretation, not as a gate.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/p3-gain.el

(defvar nso-pg--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-pg--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-pg--here))

(require 'nso-head)
(require 'nso-stub)

(defvar nso-pg--ntrain 168)
(defvar nso-pg--ntest 84)
(defvar nso-pg--draws 400)
(defvar nso-pg--observed 0.0220 "The gain the P3 run measured, for reference.")

(defun nso-pg--draw (rng n sharpen)
  "N logit/label pairs.  Labels come from the UNSHARPENED logit, so SHARPEN
above 1 produces a model that is overconfident by exactly that factor."
  (let ((zs nil) (ys nil))
    (dotimes (_ n)
      (let* ((z (* 4.0 (- (nso-rng-float rng) 0.5)))
             (y (if (< (nso-rng-float rng) (nso-sigmoid z)) 1.0 0.0)))
        (push (* sharpen z) zs)
        (push y ys)))
    (cons (nreverse zs) (nreverse ys))))

(defun nso-pg--gain (rng sharpen)
  "One draw's calibration gain: NLL before minus NLL after recalibration."
  (let* ((tr (nso-pg--draw rng nso-pg--ntrain sharpen))
         (te (nso-pg--draw rng nso-pg--ntest sharpen))
         (temp (plist-get (nso-temperature-fit (car tr) (cdr tr)) :temperature))
         (before (nso-platt-nll (car te) (cdr te) 1.0 0.0))
         (after (nso-platt-nll (car te) (cdr te) (/ 1.0 temp) 0.0)))
    (- before after)))

(defun nso-pg--study (label sharpen)
  (let ((gains nil) (rng (nso-rng 4242)) (i 0))
    (while (< i nso-pg--draws)
      (push (nso-pg--gain rng sharpen) gains)
      (setq i (1+ i)))
    (setq gains (sort gains #'<))
    (let* ((n (length gains))
           (mean (let ((s 0.0)) (dolist (g gains) (setq s (+ s g))) (/ s n)))
           (p95 (nth (min (1- n) (floor (* 0.95 n))) gains))
           (p05 (nth (floor (* 0.05 n)) gains))
           (over (let ((k 0))
                   (dolist (g gains) (when (>= g nso-pg--observed) (setq k (1+ k))))
                   (/ (float k) n))))
      (princ (format "  %-26s mean %+.4f   5%%..95%% [%+.4f, %+.4f]   P(gain >= %.4f) = %.3f\n"
                     label mean p05 p95 nso-pg--observed over))
      (list :mean mean :p95 p95 :over over))))

(princ (format "calibration gain = NLL(uncorrected) - NLL(recalibrated)\n"))
(princ (format "%d draws, train %d / held-out %d, temperature fitted on train\n\n"
               nso-pg--draws nso-pg--ntrain nso-pg--ntest))

(let ((cal (nso-pg--study "calibrated by construction" 1.0))
      (over (nso-pg--study "4x overconfident" 4.0)))
  (princ "\n")
  (princ (format "The P3 run measured a gain of %.4f on the real model.\n" nso-pg--observed))
  (princ (format "Against a model that needs no correction, that lands at the %.1f%% point.\n"
                 (* 100 (- 1.0 (plist-get cal :over)))))
  (princ (format "A 4x-overconfident model at this n gains %+.4f on average -- %.0fx more.\n"
                 (plist-get over :mean)
                 (/ (plist-get over :mean) (max 1e-9 (abs nso-pg--observed)))))
  (princ "\n")
  (princ (if (> (plist-get cal :over) 0.05)
             "So 0.0220 is inside what a calibrated model produces by chance here:\nthe run does not show miscalibration.\n"
           "So 0.0220 is outside what a calibrated model produces by chance here:\nthe run does show miscalibration, and temperature removed it.\n")))

;;; p3-gain.el ends here
