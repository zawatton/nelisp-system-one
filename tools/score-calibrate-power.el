;;; score-calibrate-power.el --- choose the re-test's split sizes before writing it -*- lexical-binding: t; -*-

;; The calibration re-test needs a three-way scenario split -- train, calibrate,
;; test -- and the sizes have to be chosen.  Choosing them by eye is how the
;; first attempt ended up with a calibrate half of two scenarios and a verdict
;; the simulation's seed could flip.
;;
;; Everything here runs on calibrated-by-construction stubs and touches no
;; encoded state, so it decides the design without seeing the answer.  It is
;; the same move P3's bin-count study made: pick the instrument's parameters
;; from the instrument's own behaviour, before the instrument is pointed at
;; anything.
;;
;; Three quantities decide it:
;;
;;   band width   how tightly the null is pinned.  A wide band accepts
;;                everything, which is not evidence.
;;   p95 error    the 95th percentile is the threshold, and the first attempt
;;                compared a gain against it without an error bar.  Two
;;                readings of the SAME band at 400 draws gave +0.0160 and
;;                +0.0268 while the gain was +0.0253.  The threshold's own
;;                interval must be narrow relative to what is being judged.
;;   power        a mildly overconfident model must be caught.  A 4x stub is
;;                caught by anything; the design question is where the floor
;;                sits, so 1.25x and 1.5x are what gets swept.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/score-calibrate-power.el

(defvar nso-pw--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-pw--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-pw--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-score)
(require 'nso-stub)

(defvar nso-pw--k 5)
(defvar nso-pw--draws (string-to-number (or (getenv "NSO_POWER_DRAWS") "1500")))
(defvar nso-pw--boots 800)

(defun nso-pw--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))

(defun nso-pw--pct (sorted q)
  (nth (min (1- (length sorted)) (max 0 (floor (* q (length sorted))))) sorted))

(defun nso-pw--draw (rng n k sharpen)
  (let ((zs nil) (ys nil))
    (dotimes (_ n)
      (let* ((z (let ((v (make-vector k 0.0)))
                  (dotimes (j k) (aset v j (* 4.0 (- (nso-rng-float rng) 0.5))))
                  v))
             (p (nso-softmax-vec z))
             (u (nso-rng-float rng))
             (acc 0.0) (lab (1- k)) (done nil))
        (dotimes (j k)
          (unless done
            (setq acc (+ acc (aref p j)))
            (when (>= acc u) (setq lab j done t))))
        (push (car (nso-score-nominal-scale (list z) sharpen)) zs)
        (push lab ys)))
    (cons (nreverse zs) (nreverse ys))))

(defun nso-pw--gains (nfit neval sharpen draws seed)
  (let ((rng (nso-rng seed)) (out nil))
    (dotimes (_ draws)
      (let* ((fit (nso-pw--draw rng nfit nso-pw--k sharpen))
             (ev (nso-pw--draw rng neval nso-pw--k sharpen))
             (temp (plist-get (nso-score-nominal-temperature-fit (car fit) (cdr fit))
                              :temperature)))
        (push (- (nso-score-nominal-temperature-nll (car ev) (cdr ev) 1.0)
                 (nso-score-nominal-temperature-nll (car ev) (cdr ev) temp))
              out)))
    (sort out #'<)))

(defun nso-pw--p95-ci (gains boots seed)
  (let* ((v (vconcat gains)) (n (length v)) (rng (nso-rng seed)) (ps nil))
    (dotimes (_ boots)
      (let ((s (make-vector n 0.0)))
        (dotimes (i n) (aset s i (aref v (mod (nso-rng-next rng) n))))
        (push (nso-pw--pct (sort (append s nil) #'<) 0.95) ps)))
    (setq ps (sort ps #'<))
    (cons (nso-pw--pct ps 0.025) (nso-pw--pct ps 0.975))))

(nso-pw--say "Choosing the re-test's split sizes.  %d draws per cell, stubs only.\n"
             nso-pw--draws)
(nso-pw--say "  calibrate/test | band [5%%, 95%%]      | p95 CI width | power 1.25x | power 1.5x")
(nso-pw--say "  ---------------+----------------------+--------------+-------------+-----------")

(let ((seed 30001))
  ;; Two sweeps.  The first varies total size; the second holds the scenario
  ;; budget FIXED at sixteen (240 examples, leaving twenty scenarios to train
  ;; on) and moves the boundary between calibrate and test.  The calibrator
  ;; fits one scalar and the evaluator estimates a mean, so there is no reason
  ;; to expect an even split to be the right one, and no reason to guess.
  (dolist (cfg (append
                '((30 45) (75 75) (120 120) (135 135) (150 150) (180 180))
                (when (getenv "NSO_POWER_SPLIT")
                  '((60 180) (90 150) (120 120) (150 90) (180 60)))
                ;; The four three-way splits of thirty-six scenarios that are
                ;; actually on the table, measured rather than interpolated
                ;; from their neighbours.  train / calibrate / test is
                ;; 18/6/12, 20/6/10, 16/8/12, 16/10/10.
                (when (getenv "NSO_POWER_FINAL")
                  '((90 180) (90 150) (120 180) (150 150)))))
    (let* ((nfit (nth 0 cfg)) (neval (nth 1 cfg))
           (cal (nso-pw--gains nfit neval 1.0 nso-pw--draws seed))
           (p05 (nso-pw--pct cal 0.05))
           (p95 (nso-pw--pct cal 0.95))
           (ci (nso-pw--p95-ci cal nso-pw--boots (+ seed 1)))
           ;; Power: fraction of draws from a mildly overconfident model whose
           ;; gain clears the calibrated band's 95th percentile.
           (pow (lambda (sharpen s)
                  (let ((g (nso-pw--gains nfit neval sharpen
                                          (/ nso-pw--draws 2) s))
                        (hit 0))
                    (dolist (x g) (when (> x p95) (setq hit (1+ hit))))
                    (/ (float hit) (length g)))))
           (p125 (funcall pow 0.80 (+ seed 2)))   ; 1.25x overconfident
           (p150 (funcall pow (/ 1.0 1.5) (+ seed 3))))
      (nso-pw--say "  %7d/%-6d | [%+.4f, %+.4f] | %12.4f | %11.3f | %10.3f"
                   nfit neval p05 p95 (- (cdr ci) (car ci)) p125 p150)
      (setq seed (+ seed 10)))))

(nso-pw--say "")
(nso-pw--say "Read this as: how small a miscalibration the re-test could catch, and")
(nso-pw--say "how precisely its own threshold is known.  The p95 CI width is the")
(nso-pw--say "quantity the first attempt never looked at; at 400 draws and 30/45 it")
(nso-pw--say "was about 0.011, against a margin of 0.005.")

;;; score-calibrate-power.el ends here
