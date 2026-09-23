;;; score-calibrate-band.el --- how well is the threshold itself known? -*- lexical-binding: t; -*-

;; The pre-registered rule compares a gain against the 5-95% band of a
;; calibrated-by-construction model.  That band is a quantity to be ESTIMATED,
;; and the run estimated it from 400 draws without ever asking how much the
;; estimate moves.  The post-hoc diagnostic then computed the same band -- same
;; sizes, same construction, different seed -- and got a 95th percentile of
;; +0.0268 where the run had +0.0160.
;;
;; Question B's gain is +0.0253.  It is above one of those numbers and below
;; the other.  The verdict was therefore decided by the seed, which is not a
;; property of the head.
;;
;; This estimates the same pre-registered quantity properly: many more draws,
;; and a bootstrap interval on the percentile itself so the threshold arrives
;; with an error bar like any other measurement.  Estimating a committed
;; quantity more precisely is not amending the rule -- the rule says "the 5-95%
;; band", and 400 draws was an imprecise reading of it.  It can move the
;; verdict in either direction and the result is reported whichever way it goes.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/score-calibrate-band.el

(defvar nso-bd--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-bd--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-bd--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-score)
(require 'nso-stub)

(defvar nso-bd--k 5)
(defvar nso-bd--draws (string-to-number (or (getenv "NSO_BAND_DRAWS") "12000")))
(defvar nso-bd--boots 2000)

;; The two observed gains, from the pre-registered run.  Named constants so the
;; comparison below cannot drift from what was actually measured.
(defvar nso-bd--gain-a -0.0070)
(defvar nso-bd--gain-b 0.0253)

(defun nso-bd--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))

(defun nso-bd--pct (sorted q)
  (nth (min (1- (length sorted)) (max 0 (floor (* q (length sorted))))) sorted))

(defun nso-bd--draw (rng n k sharpen)
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

(defun nso-bd--gains (nfit neval sharpen draws seed)
  (let ((rng (nso-rng seed)) (out nil))
    (dotimes (_ draws)
      (let* ((fit (nso-bd--draw rng nfit nso-bd--k sharpen))
             (ev (nso-bd--draw rng neval nso-bd--k sharpen))
             (temp (plist-get (nso-score-nominal-temperature-fit (car fit) (cdr fit))
                              :temperature)))
        (push (- (nso-score-nominal-temperature-nll (car ev) (cdr ev) 1.0)
                 (nso-score-nominal-temperature-nll (car ev) (cdr ev) temp))
              out)))
    (sort out #'<)))

(defun nso-bd--pct-ci (gains q boots seed)
  "Bootstrap interval for the Qth percentile of GAINS."
  (let* ((v (vconcat gains)) (n (length v)) (rng (nso-rng seed)) (ps nil))
    (dotimes (_ boots)
      (let ((s (make-vector n 0.0)))
        (dotimes (i n) (aset s i (aref v (mod (nso-rng-next rng) n))))
        (push (nso-bd--pct (sort (append s nil) #'<) q) ps)))
    (setq ps (sort ps #'<))
    (cons (nso-bd--pct ps 0.025) (nso-bd--pct ps 0.975))))

(nso-bd--say "The pre-registered band, estimated properly.  %d draws.\n" nso-bd--draws)

(dolist (cfg (list (list "A  raw output" 165 75 nso-bd--gain-a 55501)
                   (list "B  corrected output" 30 45 nso-bd--gain-b 55502)))
  (let* ((label (nth 0 cfg)) (nfit (nth 1 cfg)) (neval (nth 2 cfg))
         (gain (nth 3 cfg)) (seed (nth 4 cfg))
         (gains (nso-bd--gains nfit neval 1.0 nso-bd--draws seed))
         (p05 (nso-bd--pct gains 0.05))
         (p95 (nso-bd--pct gains 0.95))
         (ci05 (nso-bd--pct-ci gains 0.05 nso-bd--boots (+ seed 1)))
         (ci95 (nso-bd--pct-ci gains 0.95 nso-bd--boots (+ seed 2)))
         (inside (and (>= gain p05) (<= gain p95)))
         ;; Decided by the seed when the observed gain sits inside the
         ;; threshold's own interval: the verdict would flip with another draw
         ;; of the SIMULATION, which is not a fact about the model.
         (fragile (and (>= gain (car ci95)) (<= gain (cdr ci95)))))
    (nso-bd--say "%s   n = %d fit / %d eval" label nfit neval)
    (nso-bd--say "  observed gain      %+.4f" gain)
    (nso-bd--say "  5th  percentile    %+.4f   95%% CI [%+.4f, %+.4f]" p05 (car ci05) (cdr ci05))
    (nso-bd--say "  95th percentile    %+.4f   95%% CI [%+.4f, %+.4f]" p95 (car ci95) (cdr ci95))
    (nso-bd--say "  verdict            %s"
                 (if inside "INSIDE the band -- calibrated"
                   "OUTSIDE the band -- not calibrated"))
    (nso-bd--say "  robust?            %s\n"
                 (if fragile
                     "NO -- the gain lies inside the threshold's own interval, so the call is the simulation's, not the head's"
                   "yes -- the gain is clear of the threshold's uncertainty"))))

;; The control, at B's sizes: a band that admits everything proves nothing, so
;; the separation is re-checked at the higher draw count too.
(let* ((over (nso-bd--gains 30 45 0.25 (/ nso-bd--draws 4) 55503))
       (cal (nso-bd--gains 30 45 1.0 (/ nso-bd--draws 4) 55504)))
  (nso-bd--say "control at 30/45: 4x overconfident [%+.4f, %+.4f] against calibrated [%+.4f, %+.4f] -- %s"
               (nso-bd--pct over 0.05) (nso-bd--pct over 0.95)
               (nso-bd--pct cal 0.05) (nso-bd--pct cal 0.95)
               (if (> (nso-bd--pct over 0.05) (nso-bd--pct cal 0.95))
                   "separated" "OVERLAPS")))

;;; score-calibrate-band.el ends here
