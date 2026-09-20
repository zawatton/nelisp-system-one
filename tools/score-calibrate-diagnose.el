;;; score-calibrate-diagnose.el --- why did A and B disagree? -*- lexical-binding: t; -*-

;; POST HOC.  Nothing here is a gate and nothing here changes the verdict: the
;; pre-registered rule was applied, question B's residual gain landed outside
;; its band, and the retained head's calibration is NOT ACCEPTED.  This file
;; exists because the two questions disagreed in a way that cannot both be
;; true, and a disagreement like that is a fact about the instrument until it
;; is shown to be a fact about the model.
;;
;;   A  fit 165 / eval 75   gain -0.0070  band [-0.0202, +0.0096]  inside
;;   B  fit  30 / eval 45   gain +0.0253  band [-0.0968, +0.0160]  outside
;;
;; The corrected output is the raw output scaled by T = 0.946, which is very
;; nearly not scaled at all.  A says that output is calibrated; B says it is
;; not.  One of the two measurements is describing something else.
;;
;; The suspect is named before it is tested: B's band draws its fit and eval
;; halves as independent examples, and the real halves are SCENARIOS.  Thirty
;; examples from two scenarios are not thirty independent draws -- the fifteen
;; sentences of a scenario share a subject and a verb phrase, and a
;; temperature fitted on two of them can be wrong about the other three in a
;; way fifteen independent draws never would be.  If that is what happened,
;; B's band is too narrow and its verdict is a property of the arithmetic
;; rather than of the head.
;;
;; Two checks, neither of which can rescue the verdict and both of which can
;; refute the suspicion:
;;
;;   1. All ten ways of choosing two held-out scenarios as the fit half.  The
;;      pre-registration fixed one of them; if the other nine sit comfortably
;;      inside the band, the chosen split was unlucky and nothing more.  If
;;      they are all outside, the miscalibration is real and the clustering
;;      story is wrong.
;;   2. A band that MODELS the clustering: the same calibrated-by-construction
;;      stub, but drawn as five groups of fifteen sharing a per-group offset,
;;      swept over offset sizes.  This says how large a scenario effect would
;;      have to be for the observed gain to be ordinary.  The offset is swept
;;      rather than estimated from the data, so it cannot be tuned to cover
;;      the answer.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/score-calibrate-diagnose.el

(defvar nso-dg--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-dg--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-dg--here))

(require 'nso-score)
(require 'nso-probe)
(require 'nso-stub)

(defvar nso-dg--states (expand-file-name "../build/score-states.eld" nso-dg--here))
(defvar nso-dg--draws 400)
(defvar nso-dg--k 5)

(defun nso-dg--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))

(defun nso-dg--pct (sorted q)
  (nth (min (1- (length sorted)) (max 0 (floor (* q (length sorted))))) sorted))

;;; --- a band that can be told to cluster ----------------------------------

(defun nso-dg--draw-grouped (rng ngroups per k offset)
  "NGROUPS groups of PER examples, each group sharing an OFFSET direction.

OFFSET 0 reproduces the independent band the pre-registered run used.  Above
zero, every example in a group has the same vector added to its logits before
the label is drawn AND before the logits are reported, which is what a
scenario does: it moves the whole group together, and a temperature fitted on
some groups meets that shift unmodelled on the others."
  (let ((zs nil) (ys nil))
    (dotimes (_ ngroups)
      (let ((shift (let ((v (make-vector k 0.0)))
                     (dotimes (j k) (aset v j (* offset (- (nso-rng-float rng) 0.5))))
                     v)))
        (dotimes (_ per)
          (let* ((z (let ((v (make-vector k 0.0)))
                      (dotimes (j k)
                        (aset v j (+ (* 4.0 (- (nso-rng-float rng) 0.5)) (aref shift j))))
                      v))
                 (p (nso-softmax-vec z))
                 (u (nso-rng-float rng))
                 (acc 0.0) (lab (1- k)) (done nil))
            (dotimes (j k)
              (unless done
                (setq acc (+ acc (aref p j)))
                (when (>= acc u) (setq lab j done t))))
            (push z zs) (push lab ys)))))
    (cons (nreverse zs) (nreverse ys))))

(defun nso-dg--band-grouped (fit-groups eval-groups per offset draws seed)
  "Gain band when the fit and eval halves are GROUPS rather than examples."
  (let ((rng (nso-rng seed)) (gains nil))
    (dotimes (_ draws)
      (let* ((fit (nso-dg--draw-grouped rng fit-groups per nso-dg--k offset))
             (ev (nso-dg--draw-grouped rng eval-groups per nso-dg--k offset))
             (temp (plist-get (nso-score-nominal-temperature-fit (car fit) (cdr fit))
                              :temperature)))
        (push (- (nso-score-nominal-temperature-nll (car ev) (cdr ev) 1.0)
                 (nso-score-nominal-temperature-nll (car ev) (cdr ev) temp))
              gains)))
    (setq gains (sort gains #'<))
    (cons (nso-dg--pct gains 0.05) (nso-dg--pct gains 0.95))))

;;; --- the run --------------------------------------------------------------

(let* ((saved (with-temp-buffer (insert-file-contents nso-dg--states)
                                (read (buffer-string))))
       (rows (plist-get saved :rows))
       (pool (lambda (r) (nso-pool-last (plist-get r :mid))))
       (train nil) (test nil))
  (dolist (r rows)
    (if (= 0 (mod (plist-get r :scenario) 3)) (push r test) (push r train)))
  (setq train (nreverse train) test (nreverse test))
  (let* ((k nso-dg--k)
         (try (mapcar (lambda (r) (plist-get r :level)) train))
         (tey (mapcar (lambda (r) (plist-get r :level)) test))
         (std (nso-standardizer (mapcar pool train)))
         (head (nso-score-nominal-train
                (mapcar (lambda (r) (nso-standardize std (funcall pool r))) train)
                try k 6000 0.5 0.01))
         (oof (nso-score-nominal-oof-logits
               train try (mapcar (lambda (r) (plist-get r :scenario)) train)
               (lambda (_a _b) pool) k 3 6000 0.5 0.01))
         (temp (plist-get (nso-score-nominal-temperature-fit oof try) :temperature))
         (corrected (nso-score-nominal-scale
                     (mapcar (lambda (r) (nso-score-nominal-logits
                                          head (nso-standardize std (funcall pool r))))
                             test)
                     temp))
         (scenarios (let ((seen nil))
                      (dolist (r test) (unless (memq (plist-get r :scenario) seen)
                                         (push (plist-get r :scenario) seen)))
                      (sort seen #'<))))
    (nso-dg--say "POST HOC.  The verdict stands: question B failed its pre-registered band.\n")
    (nso-dg--say "held-out scenarios: %s, main temperature %.3f\n" scenarios temp)

    ;; --- 1. every choice of the fit half ---------------------------------
    (nso-dg--say "1. All ten ways of choosing two held-out scenarios as the fit half")
    (nso-dg--say "   (the pre-registration fixed (6 12); the band was [-0.0968, +0.0160])\n")
    (nso-dg--say "   fit half | residual T | gain    | vs the band")
    (nso-dg--say "   ---------+------------+---------+------------")
    (let ((gains nil))
      (dolist (a scenarios)
        (dolist (b scenarios)
          (when (< a b)
            (let ((fz nil) (fy nil) (ez nil) (ey nil) (rz corrected) (ry tey))
              (dolist (r test)
                (if (or (= (plist-get r :scenario) a) (= (plist-get r :scenario) b))
                    (progn (push (car rz) fz) (push (car ry) fy))
                  (push (car rz) ez) (push (car ry) ey))
                (setq rz (cdr rz) ry (cdr ry)))
              (setq fz (nreverse fz) fy (nreverse fy) ez (nreverse ez) ey (nreverse ey))
              (let* ((t2 (plist-get (nso-score-nominal-temperature-fit fz fy)
                                    :temperature))
                     (g (- (nso-score-nominal-temperature-nll ez ey 1.0)
                           (nso-score-nominal-temperature-nll ez ey t2))))
                (push g gains)
                (nso-dg--say "   (%2d %2d)  |   %6.3f   | %+.4f | %s%s"
                             a b t2 g
                             (cond ((> g 0.0160) "above")
                                   ((< g -0.0968) "below")
                                   (t "inside"))
                             (if (and (= a 6) (= b 12)) "   <- pre-registered" "")))))))
      (setq gains (sort gains #'<))
      (let ((out 0))
        (dolist (g gains) (when (or (> g 0.0160) (< g -0.0968)) (setq out (1+ out))))
        (nso-dg--say "\n   %d of %d splits fall outside the band, range %+.4f to %+.4f"
                     out (length gains) (car gains) (car (last gains)))
        (nso-dg--say "   %s"
                     (if (> out (/ (length gains) 2))
                         "Most splits fail, so the miscalibration is not an artefact of the choice."
                       "The verdict depends heavily on which two scenarios were named."))))

    ;; --- 2. a band that models the clustering ----------------------------
    (nso-dg--say "\n2. How wide is the band once the fit and eval halves are SCENARIOS?")
    (nso-dg--say "   Five groups of fifteen, sharing a per-group offset, fit on two")
    (nso-dg--say "   and measured on three.  Offset swept, never estimated from the data.\n")
    (nso-dg--say "   group offset | 5%%..95%% band        | covers +0.0253?")
    (nso-dg--say "   -------------+----------------------+----------------")
    (dolist (offset '(0.0 0.25 0.5 1.0 2.0))
      (let* ((b (nso-dg--band-grouped 2 3 15 offset nso-dg--draws (+ 71717 (round (* 100 offset)))))
             (covers (and (>= 0.0253 (car b)) (<= 0.0253 (cdr b)))))
        (nso-dg--say "   %12.2f | [%+.4f, %+.4f] | %s"
                     offset (car b) (cdr b) (if covers "yes" "no"))))
    (nso-dg--say "\n   Offset 0 reproduces the pre-registered band, as it must.")))

;;; score-calibrate-diagnose.el ends here
