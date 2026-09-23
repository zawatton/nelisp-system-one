;;; score-calibrate.el --- is the head Score ships actually calibrated? -*- lexical-binding: t; -*-

;; The nominal head won gate 2, so it is the one that has to be calibrated, and
;; the ordinal head's temperature -- already reported -- belongs to a head that
;; was dropped.  Reads the states the encode wrote; needs no GPU and no donor.
;;
;; The rule is section 6's "Calibration of the retained head", committed before
;; this file was run.  In short: P3's binning-free gain, the two questions kept
;; apart, both bands computed at the sizes they are compared against, and the
;; acceptance taken on question B.
;;
;; Run:  make deps && emacs -Q --batch -L build/elc -L lisp -l tools/score-calibrate.el

(defvar nso-cal--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(defun nso-cal--sib (name) (expand-file-name (concat "../../" name) nso-cal--here))
(defun nso-cal--own (name) (expand-file-name (concat "../" name) nso-cal--here))

(add-to-list 'load-path (nso-cal--own "lisp"))
(add-to-list 'load-path (nso-cal--own "build/elc"))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-score)
(require 'nso-probe)
(require 'nso-stub)

(defvar nso-cal--states
  (or (getenv "NSO_SCORE_STATES") (nso-cal--own "build/score-states.eld")))
(defvar nso-cal--results (nso-cal--own "build/score-calibration.org"))
(defvar nso-cal--steps 6000)
(defvar nso-cal--lr 0.5)
(defvar nso-cal--l2 0.01)
(defvar nso-cal--gnorm-limit 0.05)
(defvar nso-cal--draws 400)
(defvar nso-cal--bins 10)

;; Fixed by the pre-registration, not chosen while reading a result: the
;; residual calibrator for question B is fitted on these held-out scenarios
;; and measured on the rest of the held-out split.
(defvar nso-cal--residual-fit-scenarios '(6 12))

(defun nso-cal--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))

;;; --- the calibrated-by-construction band ---------------------------------

(defun nso-cal--draw (rng n k sharpen)
  "N (logits . label) pairs from a model SHARPEN times overconfident.

Labels come from softmax of the UNSCALED logits, so the reported distribution
is off by exactly the factor named: SHARPEN 1.0 is calibrated by construction
and 0.25 is four times too confident."
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

(defun nso-cal--percentile (sorted q)
  (nth (min (1- (length sorted)) (max 0 (floor (* q (length sorted))))) sorted))

(defun nso-cal--band (nfit neval k sharpen draws seed)
  "The 5th and 95th percentile of the recalibration gain at these split sizes."
  (let ((rng (nso-rng seed)) (gains nil))
    (dotimes (_ draws)
      (let* ((fit (nso-cal--draw rng nfit k sharpen))
             (ev (nso-cal--draw rng neval k sharpen))
             (temp (plist-get (nso-score-nominal-temperature-fit (car fit) (cdr fit))
                              :temperature)))
        (push (- (nso-score-nominal-temperature-nll (car ev) (cdr ev) 1.0)
                 (nso-score-nominal-temperature-nll (car ev) (cdr ev) temp))
              gains)))
    (setq gains (sort gains #'<))
    (list :p05 (nso-cal--percentile gains 0.05)
          :p95 (nso-cal--percentile gains 0.95))))

;;; --- the run --------------------------------------------------------------

(let* ((saved (with-temp-buffer (insert-file-contents nso-cal--states)
                                (read (buffer-string))))
       (rows (plist-get saved :rows))
       (pool (lambda (r) (nso-pool-last (plist-get r :mid))))
       (train nil) (test nil))
  (dolist (r rows)
    (if (= 0 (mod (plist-get r :scenario) 3)) (push r test) (push r train)))
  (setq train (nreverse train) test (nreverse test))
  (let* ((k 5)
         (try (mapcar (lambda (r) (plist-get r :level)) train))
         (tey (mapcar (lambda (r) (plist-get r :level)) test))
         (trx (mapcar pool train))
         (tex (mapcar pool test))
         (std (nso-standardizer trx))
         (head (nso-score-nominal-train
                (mapcar (lambda (v) (nso-standardize std v)) trx)
                try k nso-cal--steps nso-cal--lr nso-cal--l2))
         (te-logits (mapcar (lambda (v) (nso-score-nominal-logits
                                         head (nso-standardize std v)))
                            tex)))
    (nso-cal--say "Calibration of the retained head -- %d train / %d held-out, %d levels\n"
                  (length train) (length test) k)

    ;; --- gate 0 ----------------------------------------------------------
    (let ((ok (< (plist-get head :final-gnorm) nso-cal--gnorm-limit)))
      (nso-cal--say "GATE 0 convergence: |g| %.2e in %d iterations -- %s"
                    (plist-get head :final-gnorm) (plist-get head :steps-taken)
                    (if ok "PASS" "VOID"))
      (unless ok (error "score-calibrate: the head did not converge; nothing below is read")))

    (let* ((oof (nso-score-nominal-oof-logits
                 train try (mapcar (lambda (r) (plist-get r :scenario)) train)
                 (lambda (_a _b) pool) k 3
                 nso-cal--steps nso-cal--lr nso-cal--l2))
           (tfit (nso-score-nominal-temperature-fit oof try))
           (temp (plist-get tfit :temperature))
           (corrected (nso-score-nominal-scale te-logits temp))
           ;; --- question A -------------------------------------------------
           (gain-a (- (nso-score-nominal-temperature-nll te-logits tey 1.0)
                      (nso-score-nominal-temperature-nll te-logits tey temp)))
           (band-a (nso-cal--band (length train) (length test) k 1.0
                                  nso-cal--draws 13131))
           (over-a (nso-cal--band (length train) (length test) k 0.25
                                  nso-cal--draws 13131))
           (inside-a (and (>= gain-a (plist-get band-a :p05))
                          (<= gain-a (plist-get band-a :p95))))
           (see-a (> (plist-get over-a :p05) (plist-get band-a :p95)))
           ;; --- question B -------------------------------------------------
           ;; The residual calibrator sees only the scenarios the
           ;; pre-registration names; the gain is measured on the others.
           (fit-z nil) (fit-y nil) (ev-z nil) (ev-y nil))
      (let ((rz corrected) (ry tey))
        (dolist (r test)
          (if (memq (plist-get r :scenario) nso-cal--residual-fit-scenarios)
              (progn (push (car rz) fit-z) (push (car ry) fit-y))
            (push (car rz) ev-z) (push (car ry) ev-y))
          (setq rz (cdr rz) ry (cdr ry))))
      (setq fit-z (nreverse fit-z) fit-y (nreverse fit-y)
            ev-z (nreverse ev-z) ev-y (nreverse ev-y))
      (let* ((t2 (plist-get (nso-score-nominal-temperature-fit fit-z fit-y)
                            :temperature))
             (gain-b (- (nso-score-nominal-temperature-nll ev-z ev-y 1.0)
                        (nso-score-nominal-temperature-nll ev-z ev-y t2)))
             (band-b (nso-cal--band (length fit-z) (length ev-z) k 1.0
                                    nso-cal--draws 24242))
             (over-b (nso-cal--band (length fit-z) (length ev-z) k 0.25
                                    nso-cal--draws 24242))
             (inside-b (and (>= gain-b (plist-get band-b :p05))
                            (<= gain-b (plist-get band-b :p95))))
             (see-b (> (plist-get over-b :p05) (plist-get band-b :p95)))
             ;; --- reported ------------------------------------------------
             (probs-raw (mapcar (lambda (z) (nso-softmax-vec z)) te-logits))
             (probs-cor (mapcar (lambda (z) (nso-softmax-vec z)) corrected))
             (s-raw (nso-score-report probs-raw tey))
             (s-cor (nso-score-report probs-cor tey))
             (ece-raw (nso-ece (let ((out nil) (r tey))
                                 (dolist (p probs-raw)
                                   (push (nso-sample (append p nil) (car r)) out)
                                   (setq r (cdr r)))
                                 (nreverse out))
                               nso-cal--bins))
             (ece-cor (nso-ece (let ((out nil) (r tey))
                                 (dolist (p probs-cor)
                                   (push (nso-sample (append p nil) (car r)) out)
                                   (setq r (cdr r)))
                                 (nreverse out))
                               nso-cal--bins)))
        (nso-cal--say "temperature fitted out of fold on train: T = %.3f%s\n"
                      temp (if (plist-get tfit :saturated) "  AT BOUND" ""))

        (nso-cal--say "GATE 1 can the statistic see?")
        (nso-cal--say "  at n=%d/%d: calibrated [%+.4f, %+.4f], 4x overconfident [%+.4f, %+.4f] -- %s"
                      (length train) (length test)
                      (plist-get band-a :p05) (plist-get band-a :p95)
                      (plist-get over-a :p05) (plist-get over-a :p95)
                      (if see-a "separated" "OVERLAPS -- blind"))
        (nso-cal--say "  at n=%d/%d: calibrated [%+.4f, %+.4f], 4x overconfident [%+.4f, %+.4f] -- %s"
                      (length fit-z) (length ev-z)
                      (plist-get band-b :p05) (plist-get band-b :p95)
                      (plist-get over-b :p05) (plist-get over-b :p95)
                      (if see-b "separated" "OVERLAPS -- blind"))
        (nso-cal--say "  -- %s\n" (if (and see-a see-b) "PASS" "FAIL, no verdict is read"))

        (nso-cal--say "A. was the RAW output calibrated?  (reported, not the acceptance)")
        (nso-cal--say "   gain %+.4f   band [%+.4f, %+.4f]   %s\n"
                      gain-a (plist-get band-a :p05) (plist-get band-a :p95)
                      (if inside-a "inside -- it already was"
                        "OUTSIDE -- it was not"))

        (nso-cal--say "B. is the CORRECTED output calibrated?  (the acceptance question)")
        (nso-cal--say "   residual calibrator on scenarios %s (%d), measured on the rest (%d)"
                      nso-cal--residual-fit-scenarios (length fit-z) (length ev-z))
        (nso-cal--say "   residual T = %.3f" t2)
        (nso-cal--say "   gain %+.4f   band [%+.4f, %+.4f]   %s\n"
                      gain-b (plist-get band-b :p05) (plist-get band-b :p95)
                      (if inside-b "inside -- calibrated" "OUTSIDE -- not calibrated"))

        (nso-cal--say "GATE 2 acceptance: %s"
                      (if (and inside-b see-a see-b)
                          "PASS -- the shipped head is calibrated, and the rule could have said otherwise"
                        "FAIL"))
        (nso-cal--say "")
        (nso-cal--say "reported, not gated:")
        (nso-cal--say "  accuracy %.3f -> %.3f across the temperature (must be equal)"
                      (plist-get s-raw :accuracy) (plist-get s-cor :accuracy))
        (nso-cal--say "  MAE %.3f -> %.3f" (plist-get s-raw :mae) (plist-get s-cor :mae))
        (nso-cal--say "  NLL %.4f -> %.4f" (plist-get s-raw :nll) (plist-get s-cor :nll))
        (nso-cal--say "  ECE %.4f -> %.4f at %d bins, n=%d (NOT a gate; see P3)"
                      (plist-get ece-raw :ece) (plist-get ece-cor :ece)
                      (plist-get ece-raw :bins) (plist-get ece-raw :n))
        (unless (= (plist-get s-raw :accuracy) (plist-get s-cor :accuracy))
          (nso-cal--say "  WARNING: the temperature moved an answer.  It cannot; the run is wrong."))

        (with-temp-file nso-cal--results
          (insert "#+TITLE: Calibration of the head Score ships\n")
          (insert (format "#+DATE: %s\n\n" (format-time-string "%Y-%m-%d %H:%M")))
          (insert (format "Nominal softmax head, %d train / %d held-out, split by scenario.\n"
                          (length train) (length test)))
          (insert (format "Temperature fitted out of fold on train: T = %.3f.\n\n" temp))
          (insert "| question | gain | band | inside? |\n")
          (insert "|----------+------+------+---------|\n")
          (insert (format "| A: raw output calibrated? | %+.4f | [%+.4f, %+.4f] | %s |\n"
                          gain-a (plist-get band-a :p05) (plist-get band-a :p95)
                          (if inside-a "yes" "no")))
          (insert (format "| B: corrected output calibrated? | %+.4f | [%+.4f, %+.4f] | %s |\n"
                          gain-b (plist-get band-b :p05) (plist-get band-b :p95)
                          (if inside-b "yes" "no")))
          (insert (format "\nControls, same runs: 4x overconfident [%+.4f, %+.4f] at A, [%+.4f, %+.4f] at B.\n"
                          (plist-get over-a :p05) (plist-get over-a :p95)
                          (plist-get over-b :p05) (plist-get over-b :p95)))
          (insert (format "\n| metric | raw | corrected |\n|--------+-----+-----------|\n"))
          (insert (format "| accuracy | %.3f | %.3f |\n"
                          (plist-get s-raw :accuracy) (plist-get s-cor :accuracy)))
          (insert (format "| MAE | %.3f | %.3f |\n"
                          (plist-get s-raw :mae) (plist-get s-cor :mae)))
          (insert (format "| NLL | %.4f | %.4f |\n"
                          (plist-get s-raw :nll) (plist-get s-cor :nll)))
          (insert (format "| ECE (%d bins, n=%d) | %.4f | %.4f |\n"
                          (plist-get ece-raw :bins) (plist-get ece-raw :n)
                          (plist-get ece-raw :ece) (plist-get ece-cor :ece)))
          (insert (format "\nVERDICT: %s\n"
                          (if (and inside-b see-a see-b) "ACCEPTED" "NOT ACCEPTED"))))
        (nso-cal--say "\nreport written to %s" nso-cal--results)))))

;;; score-calibrate.el ends here
