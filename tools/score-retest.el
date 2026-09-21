;;; score-retest.el --- the calibration re-test, three ways -*- lexical-binding: t; -*-

;; Section 6's "Calibration re-test", committed before the twenty new
;; scenarios were encoded.  It replaces nothing: the first attempt's FAIL
;; stands on its own terms, and this is a differently-built test of the same
;; question, on a split whose sizes were chosen from stubs rather than by eye.
;;
;; The one structural change: the temperature fits on CALIBRATE -- scenarios
;; the head never trained on and the test never sees -- rather than out of fold
;; on the training split, which is where the first attempt put it and where it
;; turned out to point the opposite way from what held-out scenarios needed.
;;
;; The one procedural change: the verdict is three-valued.  A gain that lies
;; inside the bootstrap interval of the percentile deciding it is
;; INDETERMINATE, because the call would then belong to the simulation.  The
;; first attempt read a FAIL off a threshold whose own uncertainty straddled
;; the observed value, and only found out afterwards.
;;
;; Run:  make deps && emacs -Q --batch -L build/elc -L lisp -l tools/score-retest.el

(defvar nso-rt--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))

(defun nso-rt--own (name) (expand-file-name (concat "../" name) nso-rt--here))

(add-to-list 'load-path (nso-rt--own "lisp"))
(add-to-list 'load-path (nso-rt--own "build/elc"))

(require 'nso-score)
(require 'nso-probe)
(require 'nso-stub)

(defvar nso-rt--states
  (or (getenv "NSO_SCORE_STATES") (nso-rt--own "build/score-states.eld")))
(defvar nso-rt--results (nso-rt--own "build/score-retest.org"))
(defvar nso-rt--k 5)
(defvar nso-rt--steps 6000)
(defvar nso-rt--lr 0.5)
(defvar nso-rt--l2 0.01)
(defvar nso-rt--gnorm-limit 0.05)
(defvar nso-rt--draws (string-to-number (or (getenv "NSO_RETEST_DRAWS") "12000")))
(defvar nso-rt--boots 2000)
(defvar nso-rt--bins 10)
(defvar nso-rt--bag-acc-ceiling 0.600)
(defvar nso-rt--bag-mae-floor 1.200)

(defun nso-rt--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))

(defun nso-rt--pct (sorted q)
  (nth (min (1- (length sorted)) (max 0 (floor (* q (length sorted))))) sorted))

;;; --- the split, exactly as pre-registered --------------------------------

(defun nso-rt--assign (scenarios)
  "Return (TEST CALIBRATE TRAIN) as lists of scenario ids.

Computed from the rule rather than listed from it, so the code and the
document cannot drift apart: test is every id divisible by three; calibrate is
every third of what remains, in id order; train is the rest."
  (let ((test nil) (rest nil))
    (dolist (s scenarios)
      (if (= 0 (mod s 3)) (push s test) (push s rest)))
    (setq test (nreverse test) rest (nreverse rest))
    (let ((cal nil) (train nil) (i 0))
      (dolist (s rest)
        (if (= 2 (mod i 3)) (push s cal) (push s train))
        (setq i (1+ i)))
      (list test (nreverse cal) (nreverse train)))))

;;; --- the band ------------------------------------------------------------

(defun nso-rt--draw (rng n k sharpen)
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

(defun nso-rt--gains (nfit neval sharpen draws seed)
  (let ((rng (nso-rng seed)) (out nil))
    (dotimes (_ draws)
      (let* ((fit (nso-rt--draw rng nfit nso-rt--k sharpen))
             (ev (nso-rt--draw rng neval nso-rt--k sharpen))
             (temp (plist-get (nso-score-nominal-temperature-fit (car fit) (cdr fit))
                              :temperature)))
        (push (- (nso-score-nominal-temperature-nll (car ev) (cdr ev) 1.0)
                 (nso-score-nominal-temperature-nll (car ev) (cdr ev) temp))
              out)))
    (sort out #'<)))

(defun nso-rt--pct-ci (gains q boots seed)
  (let* ((v (vconcat gains)) (n (length v)) (rng (nso-rng seed)) (ps nil))
    (dotimes (_ boots)
      (let ((s (make-vector n 0.0)))
        (dotimes (i n) (aset s i (aref v (mod (nso-rng-next rng) n))))
        (push (nso-rt--pct (sort (append s nil) #'<) q) ps)))
    (setq ps (sort ps #'<))
    (cons (nso-rt--pct ps 0.025) (nso-rt--pct ps 0.975))))

(defun nso-rt--verdict (gain p05 p95 ci05 ci95)
  "PASS, FAIL or INDETERMINATE, by the three-valued pre-registered rule."
  (cond
   ;; The percentile that decides the call is the one the gain is nearest to
   ;; crossing; if the gain lies inside THAT percentile's own interval, the
   ;; call belongs to the simulation and neither word is said.
   ((and (>= gain (car ci95)) (<= gain (cdr ci95))) 'indeterminate)
   ((and (>= gain (car ci05)) (<= gain (cdr ci05))) 'indeterminate)
   ((and (>= gain p05) (<= gain p95)) 'pass)
   (t 'fail)))

;;; --- the run --------------------------------------------------------------

(let* ((saved (with-temp-buffer (insert-file-contents nso-rt--states)
                                (read (buffer-string))))
       (rows (plist-get saved :rows))
       (pool (lambda (r) (nso-pool-last (plist-get r :mid))))
       (scenarios (let ((seen nil))
                    (dolist (r rows)
                      (unless (memq (plist-get r :scenario) seen)
                        (push (plist-get r :scenario) seen)))
                    (sort seen #'<)))
       (parts (nso-rt--assign scenarios))
       (test-ids (nth 0 parts)) (cal-ids (nth 1 parts)) (train-ids (nth 2 parts))
       (pick (lambda (ids) (let (out)
                             (dolist (r rows)
                               (when (memq (plist-get r :scenario) ids) (push r out)))
                             (nreverse out))))
       (train (funcall pick train-ids))
       (cal (funcall pick cal-ids))
       (test (funcall pick test-ids)))
  (when (< (length scenarios) 36)
    (error "score-retest: only %d scenarios encoded; the pre-registered split needs 36"
           (length scenarios)))
  (nso-rt--say "Calibration re-test -- three-way split by scenario\n")
  (nso-rt--say "  train     %2d scenarios, %3d examples  %s"
               (length train-ids) (length train) train-ids)
  (nso-rt--say "  calibrate %2d scenarios, %3d examples  %s"
               (length cal-ids) (length cal) cal-ids)
  (nso-rt--say "  test      %2d scenarios, %3d examples  %s\n"
               (length test-ids) (length test) test-ids)

  (let* ((k nso-rt--k)
         (try (mapcar (lambda (r) (plist-get r :level)) train))
         (cay (mapcar (lambda (r) (plist-get r :level)) cal))
         (tey (mapcar (lambda (r) (plist-get r :level)) test))
         (std (nso-standardizer (mapcar pool train)))
         (feat (lambda (r) (nso-standardize std (funcall pool r))))
         (head (nso-score-nominal-train
                (mapcar feat train) try k nso-rt--steps nso-rt--lr nso-rt--l2))
         (logits (lambda (rs) (mapcar (lambda (r)
                                        (nso-score-nominal-logits head (funcall feat r)))
                                      rs)))
         (cal-z (funcall logits cal))
         (te-z (funcall logits test)))

    ;; --- gate 0 ---------------------------------------------------------
    (let ((ok (< (plist-get head :final-gnorm) nso-rt--gnorm-limit)))
      (nso-rt--say "GATE 0 convergence: |g| %.2e in %d iterations -- %s"
                   (plist-get head :final-gnorm) (plist-get head :steps-taken)
                   (if ok "PASS" "VOID"))
      (unless ok (error "score-retest: the head did not converge")))

    (let* ((tfit (nso-score-nominal-temperature-fit cal-z cay))
           (temp (plist-get tfit :temperature))
           (gain (- (nso-score-nominal-temperature-nll te-z tey 1.0)
                    (nso-score-nominal-temperature-nll te-z tey temp)))
           (band (nso-rt--gains (length cal) (length test) 1.0 nso-rt--draws 77001))
           (over (nso-rt--gains (length cal) (length test) 0.25
                                (/ nso-rt--draws 4) 77002))
           (p05 (nso-rt--pct band 0.05))
           (p95 (nso-rt--pct band 0.95))
           (ci05 (nso-rt--pct-ci band 0.05 nso-rt--boots 77003))
           (ci95 (nso-rt--pct-ci band 0.95 nso-rt--boots 77004))
           (see (> (nso-rt--pct over 0.05) p95))
           (verdict (nso-rt--verdict gain p05 p95 ci05 ci95)))

      (nso-rt--say "\nGATE 1 can the statistic see?")
      (nso-rt--say "  calibrated [%+.4f, %+.4f], 4x overconfident [%+.4f, %+.4f] -- %s"
                   p05 p95 (nso-rt--pct over 0.05) (nso-rt--pct over 0.95)
                   (if see "PASS, separated" "FAIL, overlaps"))

      (nso-rt--say "\nGATE 2 acceptance -- temperature fitted on CALIBRATE, gain measured on TEST")
      (nso-rt--say "  temperature   T = %.3f%s"
                   temp (if (plist-get tfit :saturated) "  AT BOUND" ""))
      (nso-rt--say "  gain on test  %+.4f" gain)
      (nso-rt--say "  band          [%+.4f, %+.4f]   (%d draws)" p05 p95 nso-rt--draws)
      (nso-rt--say "  5th  pct CI   [%+.4f, %+.4f]" (car ci05) (cdr ci05))
      (nso-rt--say "  95th pct CI   [%+.4f, %+.4f]" (car ci95) (cdr ci95))
      (nso-rt--say "  VERDICT       %s"
                   (cond ((eq verdict 'pass) "PASS -- the shipped head is calibrated")
                         ((eq verdict 'fail) "FAIL -- it is not")
                         (t (concat "INDETERMINATE -- the gain lies inside the deciding "
                                    "percentile's own\n                interval, so the call "
                                    "would be the simulation's, not the head's.\n                "
                                    "Foreseen: see the declared power in section 6."))))

      ;; --- reported, not gated -------------------------------------------
      (let* ((oof (nso-score-nominal-oof-logits
                   train try (mapcar (lambda (r) (plist-get r :scenario)) train)
                   (lambda (_a _b) pool) k 3 nso-rt--steps nso-rt--lr nso-rt--l2))
             (t-oof (plist-get (nso-score-nominal-temperature-fit oof try) :temperature))
             (t-direct (plist-get (nso-score-nominal-temperature-fit te-z tey)
                                  :temperature))
             (p-raw (mapcar #'nso-softmax-vec te-z))
             (p-cor (mapcar #'nso-softmax-vec (nso-score-nominal-scale te-z temp)))
             (s-raw (nso-score-report p-raw tey))
             (s-cor (nso-score-report p-cor tey))
             (samples (lambda (ps) (let ((out nil) (r tey))
                                     (dolist (p ps)
                                       (push (nso-sample (append p nil) (car r)) out)
                                       (setq r (cdr r)))
                                     (nreverse out))))
             (hard (let ((hp nil) (hl nil) (rp p-raw) (rl tey))
                     (dolist (r test)
                       (when (plist-get r :hard) (push (car rp) hp) (push (car rl) hl))
                       (setq rp (cdr rp) rl (cdr rl)))
                     (cons (nreverse hp) (nreverse hl))))
             (h-s (nso-score-report (car hard) (cdr hard))))
        (nso-rt--say "\nreported, not gated:")
        (nso-rt--say "  temperatures: calibrate %.3f | out-of-fold on train %.3f | direct on test %.3f"
                     temp t-oof t-direct)
        (nso-rt--say "    (the first attempt's finding was that the middle one points the wrong")
        (nso-rt--say "     way; whether the first agrees with the third is what says the change worked)")
        (nso-rt--say "  accuracy %.3f -> %.3f across the temperature (must be equal)"
                     (plist-get s-raw :accuracy) (plist-get s-cor :accuracy))
        (nso-rt--say "  MAE %.3f -> %.3f, NLL %.4f -> %.4f"
                     (plist-get s-raw :mae) (plist-get s-cor :mae)
                     (plist-get s-raw :nll) (plist-get s-cor :nll))
        (nso-rt--say "  ECE %.4f -> %.4f at %d bins, n=%d (NOT a gate)"
                     (plist-get (nso-ece (funcall samples p-raw) nso-rt--bins) :ece)
                     (plist-get (nso-ece (funcall samples p-cor) nso-rt--bins) :ece)
                     nso-rt--bins (length tey))
        (nso-rt--say "  Score re-measured on the enlarged test set:")
        (nso-rt--say "    all        acc %.3f [%.3f,%.3f]  MAE %.3f  n=%d"
                     (plist-get s-raw :accuracy) (plist-get s-raw :ci-lo)
                     (plist-get s-raw :ci-hi) (plist-get s-raw :mae) (plist-get s-raw :n))
        (nso-rt--say "    hard       acc %.3f [%.3f,%.3f]  MAE %.3f  n=%d  (bag bound %.3f / %.3f)"
                     (plist-get h-s :accuracy) (plist-get h-s :ci-lo)
                     (plist-get h-s :ci-hi) (plist-get h-s :mae) (plist-get h-s :n)
                     nso-rt--bag-acc-ceiling nso-rt--bag-mae-floor)

        (with-temp-file nso-rt--results
          (insert "#+TITLE: Calibration re-test -- three-way scenario split\n")
          (insert (format "#+DATE: %s\n\n" (format-time-string "%Y-%m-%d %H:%M")))
          (insert (format "train %d / calibrate %d / test %d examples, disjoint by scenario.\n"
                          (length train) (length cal) (length test)))
          (insert "Temperature fitted on calibrate; gain measured on test.\n\n")
          (insert "| quantity | value |\n|----------+-------|\n")
          (insert (format "| temperature on calibrate | %.3f |\n" temp))
          (insert (format "| gain on test | %+.4f |\n" gain))
          (insert (format "| band | [%+.4f, %+.4f] |\n" p05 p95))
          (insert (format "| 95th percentile CI | [%+.4f, %+.4f] |\n" (car ci95) (cdr ci95)))
          (insert (format "| verdict | %s |\n" (upcase (symbol-name verdict))))
          (insert (format "\n| temperature | fitted on | value |\n|---+---+---|\n"))
          (insert (format "| pre-registered | calibrate | %.3f |\n" temp))
          (insert (format "| first attempt's | out of fold on train | %.3f |\n" t-oof))
          (insert (format "| unshippable reference | test itself | %.3f |\n" t-direct))
          (insert (format "\n| metric | raw | corrected |\n|---+---+---|\n"))
          (insert (format "| accuracy | %.3f | %.3f |\n"
                          (plist-get s-raw :accuracy) (plist-get s-cor :accuracy)))
          (insert (format "| MAE | %.3f | %.3f |\n"
                          (plist-get s-raw :mae) (plist-get s-cor :mae)))
          (insert (format "| NLL | %.4f | %.4f |\n"
                          (plist-get s-raw :nll) (plist-get s-cor :nll)))
          (insert (format "\nScore on the enlarged test set: %.3f overall, %.3f on the hard subset (n=%d),\n"
                          (plist-get s-raw :accuracy) (plist-get h-s :accuracy)
                          (plist-get h-s :n)))
          (insert (format "against a bag-of-words ceiling of %.3f.\n" nso-rt--bag-acc-ceiling)))
        (nso-rt--say "\nreport written to %s" nso-rt--results)))))

;;; score-retest.el ends here
