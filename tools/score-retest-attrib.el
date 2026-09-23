;;; score-retest-attrib.el --- the split, or just more data? -*- lexical-binding: t; -*-

;; POST HOC, and it changes no verdict.  The re-test passed: a temperature
;; fitted on eight calibrate scenarios left a gain of -0.0006 on test, inside a
;; band of [-0.0222, +0.0049] and clear of both percentiles' intervals.
;;
;; The problem is that the re-test changed TWO things at once.  The first
;; attempt fitted the calibrator out of fold on train and had 165 training
;; examples; this one fits it on a held-back calibrate split AND has 360.  A
;; PASS cannot say which of those did the work, and the temperatures hint that
;; it may not have been the one the pre-registration argued for:
;;
;;   first attempt   out-of-fold on train 0.946   direct on held-out 1.313
;;   re-test         out-of-fold on train 0.989   direct on test     0.867
;;                   calibrate split      0.737
;;
;; In the first attempt those two pointed opposite ways, which is what the
;; three-way split was designed to fix.  Here they point the same way before
;; the split is involved at all.  That is consistent with the head simply
;; being better fitted at 360 examples -- held-out accuracy went 0.867 to 0.950
;; -- and the miscalibration having been a symptom of the smaller fit.
;;
;; So this runs the FIRST attempt's procedure on the NEW data: temperature
;; fitted out of fold on train, gain measured on test, against a band at those
;; sizes.  If it passes too, the fix was data and the structural change was not
;; what mattered, and the write-up has to say so.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/score-retest-attrib.el

(defvar nso-at--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-at--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-at--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-score)
(require 'nso-probe)
(require 'nso-stub)

(defvar nso-at--states (expand-file-name "../build/score-states.eld" nso-at--here))
(defvar nso-at--k 5)
(defvar nso-at--draws (string-to-number (or (getenv "NSO_ATTRIB_DRAWS") "8000")))
(defvar nso-at--boots 1500)

(defun nso-at--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))
(defun nso-at--pct (s q)
  (nth (min (1- (length s)) (max 0 (floor (* q (length s))))) s))

(defun nso-at--draw (rng n k sharpen)
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

(defun nso-at--band (nfit neval sharpen draws seed)
  (let ((rng (nso-rng seed)) (out nil))
    (dotimes (_ draws)
      (let* ((fit (nso-at--draw rng nfit nso-at--k sharpen))
             (ev (nso-at--draw rng neval nso-at--k sharpen))
             (temp (plist-get (nso-score-nominal-temperature-fit (car fit) (cdr fit))
                              :temperature)))
        (push (- (nso-score-nominal-temperature-nll (car ev) (cdr ev) 1.0)
                 (nso-score-nominal-temperature-nll (car ev) (cdr ev) temp))
              out)))
    (sort out #'<)))

(defun nso-at--pct-ci (gains q boots seed)
  (let* ((v (vconcat gains)) (n (length v)) (rng (nso-rng seed)) (ps nil))
    (dotimes (_ boots)
      (let ((s (make-vector n 0.0)))
        (dotimes (i n) (aset s i (aref v (mod (nso-rng-next rng) n))))
        (push (nso-at--pct (sort (append s nil) #'<) q) ps)))
    (setq ps (sort ps #'<))
    (cons (nso-at--pct ps 0.025) (nso-at--pct ps 0.975))))

(let* ((saved (with-temp-buffer (insert-file-contents nso-at--states)
                                (read (buffer-string))))
       (rows (plist-get saved :rows))
       (pool (lambda (r) (nso-pool-last (plist-get r :mid))))
       (test nil) (rest nil))
  ;; The same three-way assignment the re-test uses, so train and test mean
  ;; the same thing here as there.  Calibrate is folded back into train,
  ;; because the first attempt's procedure never had a calibrate split -- that
  ;; is the whole point of the comparison.
  (dolist (r rows)
    (if (= 0 (mod (plist-get r :scenario) 3)) (push r test) (push r rest)))
  (setq test (nreverse test) rest (nreverse rest))
  (let* ((k nso-at--k)
         (train rest)
         (try (mapcar (lambda (r) (plist-get r :level)) train))
         (tey (mapcar (lambda (r) (plist-get r :level)) test))
         (std (nso-standardizer (mapcar pool train)))
         (feat (lambda (r) (nso-standardize std (funcall pool r))))
         (head (nso-score-nominal-train (mapcar feat train) try k 6000 0.5 0.01))
         (te-z (mapcar (lambda (r) (nso-score-nominal-logits head (funcall feat r))) test))
         (oof (nso-score-nominal-oof-logits
               train try (mapcar (lambda (r) (plist-get r :scenario)) train)
               (lambda (_a _b) pool) k 3 6000 0.5 0.01))
         (temp (plist-get (nso-score-nominal-temperature-fit oof try) :temperature))
         (gain (- (nso-score-nominal-temperature-nll te-z tey 1.0)
                  (nso-score-nominal-temperature-nll te-z tey temp)))
         (band (nso-at--band (length train) (length test) 1.0 nso-at--draws 91001))
         (over (nso-at--band (length train) (length test) 0.25
                             (/ nso-at--draws 4) 91002))
         (p05 (nso-at--pct band 0.05))
         (p95 (nso-at--pct band 0.95))
         (ci95 (nso-at--pct-ci band 0.95 nso-at--boots 91003))
         (ci05 (nso-at--pct-ci band 0.05 nso-at--boots 91004))
         (inside (and (>= gain p05) (<= gain p95)))
         (fragile (or (and (>= gain (car ci95)) (<= gain (cdr ci95)))
                      (and (>= gain (car ci05)) (<= gain (cdr ci05))))))
    (nso-at--say "The FIRST attempt's procedure, on the NEW data\n")
    (nso-at--say "  train %d (calibrate folded back in), test %d"
                 (length train) (length test))
    (nso-at--say "  temperature fitted out of fold on train: %.3f" temp)
    (nso-at--say "  gain on test   %+.4f" gain)
    (nso-at--say "  band           [%+.4f, %+.4f]  (%d draws)" p05 p95 nso-at--draws)
    (nso-at--say "  95th pct CI    [%+.4f, %+.4f]" (car ci95) (cdr ci95))
    (nso-at--say "  control 4x     [%+.4f, %+.4f]  %s"
                 (nso-at--pct over 0.05) (nso-at--pct over 0.95)
                 (if (> (nso-at--pct over 0.05) p95) "separated" "OVERLAPS"))
    (nso-at--say "  verdict        %s"
                 (cond (fragile "INDETERMINATE")
                       (inside "PASS")
                       (t "FAIL")))
    (nso-at--say "")
    (nso-at--say "%s"
                 (if (and inside (not fragile))
                     (concat "The old procedure passes on the new data, so the three-way split\n"
                             "is NOT what fixed it -- the training set is.  The re-test's PASS\n"
                             "stands, and the credit for it goes to 360 examples rather than to\n"
                             "where the calibrator was fitted.")
                   (concat "The old procedure still fails on the new data, so the three-way\n"
                           "split is doing the work the pre-registration argued it would.")))))

;;; score-retest-attrib.el ends here
