;;; p3-verdict.el --- P3 judged by the pre-registered rule -*- lexical-binding: t; -*-

;; The rule, as committed: accept when the held-out recalibration gain falls
;; inside the 5-95% band of a calibrated-by-construction stub at that n, with
;; the 4x-overconfident control outside it in the same run.
;;
;; Applying it needs one distinction the rule does not spell out, and the
;; distinction decides the verdict.  There are two questions:
;;
;;   A. Is the head's RAW output calibrated?
;;      The gain of the out-of-fold temperature answers this, and P3 already
;;      measured it: +0.0220.  A positive gain means the recalibrator found
;;      real work to do, so the raw output was not calibrated.
;;
;;   B. Is the output calibrated AFTER the temperature is applied?
;;      This is what "calibration proper" has to mean -- the phase ships a
;;      calibrated model, not a raw one -- and it is the acceptance question.
;;
;; B cannot be answered by refitting a temperature on the same logits the
;; first one was fitted on: the first was the minimiser there, so the second
;; comes back as exactly 1 and the residual gain is zero by construction, not
;; by merit.  It needs a split the first calibrator never saw.
;;
;; So the held-out split is halved by pair: the residual calibrator is fitted
;; on one half and the residual gain is measured on the other.  n falls to 42
;; a side, so the stub band is recomputed at 42 rather than reused from 84 --
;; comparing a statistic at one n against a floor measured at another is the
;; error that sank the ECE gate.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/p3-verdict.el

(defvar nso-pv--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-pv--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-pv--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-probe)
(require 'nso-stub)

(defvar nso-pv--states (expand-file-name "../build/p1-states.eld" nso-pv--here))
(defvar nso-pv--draws 400)

(defun nso-pv--nll (zs ys) (nso-platt-nll zs ys 1.0 0.0))

(defun nso-pv--band (n draws sharpen)
  "5th, 95th percentile and mean of the recalibration gain at N, for a model
that is SHARPEN times overconfident (1.0 = calibrated by construction)."
  (let ((rng (nso-stub-calibrated 1 1 2)) (gains nil) (i 0)
        (r (nso-rng 8801)))
    (ignore rng)
    (while (< i draws)
      (let* ((mk (lambda (m)
                   (let ((zs nil) (ys nil))
                     (dotimes (_ m)
                       (let* ((z (* 4.0 (- (nso-rng-float r) 0.5)))
                              (y (if (< (nso-rng-float r) (nso-sigmoid z)) 1.0 0.0)))
                         (push (* sharpen z) zs) (push y ys)))
                     (cons (nreverse zs) (nreverse ys)))))
             (fit (funcall mk n))
             (ev (funcall mk n))
             (temp (plist-get (nso-temperature-fit (car fit) (cdr fit)) :temperature)))
        (push (- (nso-pv--nll (car ev) (cdr ev))
                 (nso-pv--nll (mapcar (lambda (z) (/ z temp)) (car ev)) (cdr ev)))
              gains))
      (setq i (1+ i)))
    (setq gains (sort gains #'<))
    (let ((m (length gains)) (s 0.0))
      (dolist (g gains) (setq s (+ s g)))
      (list :mean (/ s m)
            :p05 (nth (floor (* 0.05 m)) gains)
            :p95 (nth (min (1- m) (floor (* 0.95 m))) gains)))))

(if (not (file-readable-p nso-pv--states))
    (princ "SKIP: no P1 states\n")
  (let* ((rows (plist-get (with-temp-buffer
                            (insert-file-contents nso-pv--states)
                            (read (buffer-string)))
                          :rows))
         (train nil) (test nil))
    (dolist (r rows)
      (if (= 0 (mod (plist-get r :pair) 3)) (push r test) (push r train)))
    (setq train (nreverse train) test (nreverse test))
    (let* ((pool (lambda (r) (nso-pool-last (plist-get r :mid))))
           (try (mapcar (lambda (r) (float (plist-get r :label))) train))
           (fit (nso-probe-fit-and-score (mapcar pool train) try
                                         (mapcar pool test)
                                         (mapcar (lambda (r) (float (plist-get r :label))) test)
                                         600 0.5 0.05 10))
           (te-logits (plist-get fit :logits))
           (tey (mapcar (lambda (r) (float (plist-get r :label))) test))
           (oof (nso-probe-oof-logits train try
                                      (mapcar (lambda (r) (plist-get r :pair)) train)
                                      (lambda (_a _b) pool) 3 600 0.5 0.05))
           (temp (plist-get (nso-temperature-fit oof try) :temperature))
           (corrected (mapcar (lambda (z) (/ z temp)) te-logits))
           ;; --- question A: was the raw output calibrated? -----------------
           (gain-a (- (nso-pv--nll te-logits tey) (nso-pv--nll corrected tey)))
           (band-84 (nso-pv--band 84 nso-pv--draws 1.0))
           (over-84 (nso-pv--band 84 nso-pv--draws 4.0))
           ;; --- question B: is the corrected output calibrated? ------------
           ;; Halve the held-out split BY PAIR; fit the residual calibrator on
           ;; one half, measure the gain on the other.
           (h1z nil) (h1y nil) (h2z nil) (h2y nil))
      (let ((rz corrected) (ry tey))
        (dolist (r test)
          (if (= 0 (mod (/ (plist-get r :pair) 3) 2))
              (progn (push (car rz) h1z) (push (car ry) h1y))
            (push (car rz) h2z) (push (car ry) h2y))
          (setq rz (cdr rz) ry (cdr ry))))
      (setq h1z (nreverse h1z) h1y (nreverse h1y)
            h2z (nreverse h2z) h2y (nreverse h2y))
      (let* ((t2 (plist-get (nso-temperature-fit h1z h1y) :temperature))
             (gain-b (- (nso-pv--nll h2z h2y)
                        (nso-pv--nll (mapcar (lambda (z) (/ z t2)) h2z) h2y)))
             (band-42 (nso-pv--band (length h2z) nso-pv--draws 1.0))
             (over-42 (nso-pv--band (length h2z) nso-pv--draws 4.0))
             (inside-a (and (>= gain-a (plist-get band-84 :p05))
                            (<= gain-a (plist-get band-84 :p95))))
             (inside-b (and (>= gain-b (plist-get band-42 :p05))
                            (<= gain-b (plist-get band-42 :p95))))
             (control-a (> (plist-get over-84 :p05) (plist-get band-84 :p95)))
             (control-b (> (plist-get over-42 :p05) (plist-get band-42 :p95))))
        (princ "P3, judged by the pre-registered rule\n\n")
        (princ (format "  temperature fitted out of fold on train: T = %.3f\n\n" temp))
        (princ "  A. was the head's RAW output calibrated?\n")
        (princ (format "     gain %+.4f   band at n=84 [%+.4f, %+.4f]   %s\n"
                       gain-a (plist-get band-84 :p05) (plist-get band-84 :p95)
                       (if inside-a "inside -- calibrated" "OUTSIDE -- not calibrated")))
        (princ (format "     control: 4x overconfident [%+.4f, %+.4f]  %s\n\n"
                       (plist-get over-84 :p05) (plist-get over-84 :p95)
                       (if control-a "separated" "OVERLAPS -- rule is blind")))
        (princ "  B. is the CORRECTED output calibrated?  (acceptance question)\n")
        (princ (format "     residual calibrator fitted on %d held-out, measured on %d\n"
                       (length h1z) (length h2z)))
        (princ (format "     gain %+.4f   band at n=%d [%+.4f, %+.4f]   %s\n"
                       gain-b (length h2z)
                       (plist-get band-42 :p05) (plist-get band-42 :p95)
                       (if inside-b "inside -- calibrated" "OUTSIDE -- not calibrated")))
        (princ (format "     control: 4x overconfident [%+.4f, %+.4f]  %s\n\n"
                       (plist-get over-42 :p05) (plist-get over-42 :p95)
                       (if control-b "separated" "OVERLAPS -- rule is blind")))
        (princ (format "  VERDICT: P3 %s\n"
                       (if (and inside-b control-b)
                           "ACCEPTED -- the calibrated model is calibrated, and the rule could have said otherwise"
                         "NOT ACCEPTED")))))))

;;; p3-verdict.el ends here
