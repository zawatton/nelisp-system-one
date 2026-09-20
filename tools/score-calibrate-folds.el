;;; score-calibrate-folds.el --- why the out-of-fold temperature under-corrects -*- lexical-binding: t; -*-

;; POST HOC, and it does not re-judge anything.  The pre-registered verdict is
;; FAIL and stands: question B's residual gain is +0.0253 against a band whose
;; 95th percentile is +0.0201, and at 12000 draws that call is clear of the
;; threshold's own uncertainty.
;;
;; What is left is a mechanism, and there is a clean fingerprint pointing at
;; one.  The temperature fitted OUT OF FOLD ON TRAIN came back at 0.946 --
;; slightly sharpening.  Every temperature fitted on HELD-OUT data came back
;; between 1.10 and 1.62 -- flattening.  Those are opposite directions, so the
;; question is not how much correction the head needs but why the training
;; split says it needs the correction backwards.
;;
;; The hypothesis, stated before it is tested: each out-of-fold head is fitted
;; on about seven of the eleven training scenarios, and the shipped head is
;; fitted on all eleven.  More training data makes a head sharper, so the
;; shipped head is more overconfident on unseen scenarios than the fold heads
;; ever were, and a temperature measured on the fold heads under-corrects it.
;; That is the standard caveat of cross-validated calibration, and if it is
;; what happened here then the FAIL is a property of the PROCEDURE rather than
;; a verdict on the head.
;;
;; The test: refit the out-of-fold temperature with more folds, so each fold
;; head trains on nearly as much as the shipped head.  At leave-one-scenario-
;; out the fold heads see ten of eleven scenarios.  If the hypothesis holds the
;; temperature should rise toward the held-out figures; if it stays near 0.946
;; the hypothesis is wrong and the disagreement is something else.
;;
;; What this deliberately does NOT do: recompute question B under a better
;; temperature and report the result as a verdict.  That would be a second
;; attempt at a gate that has already been taken.  A re-test needs a new
;; pre-registration and more held-out scenarios, and the write-up says so.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/score-calibrate-folds.el

(defvar nso-fd--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-fd--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-fd--here))

(require 'nso-score)
(require 'nso-probe)

(defvar nso-fd--states (expand-file-name "../build/score-states.eld" nso-fd--here))

(defun nso-fd--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))

(let* ((saved (with-temp-buffer (insert-file-contents nso-fd--states)
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
         (groups (mapcar (lambda (r) (plist-get r :scenario)) train))
         (nscen (let ((h (make-hash-table :test 'eql)))
                  (dolist (g groups) (puthash g t h))
                  (hash-table-count h)))
         (std (nso-standardizer (mapcar pool train)))
         (head (nso-score-nominal-train
                (mapcar (lambda (r) (nso-standardize std (funcall pool r))) train)
                try k 6000 0.5 0.01))
         (te-logits (mapcar (lambda (r) (nso-score-nominal-logits
                                         head (nso-standardize std (funcall pool r))))
                            test)))
    (nso-fd--say "Why does the out-of-fold temperature point the wrong way?\n")
    (nso-fd--say "%d training scenarios, %d held-out.  The shipped head sees all %d."
                 nscen (- 5) nscen)
    (nso-fd--say "Held-out temperatures, fitted directly, ran 1.10 to 1.62.\n")
    (nso-fd--say "  folds | scenarios per fold head | out-of-fold T | direction")
    (nso-fd--say "  ------+-------------------------+---------------+----------")
    (dolist (folds (list 2 3 5 nscen))
      (let* ((oof (nso-score-nominal-oof-logits
                   train try groups (lambda (_a _b) pool) k folds 6000 0.5 0.01))
             (fit (nso-score-nominal-temperature-fit oof try))
             (temp (plist-get fit :temperature))
             (per (- nscen (max 1 (/ nscen folds)))))
        (nso-fd--say "  %5d | %23d | %13.3f | %s"
                     folds per temp
                     (cond ((> temp 1.05) "flatten")
                           ((< temp 0.95) "sharpen")
                           (t "neither")))))
    (nso-fd--say "")
    ;; The direct comparison the hypothesis is about: a temperature fitted on
    ;; the held-out logits themselves.  It is not a calibrator anyone could
    ;; ship -- it has seen the data it would be judged on -- and it is here
    ;; only as the number the out-of-fold estimates are trying to approach.
    (let ((direct (plist-get (nso-score-nominal-temperature-fit te-logits tey)
                             :temperature)))
      (nso-fd--say "For reference, a temperature fitted ON the held-out logits: %.3f" direct)
      (nso-fd--say "(not shippable -- it has seen what it would be judged on --")
      (nso-fd--say " but it is the target the out-of-fold estimates are aiming at.)"))))

;;; score-calibrate-folds.el ends here
