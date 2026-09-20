;;; p3-calibrate.el --- P3: temperature against vector scaling, on the P1 split -*- lexical-binding: t; -*-

;; Reads the states P1 already encoded, so this needs no GPU and no donor.
;; The thresholds it is judged against were committed before it ran; see the
;; P3 pre-registration in docs/design/01-system-one.org.
;;
;; Run:  make compile && emacs -Q --batch -L build/elc -L lisp -l tools/p3-calibrate.el

(defvar nso-p3--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-p3--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-p3--here))

(require 'nso-probe)
(require 'nso-stub)

(defvar nso-p3--states (expand-file-name "../build/p1-states.eld" nso-p3--here))
(defvar nso-p3--results (expand-file-name "../build/p3-results.org" nso-p3--here))
(defvar nso-p3--bins 10)
(defvar nso-p3--threshold 0.05)

(defun nso-p3--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))

(defun nso-p3--scores (probs labels)
  (nso-probe-score probs labels nso-p3--bins))

(defun nso-p3--line (label s)
  (format "  %-26s acc %.3f  ECE %.4f  NLL %.4f  Brier %.4f  n=%d"
          label (plist-get s :accuracy) (plist-get s :ece)
          (plist-get s :nll) (plist-get s :brier) (plist-get s :n)))

(if (not (file-readable-p nso-p3--states))
    (nso-p3--say "SKIP: no P1 states at %s" nso-p3--states)
  (let* ((rows (plist-get (with-temp-buffer
                            (insert-file-contents nso-p3--states)
                            (read (buffer-string)))
                          :rows))
         (split (let ((tr nil) (te nil))
                  (dolist (r rows)
                    (if (= 0 (mod (plist-get r :pair) 3)) (push r te) (push r tr)))
                  (cons (nreverse tr) (nreverse te))))
         (train (car split)) (test (cdr split))
         (pool (lambda (r) (nso-pool-last (plist-get r :mid))))
         (try (mapcar (lambda (r) (float (plist-get r :label))) train))
         (tey (mapcar (lambda (r) (float (plist-get r :label))) test))
         (featurizer (lambda (_rows _ys) pool))
         (trx (mapcar pool train))
         (tex (mapcar pool test))
         (fit (nso-probe-fit-and-score trx try tex tey 600 0.5 0.05 nso-p3--bins))
         (te-logits (plist-get fit :logits))
         ;; Out of fold, from the training split.  Not held-out (which is what
         ;; is reported on) and not the training logits the head has already
         ;; separated -- the two mistakes P1 made, in that order.
         (oof (nso-probe-oof-logits train try
                                    (mapcar (lambda (r) (plist-get r :pair)) train)
                                    featurizer 3 600 0.5 0.05))
         (tfit (nso-temperature-fit oof try))
         (pfit (nso-platt-fit oof try))
         (temp (plist-get tfit :temperature))
         (pa (plist-get pfit :a)) (pb (plist-get pfit :b))
         (p-raw (mapcar #'nso-sigmoid te-logits))
         (p-temp (mapcar (lambda (z) (nso-sigmoid (/ z temp))) te-logits))
         (p-platt (mapcar (lambda (z) (nso-sigmoid (+ (* pa z) pb))) te-logits))
         (s-raw (nso-p3--scores p-raw tey))
         (s-temp (nso-p3--scores p-temp tey))
         (s-platt (nso-p3--scores p-platt tey))
         (flips (let ((k 0) (r1 p-raw) (r2 p-platt))
                  (while r1
                    (unless (eq (>= (car r1) 0.5) (>= (car r2) 0.5))
                      (setq k (1+ k)))
                    (setq r1 (cdr r1) r2 (cdr r2)))
                  k)))
    (nso-p3--say "P3 calibration -- %d train / %d held-out, %d equal-width bins"
                 (length train) (length test) nso-p3--bins)
    (nso-p3--say "calibrators fitted on out-of-fold training logits (3 folds by pair)")
    (nso-p3--say "  temperature %.3f     Platt a=%.3f b=%.3f" temp pa pb)
    (nso-p3--say "")
    (nso-p3--say "%s" (nso-p3--line "uncorrected" s-raw))
    (nso-p3--say "%s" (nso-p3--line "temperature" s-temp))
    (nso-p3--say "%s" (nso-p3--line "vector (Platt)" s-platt))
    (nso-p3--say "  answers moved by Platt: %d of %d" flips (length tey))
    (nso-p3--say "")

    ;; --- the pre-registered decisions ------------------------------------
    (let* ((gate-temp (< (plist-get s-temp :ece) nso-p3--threshold))
           (gate-platt (< (plist-get s-platt :ece) nso-p3--threshold))
           (ece-gain (- (plist-get s-temp :ece) (plist-get s-platt :ece)))
           (acc-cost (- (plist-get s-temp :accuracy) (plist-get s-platt :accuracy)))
           (adopt (and (>= ece-gain 0.005) (<= acc-cost 0.0))))
      (nso-p3--say "gate (ECE < %.2f): temperature %s, Platt %s"
                   nso-p3--threshold (if gate-temp "PASS" "FAIL")
                   (if gate-platt "PASS" "FAIL"))
      (nso-p3--say "adoption rule: ECE gain %.4f (need >= 0.005), accuracy cost %.4f (need <= 0)"
                   ece-gain acc-cost)
      (nso-p3--say "  -> %s" (if adopt "ADOPT vector scaling" "KEEP temperature"))
      (nso-p3--say "")

      ;; --- honesty clause: per difficulty ----------------------------------
      (let* ((easy (nso-probe-subset p-temp tey test
                                     (lambda (r) (not (plist-get r :hard)))))
             (hard (nso-probe-subset p-temp tey test
                                     (lambda (r) (plist-get r :hard))))
             (s-easy (nso-p3--scores (nth 0 easy) (nth 1 easy)))
             (s-hard (nso-p3--scores (nth 0 hard) (nth 1 hard))))
        (nso-p3--say "after temperature, by difficulty:")
        (nso-p3--say "%s" (nso-p3--line "antonym pairs" s-easy))
        (nso-p3--say "%s" (nso-p3--line "compositional pairs" s-hard))
        (nso-p3--say "")

        ;; --- negative control, same gate, same bins ------------------------
        (let* ((cal (nso-stub-calibrated 20260920 2000 2))
               (over (nso-stub-tempered cal 0.25))
               (g-cal (nso-calibration-gate cal nso-p3--threshold nso-p3--bins))
               (g-over (nso-calibration-gate over nso-p3--threshold nso-p3--bins)))
          (nso-p3--say "negative control, same gate and bins:")
          (nso-p3--say "  calibrated stub    ECE %.4f  %s"
                       (plist-get g-cal :ece)
                       (if (plist-get g-cal :pass) "PASS" "FAIL"))
          (nso-p3--say "  4x overconfident   ECE %.4f  %s"
                       (plist-get g-over :ece)
                       (if (plist-get g-over :pass) "PASS -- GATE IS BROKEN" "red, as required"))
          (nso-p3--say "")
          (nso-p3--say "%s" (nso-format-reliability
                             (nso-ece (nso-probe-samples p-temp tey) nso-p3--bins)
                             "held-out after temperature"))
          ;; secondary, not the gate
          (nso-p3--say "equal-mass ECE (secondary): temperature %.4f, Platt %.4f"
                       (plist-get (nso-ece (nso-probe-samples p-temp tey)
                                           nso-p3--bins 'equal-mass) :ece)
                       (plist-get (nso-ece (nso-probe-samples p-platt tey)
                                           nso-p3--bins 'equal-mass) :ece))

          (with-temp-file nso-p3--results
            (insert "#+TITLE: P3 results -- calibration on the P1 held-out split\n")
            (insert (format "#+DATE: %s\n\n" (format-time-string "%Y-%m-%d %H:%M")))
            (insert (format "%d train / %d held-out, %d equal-width bins.  Calibrators\n"
                            (length train) (length test) nso-p3--bins))
            (insert "fitted on out-of-fold training logits, three folds by pair.\n\n")
            (insert "| calibration | accuracy | ECE | NLL | Brier | answers moved |\n")
            (insert "|-------------+----------+-----+-----+-------+---------------|\n")
            (insert (format "| uncorrected | %.3f | %.4f | %.4f | %.4f | -- |\n"
                            (plist-get s-raw :accuracy) (plist-get s-raw :ece)
                            (plist-get s-raw :nll) (plist-get s-raw :brier)))
            (insert (format "| temperature (T=%.2f) | %.3f | %.4f | %.4f | %.4f | 0 |\n"
                            temp (plist-get s-temp :accuracy) (plist-get s-temp :ece)
                            (plist-get s-temp :nll) (plist-get s-temp :brier)))
            (insert (format "| vector (a=%.2f b=%.2f) | %.3f | %.4f | %.4f | %.4f | %d |\n"
                            pa pb (plist-get s-platt :accuracy) (plist-get s-platt :ece)
                            (plist-get s-platt :nll) (plist-get s-platt :brier) flips))
            (insert (format "\nDecision: %s (ECE gain %.4f, accuracy cost %.4f)\n"
                            (if adopt "adopt vector scaling" "keep temperature")
                            ece-gain acc-cost))
            (insert "\n| subset after temperature | accuracy | ECE | n |\n")
            (insert "|--------------------------+----------+-----+---|\n")
            (insert (format "| antonym pairs | %.3f | %.4f | %d |\n"
                            (plist-get s-easy :accuracy) (plist-get s-easy :ece)
                            (plist-get s-easy :n)))
            (insert (format "| compositional pairs | %.3f | %.4f | %d |\n"
                            (plist-get s-hard :accuracy) (plist-get s-hard :ece)
                            (plist-get s-hard :n)))
            (insert (format "\nNegative control: 4x-overconfident stub ECE %.4f, %s\n"
                            (plist-get g-over :ece)
                            (if (plist-get g-over :pass) "PASSED (gate broken)" "red")))))))))

;;; p3-calibrate.el ends here
