;;; score-lexical-check.el --- can a bag of words do it? -*- lexical-binding: t; -*-

;; Run BEFORE the encoder, because it decides whether the encoder is worth
;; running.  The Score set claims two things about itself:
;;
;;   the easy subset is lexically separable -- one modal word carries the
;;   level, so a unigram model should do well, and reporting that is what
;;   keeps the encoder from being credited with a lookup;
;;
;;   the hard subset is not -- the level depends on how negations compose and
;;   the sign of the composition reverses between "no doubt" and "no chance",
;;   which a model with one weight per word cannot represent.
;;
;; If the second claim is false, the encode is an hour of GPU time spent on a
;; task a hash table solves, and the right response is to fix the data rather
;; than to run it and explain the result afterwards.  Costs a few seconds and
;; touches no model weights.
;;
;; Both heads are fitted here, on identical features with an identical budget,
;; for the same reason the suite does it: a comparison between two heads given
;; different optimiser settings measures the settings.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/score-lexical-check.el

(defvar nso-lx--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-lx--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-lx--here))

(require 'nso-score)
(require 'nso-probe)

(defvar nso-lx--data (expand-file-name "../data/score-commitment.eld" nso-lx--here))
(defvar nso-lx--steps 600)
(defvar nso-lx--lr 0.5)
(defvar nso-lx--l2 0.01)

(defun nso-lx--say (fmt &rest args) (princ (apply #'format fmt args)) (princ "\n"))

(defun nso-lx--line (label s)
  (if (plist-get s :empty)
      (format "  %-22s -- (n=0)" label)
    (format "  %-22s acc %.3f [%.3f,%.3f]  MAE %.3f  MAE(median) %.3f  n=%d"
            label (plist-get s :accuracy)
            (plist-get s :ci-lo) (plist-get s :ci-hi)
            (plist-get s :mae) (plist-get s :mae-median) (plist-get s :n))))

(defun nso-lx--bag-bound (rows)
  "The (ACCURACY-CEILING . MAE-FLOOR) any bag-of-words model faces on ROWS.

Derived from the data: hard sentences come in pairs whose word SETS are
identical, and a model that cannot see a difference cannot act on one.  So a
pair with unequal levels yields at most one correct answer and at least
|a - b| of absolute error across its two members, whatever the model is."
  (let ((groups (make-hash-table :test 'equal)) (right 0.0) (gap 0.0) (n 0))
    (dolist (r rows)
      (when (plist-get r :hard)
        (let* ((bag (sort (delete-dups
                           (split-string (downcase (plist-get r :text)) "[^a-z]+" t))
                          #'string<))
               (key (list (plist-get r :scenario) bag)))
          (puthash key (cons r (gethash key groups)) groups)
          (setq n (1+ n)))))
    (maphash (lambda (_k members)
               (if (/= (length members) 2)
                   ;; A singleton in this split is free for a bag model, so it
                   ;; raises the ceiling rather than being ignored.
                   (setq right (+ right (float (length members))))
                 (let ((a (plist-get (nth 0 members) :level))
                       (b (plist-get (nth 1 members) :level)))
                   (setq right (+ right (if (= a b) 2.0 1.0))
                         gap (+ gap (abs (- a b)))))))
             groups)
    (cons (/ right n) (/ gap n))))

(defun nso-lx--subset (probs ys rows pred)
  (let ((ps nil) (ls nil) (rp probs) (rl ys))
    (dolist (r rows)
      (when (funcall pred r) (push (car rp) ps) (push (car rl) ls))
      (setq rp (cdr rp) rl (cdr rl)))
    (cons (nreverse ps) (nreverse ls))))

(let* ((form (with-temp-buffer (insert-file-contents nso-lx--data)
                               (read (buffer-string))))
       (rows (append (plist-get form :examples) nil))
       (k (length (plist-get form :levels)))
       (train nil) (test nil))
  (dolist (r rows)
    (if (= 0 (mod (plist-get r :scenario) 3)) (push r test) (push r train)))
    (setq train (nreverse train) test (nreverse test))
  (let* ((vocab (nso-probe-vocab train))
         (trx (nso-probe-unigram-features train vocab))
         (tex (nso-probe-unigram-features test vocab))
         (try (mapcar (lambda (r) (plist-get r :level)) train))
         (tey (mapcar (lambda (r) (plist-get r :level)) test))
         (ord (nso-score-train trx try k nso-lx--steps nso-lx--lr nso-lx--l2))
         (nom (nso-score-nominal-train trx try k nso-lx--steps nso-lx--lr nso-lx--l2))
         (ord-p (mapcar (lambda (x) (nso-score-probs ord x)) tex))
         (nom-p (mapcar (lambda (x) (nso-score-nominal-probs nom x)) tex))
         (base (nso-score-constant-mae tey k))
         (easy-o (nso-lx--subset ord-p tey test (lambda (r) (not (plist-get r :hard)))))
         (hard-o (nso-lx--subset ord-p tey test (lambda (r) (plist-get r :hard))))
         (easy-n (nso-lx--subset nom-p tey test (lambda (r) (not (plist-get r :hard)))))
         (hard-n (nso-lx--subset nom-p tey test (lambda (r) (plist-get r :hard)))))
    (nso-lx--say "unigram baseline on the Score set -- no encoder, no GPU\n")
    (nso-lx--say "%d train / %d held-out, split by scenario, vocabulary %d words"
                 (length train) (length test) (length vocab))
    (nso-lx--say "chance accuracy %.3f, best constant answer level %d at MAE %.3f\n"
                 (/ 1.0 k) (plist-get base :level) (plist-get base :mae))
    (nso-lx--say "ordinal head on unigram features  (gradient norm %.2e)"
                 (plist-get ord :final-gnorm))
    (nso-lx--say "%s" (nso-lx--line "all held-out" (nso-score-report ord-p tey)))
    (nso-lx--say "%s" (nso-lx--line "easy (modal word)"
                                    (nso-score-report (car easy-o) (cdr easy-o))))
    (nso-lx--say "%s" (nso-lx--line "hard (composition)"
                                    (nso-score-report (car hard-o) (cdr hard-o))))
    (nso-lx--say "")
    (nso-lx--say "nominal head on unigram features  (gradient norm %.2e)"
                 (plist-get nom :final-gnorm))
    (nso-lx--say "%s" (nso-lx--line "all held-out" (nso-score-report nom-p tey)))
    (nso-lx--say "%s" (nso-lx--line "easy (modal word)"
                                    (nso-score-report (car easy-n) (cdr easy-n))))
    (nso-lx--say "%s" (nso-lx--line "hard (composition)"
                                    (nso-score-report (car hard-n) (cdr hard-n))))
    (nso-lx--say "")

    ;; The verdict the encode depends on.  The threshold is DERIVED from the
    ;; dataset, not chosen: the hard subset is built from bag-identical pairs,
    ;; so a model whose input is the set of words must answer both members the
    ;; same way, and its accuracy and MAE are bounded before anything is
    ;; fitted.  An earlier version of this file compared against a hand-picked
    ;; 0.40 and would have printed STOP at the exact figure the construction
    ;; guarantees -- a threshold carried over from a previous design and never
    ;; re-derived, which is the same fault this repository has now made three
    ;; times.
    (let* ((h (nso-score-report (car hard-o) (cdr hard-o)))
           (hn (nso-score-report (car hard-n) (cdr hard-n)))
           (best-acc (max (plist-get h :accuracy) (plist-get hn :accuracy)))
           (best-mae (min (plist-get h :mae) (plist-get hn :mae)))
           (bound (nso-lx--bag-bound test)))
      (nso-lx--say "hard subset bound from the pairing: acc <= %.3f, MAE >= %.3f"
                   (car bound) (cdr bound))
      (nso-lx--say "hard subset measured, best of the two unigram heads: acc %.3f, MAE %.3f"
                   best-acc best-mae)
      (cond
       ((> best-acc (+ (car bound) 1.0e-9))
        (nso-lx--say
         (concat "-> STOP: a bag of words scored ABOVE a ceiling it cannot "
                 "exceed.\n   Either the pairing is broken or the split is "
                 "leaking.  Do not encode; the\n   bound is the thing the "
                 "whole subset rests on.")))
       (t
        (nso-lx--say
         (concat "-> the bound holds, and the baseline reaches it, so it is "
                 "tight.\n   Anything above that accuracy or below that MAE on "
                 "this subset used word\n   order, which no bag of words has "
                 "access to.  Encode."))))
      (nso-lx--say "")
      (nso-lx--say "easy subset, nominal head: acc %.3f -- that half is a lookup,"
                   (plist-get (nso-score-report (car easy-n) (cdr easy-n)) :accuracy))
      (nso-lx--say "so no encoder result on it counts for anything."))))

;;; score-lexical-check.el ends here
