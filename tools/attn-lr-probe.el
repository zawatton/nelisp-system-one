;;; attn-lr-probe.el --- why the learned pool fails at mid depth -*- lexical-binding: t; -*-

;; The scale hypothesis was refuted by `tools/attn-scale-probe.el': the
;; softmax is nearly uniform at both depths (peak 0.135 against a uniform
;; 0.115), so it never saturates, and |du| is exactly zero at initialisation
;; at BOTH depths -- a consequence of the head being zero-initialised, since
;; da_i = g (w . h_i) and w starts at zero -- which cannot explain a
;; difference between them.
;;
;; That measurement pointed at a different asymmetry, and this file tests it.
;; `last' and `mean' hand their pooled features to `nso-probe-fit-and-score',
;; which STANDARDISES them before fitting: the head sees unit-variance
;; columns.  `attn' trains through `nso-attn-train' on the RAW states, whose
;; RMS is 2.98 at final depth and 3.73 at mid.  One learning rate, 0.5, is
;; used for both paths.  If 0.5 is already at the edge of stability on raw
;; features, mid depth is the side that falls off.
;;
;; Prediction: at mid depth the joint model diverges at lr 0.5 and fits at a
;; smaller one; at final depth 0.5 is survivable.  If mid depth fails at every
;; learning rate, this is wrong too and the cause is elsewhere.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/attn-lr-probe.el

(defvar nso-lr--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-lr--here))

(require 'nso-probe)

(defvar nso-lr--states
  (or (getenv "NSO_P1_STATES")
      (expand-file-name "../build/p1-states.eld" nso-lr--here)))

(defvar nso-lr--steps (string-to-number (or (getenv "NSO_LR_STEPS") "400")))

(defun nso-lr--train-acc (model xss ys)
  (let ((ok 0) (n 0) (rest ys))
    (dolist (states xss)
      (let ((p (plist-get (nso-attn-forward model states) :p)))
        (when (eq (>= p 0.5) (= 1.0 (car rest))) (setq ok (1+ ok))))
      (setq n (1+ n) rest (cdr rest)))
    (/ (float ok) n)))

(if (not (file-readable-p nso-lr--states))
    (princ (format "SKIP: no states at %s\n" nso-lr--states))
  (let* ((saved (with-temp-buffer
                  (insert-file-contents nso-lr--states)
                  (read (buffer-string))))
         (rows (plist-get saved :rows))
         ;; Training split only, as the probe uses.
         (train (let (out) (dolist (r rows)
                             (unless (= 0 (mod (plist-get r :pair) 3)) (push r out)))
                     (nreverse out)))
         (ys (mapcar (lambda (r) (float (plist-get r :label))) train)))
    (princ (format "%d training rows, %d steps\n\n" (length train) nso-lr--steps))
    (princ "  depth    lr        train acc   loss\n")
    (princ "  ------------------------------------\n")
    (dolist (depth '(:final :mid))
      (let ((xss (mapcar (lambda (r) (plist-get r depth)) train)))
        (dolist (lr '(0.5 0.1 0.02 0.005))
          (let* ((model (nso-attn-train xss ys nso-lr--steps lr 0.05))
                 (acc (nso-lr--train-acc model xss ys))
                 (loss (nso-attn-loss model xss ys 0.05)))
            (princ (format "  %-8s %-9s %9.3f  %9.4g%s\n"
                           (substring (symbol-name depth) 1) lr acc loss
                           (if (/= loss loss) "  [NaN]" "")))))))
    (princ "\n  A row at 0.500 that recovers at a smaller lr is a step-size\n")
    (princ "  failure, not a representational one.\n")))

;;; attn-lr-probe.el ends here
