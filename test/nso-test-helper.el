;;; nso-test-helper.el --- pass/fail bookkeeping that counts NaN as a failure -*- lexical-binding: t; -*-

;;; Commentary:

;; Loaded by the suites; deliberately not named `*-test.el' so the Makefile
;; does not try to run it as one.
;;
;; The NaN handling is the point.  A max-accumulating helper written the
;; obvious way -- (when (> e m) (setq m e)) -- never updates on a NaN, so a
;; run in which every number is NaN prints a maximum deviation of exactly
;; zero and passes.  That has happened in a sibling repository, under the
;; NeLisp standalone reader, and it is invisible: the passing output of a
;; broken run is byte-identical to the passing output of a working one.
;; Every comparison here therefore treats a NaN as a failure explicitly.

;;; Code:

(defvar nso-t--pass 0)
(defvar nso-t--fail 0)

(defun nso-t--nan-p (x) (and (floatp x) (/= x x)))

(defun nso-t (name ok &optional detail)
  "Record check NAME as passing when OK, with optional DETAIL."
  (if ok
      (progn (setq nso-t--pass (1+ nso-t--pass))
             (message "  PASS  %s%s" name (if detail (format "   %s" detail) "")))
    (setq nso-t--fail (1+ nso-t--fail))
    (message "  FAIL  %s%s" name (if detail (format "   %s" detail) "")))
  ok)

(defun nso-t-num (name got want tol)
  "Check that GOT is within TOL of WANT.  A NaN GOT fails, never passes."
  (let ((nan (nso-t--nan-p got)))
    (nso-t name
           (and (not nan) (numberp got) (< (abs (- got want)) tol))
           (format "got %s, want %s +/- %s%s" got want tol (if nan "  [NaN]" "")))))

(defun nso-t-lt (name a b)
  "Check A < B.  A NaN on either side fails."
  (let ((nan (or (nso-t--nan-p a) (nso-t--nan-p b))))
    (nso-t name (and (not nan) (< a b))
           (format "%s < %s%s" a b (if nan "  [NaN]" "")))))

(defun nso-t-gt (name a b)
  "Check A > B.  A NaN on either side fails."
  (let ((nan (or (nso-t--nan-p a) (nso-t--nan-p b))))
    (nso-t name (and (not nan) (> a b))
           (format "%s > %s%s" a b (if nan "  [NaN]" "")))))

(defun nso-t-green (name result)
  "Check that gate plist RESULT reports a pass."
  (nso-t name (and (plist-get result :pass) t)
         (format "gate pass=%S" (plist-get result :pass))))

(defun nso-t-red (name result)
  "Check that gate plist RESULT reports a failure.
This is the direction that makes a gate evidence rather than decoration."
  (nso-t name (null (plist-get result :pass))
         (let ((v (plist-get result :violations)))
           (if v (format "rejected: %s" (car v))
             (format "gate pass=%S" (plist-get result :pass))))))

(defun nso-t-signals (name thunk)
  "Check that calling THUNK signals an error rather than returning."
  (nso-t name
         (condition-case err
             (progn (funcall thunk) nil)
           (error (ignore err) t))
         "expected a signal"))

(defun nso-t-done (label)
  "Print the tally for LABEL and exit non-zero if anything failed."
  (message "%s: %d passed, %d failed" label nso-t--pass nso-t--fail)
  (when (> nso-t--fail 0)
    (kill-emacs 1)))

(provide 'nso-test-helper)
;;; nso-test-helper.el ends here
