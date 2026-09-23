;;; p3-bincount.el --- how many bins does n=84 support? -*- lexical-binding: t; -*-

;; P3 failed its gate at ten bins and the post-mortem showed the bin count had
;; never been argued for -- it was carried over from the default and compared
;; against a number measured at five.  This picks one on grounds that cannot
;; be tuned to the answer, because it never looks at the answer: everything
;; here runs on synthetic stubs whose calibration is known by construction,
;; at the same n as the real held-out split.
;;
;; Two quantities decide it, and they pull opposite ways:
;;
;;   bias     A stub calibrated BY CONSTRUCTION has a true ECE of zero, so
;;            whatever the estimator reports on it at this n is bias.  More
;;            bins means fewer samples each and more of it.
;;   power    A 4x-overconfident stub must still be caught.  Fewer bins lets
;;            opposite errors cancel inside a bin, and in the limit of one bin
;;            the estimator cannot see a sign change at all.
;;
;; The rule, fixed here before the real data is touched again: *the largest
;; bin count whose bias stays at or below 0.02 -- two fifths of the 0.05 gate
;; -- while still failing the 4x-overconfident control in at least 95% of
;; draws.*  Both halves are necessary; a bin count that only satisfies the
;; first is an instrument that cannot see, and one that only satisfies the
;; second is an instrument that cries wolf.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/p3-bincount.el

(defvar nso-bc--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-bc--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-bc--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-metrics)
(require 'nso-stub)

(defvar nso-bc--n 84 "The size of the P1 held-out split.")
(defvar nso-bc--draws 200)
(defvar nso-bc--gate 0.05)
(defvar nso-bc--max-bias 0.02)
(defvar nso-bc--min-power 0.95)

(princ (format "n = %d (the P1 held-out split), %d draws per cell\n"
               nso-bc--n nso-bc--draws))
(princ (format "rule: largest bins with bias <= %.2f and power >= %.2f against\n"
               nso-bc--max-bias nso-bc--min-power))
(princ (format "      a 4x-overconfident stub at the %.2f gate\n\n" nso-bc--gate))
(princ "  bins |   bias | power | verdict\n")
(princ "  -----+--------+-------+--------\n")

(let ((chosen nil))
  (dolist (bins '(2 3 4 5 8 10 15 20))
    (let ((bias 0.0) (caught 0) (i 0))
      (while (< i nso-bc--draws)
        (let* ((cal (nso-stub-calibrated (+ 7000 i) nso-bc--n 2))
               (over (nso-stub-tempered cal 0.25)))
          (setq bias (+ bias (plist-get (nso-ece cal bins) :ece)))
          (when (>= (plist-get (nso-ece over bins) :ece) nso-bc--gate)
            (setq caught (1+ caught))))
        (setq i (1+ i)))
      (let* ((b (/ bias nso-bc--draws))
             (pw (/ (float caught) nso-bc--draws))
             (ok (and (<= b nso-bc--max-bias) (>= pw nso-bc--min-power))))
        (when ok (setq chosen bins))
        (princ (format "  %4d | %.4f | %.3f | %s\n" bins b pw
                       (cond ((> b nso-bc--max-bias) "too biased")
                             ((< pw nso-bc--min-power) "too blind")
                             (t "usable")))))))
  (princ "\n")
  (if chosen
      (progn
        (princ (format "CHOSEN: %d equal-width bins\n" chosen))
        (princ (format "At that count the estimator's own bias on a perfectly\n"))
        (princ (format "calibrated model of this size is the figure in the table,\n"))
        (princ (format "so that much of any %.2f reading is the instrument.\n" nso-bc--gate)))
    (princ "NO BIN COUNT SATISFIES BOTH HALVES AT THIS n.\n")
    (princ "That would mean the held-out split is too small to calibrate on,\n")
    (princ "and the answer is more data rather than a different bin count.\n")))


;;; --- how much held-out data would the gate need? -------------------------
;;
;; The table above says n=84 cannot support the gate at any bin count, so the
;; useful question is no longer which bin count but how much data.  Same
;; construction, sweeping n: the bias is still measured on a stub that is
;; calibrated by construction, so it is still the estimator's own floor.

(princ "\n  how the bias floor falls with n (calibrated-by-construction stub)\n\n")
(princ "       n |  5 bins | 10 bins | power@5 | power@10\n")
(princ "  -------+---------+---------+---------+---------\n")
(let ((target nil))
  (dolist (n '(84 200 500 1000 2000 5000))
    (let ((b5 0.0) (b10 0.0) (c5 0) (c10 0) (draws 60) (i 0))
      (while (< i draws)
        (let* ((cal (nso-stub-calibrated (+ 9000 i) n 2))
               (over (nso-stub-tempered cal 0.25)))
          (setq b5 (+ b5 (plist-get (nso-ece cal 5) :ece))
                b10 (+ b10 (plist-get (nso-ece cal 10) :ece)))
          (when (>= (plist-get (nso-ece over 5) :ece) nso-bc--gate) (setq c5 (1+ c5)))
          (when (>= (plist-get (nso-ece over 10) :ece) nso-bc--gate) (setq c10 (1+ c10))))
        (setq i (1+ i)))
      (setq b5 (/ b5 draws) b10 (/ b10 draws))
      (when (and (null target) (<= b10 nso-bc--max-bias)) (setq target n))
      (princ (format "  %6d |  %.4f |  %.4f |   %.2f  |   %.2f\n"
                     n b5 b10 (/ (float c5) draws) (/ (float c10) draws)))))
  (princ "\n")
  (if target
      (princ (format "The gate becomes measurable at ten bins from about n = %d.\n" target))
    (princ "Even n = 5000 does not bring the ten-bin bias under the limit.\n")))

;;; p3-bincount.el ends here
