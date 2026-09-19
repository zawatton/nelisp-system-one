;;; p1-fake-states.el --- a full-size states file with no signal in it -*- lexical-binding: t; -*-

;; Writes build/p1-states.eld with the real shape -- 140 rows, dim 1024, two
;; depths, the real pairs and labels -- and pure noise where the hidden states
;; would be.
;;
;; Two jobs.  It exercises every path in the probe stage at full scale without
;; spending an encoder pass, which is what the last smoke run failed to do: the
;; encoder worked and the probe stage was the part that broke, after 80 minutes
;; of GPU time had already been committed in the same process.
;;
;; And it is a leak control at the scale the real run uses.  These features
;; cannot predict the labels, so every held-out number the probe reports from
;; them should sit at chance.  Anything that comes back confident here is
;; measuring the pipeline, not the donor.
;;
;; Run:  emacs -Q --batch -l tools/p1-fake-states.el
;;  then emacs -Q --batch -l tools/p1-encode-probe.el   (NSO_P1_STAGE=probe)

(defvar nso-fake--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-fake--here))

(require 'nso-probe)
(require 'nso-stub)

(let* ((data (nso-probe-load (expand-file-name "../data/noul-outcome.eld"
                                               nso-fake--here)))
       (build (expand-file-name "../build" nso-fake--here))
       (dim 1024)
       (rng (nso-rng 20260919))
       (rows nil))
  (unless (file-directory-p build) (make-directory build t))
  (dolist (e (plist-get data :examples))
    (let* ((seq (+ 7 (mod (plist-get e :pair) 5)))
           (mk (lambda ()
                 (let ((out nil) (p 0))
                   (while (< p seq)
                     (let ((v (make-vector dim 0.0)))
                       (dotimes (j dim)
                         (aset v j (- (nso-rng-float rng) 0.5)))
                       (push v out))
                     (setq p (1+ p)))
                   (nreverse out)))))
      (push (list :pair (plist-get e :pair)
                  :label (plist-get e :label)
                  :hard (plist-get e :hard)
                  :text (plist-get e :text)
                  :seq seq
                  :mid (funcall mk)
                  :final (funcall mk))
            rows)))
  (setq rows (nreverse rows))
  (let ((out (expand-file-name "p1-states.eld" build)))
    (with-temp-file out
      (let ((print-level nil) (print-length nil))
        (prin1 (list :dim dim :mid-layer 13 :rows rows) (current-buffer))))
    (princ (format "wrote %d fake rows to %s (%.0f MB)\n"
                   (length rows) out
                   (/ (float (nth 7 (file-attributes out))) 1048576.0)))))

;;; p1-fake-states.el ends here
