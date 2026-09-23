;;; score-fake-states.el --- exercise the Score probe without a GPU -*- lexical-binding: t; -*-

;; Writes a states file with the same shape the encoder produces, so the whole
;; probe path -- split, standardise, both heads, three gates, bootstrap,
;; out-of-fold temperature, the band, the report writer -- runs in seconds
;; before an hour of GPU time is committed to it.  P1 lost forty encoded
;; examples to a crash after the encode; this is the cheaper order.
;;
;; Two modes, because one of them is the useful one:
;;
;;   signal (default)  the last position carries a direction scaled by the
;;                     level, plus noise.  Every gate should pass, which shows
;;                     the path can report a success.
;;   noise             pure noise.  Gates 2 and 3 should fail, which shows the
;;                     path can report a failure -- the direction that makes
;;                     the smoke test evidence rather than decoration.
;;
;; Run:  NSO_SCORE_FAKE=noise emacs -Q --batch -L build/elc -L lisp \
;;         -l tools/score-fake-states.el

(defvar nso-fk--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-fk--here))
(add-to-list 'load-path (expand-file-name "../build/elc" nso-fk--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-score)
(require 'nso-stub)

(defvar nso-fk--mode (or (getenv "NSO_SCORE_FAKE") "signal"))
(defvar nso-fk--dim 1024
  "The dimension the encoder actually produces.

Was 64, and that is how the first real run got through: the smoke matched the
real states' SCALE -- RMS about 4, deliberately -- and not their dimension,
where a standardised vector has norm about 32 and the fixed step of the day
overshot every iteration.  Matching one of the two axes and calling it a
realistic fixture is how a smoke test stays green through the failure it
exists to catch.  Slower now, and that is the price of the fixture being the
thing it stands in for.")
(defvar nso-fk--seq 3)
(defvar nso-fk--out
  (or (getenv "NSO_SCORE_STATES")
      (expand-file-name "../build/score-fake-states.eld" nso-fk--here)))

(let* ((form (with-temp-buffer
               (insert-file-contents
                (expand-file-name "../data/score-commitment.eld" nso-fk--here))
               (read (buffer-string))))
       (examples (append (plist-get form :examples) nil))
       (rng (nso-rng 13579))
       (dir (let ((v (make-vector nso-fk--dim 0.0)))
              (dotimes (j nso-fk--dim) (aset v j (- (nso-rng-float rng) 0.5)))
              v))
       (rows nil))
  (dolist (e examples)
    (let ((states nil))
      (dotimes (pos nso-fk--seq)
        (let ((v (make-vector nso-fk--dim 0.0)))
          ;; RMS near 4, like the real pooled states: P1 and P2 both had a
          ;; synthetic suite pass on unit-norm features while the shipped
          ;; setting diverged on the real geometry, and a smoke file that
          ;; repeats that mistake is worse than none.
          (dotimes (j nso-fk--dim) (aset v j (* 8.0 (- (nso-rng-float rng) 0.5))))
          (when (and (equal nso-fk--mode "signal") (= pos (1- nso-fk--seq)))
            (let ((g (* 3.0 (- (plist-get e :level) 2))))
              (dotimes (j nso-fk--dim)
                (aset v j (+ (aref v j) (* g (aref dir j)))))))
          (push v states)))
      (push (list :scenario (plist-get e :scenario)
                  :level (plist-get e :level)
                  :hard (plist-get e :hard)
                  :family (plist-get e :family)
                  :text (plist-get e :text)
                  :seq nso-fk--seq
                  :mid (nreverse states)
                  :final (nreverse (copy-sequence states)))
            rows)))
  (setq rows (nreverse rows))
  (with-temp-file nso-fk--out
    (let ((print-level nil) (print-length nil))
      (prin1 (list :dim nso-fk--dim :mid-layer 13 :rows rows) (current-buffer))))
  (message "wrote %d fake rows (%s) to %s" (length rows) nso-fk--mode nso-fk--out))

;;; score-fake-states.el ends here
