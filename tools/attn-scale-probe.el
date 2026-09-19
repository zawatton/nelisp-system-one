;;; attn-scale-probe.el --- does the learned pool saturate at mid depth? -*- lexical-binding: t; -*-

;; P1 left one row unexplained.  The learned attention pool trains fine at
;; final depth (training accuracy 1.000, held-out 0.702) and does not train at
;; all at mid depth (training accuracy 0.500 at both 140 and 252 examples, on
;; data every other configuration memorises whole).  The noise control ruled
;; out capacity: the same pool reaches 1.000 on random features of the same
;; shape.  What is left is a property of the real mid-depth features.
;;
;; The hypothesis on record is scale.  Real hidden states carry large
;; magnitudes; if they are larger at mid depth then the initial scores
;; u . h_i are spread further apart there, softmax(scores) starts near
;; one-hot, and the gradient through a saturated softmax -- which carries a
;; factor a_i(1 - a_i) -- is near zero.  The pool would then never move.
;;
;; Stated as predictions, so the measurement can refute them:
;;
;;   P1  mid-depth states have larger magnitude than final-depth ones.
;;   P2  at initialisation, the spread of scores within an example is larger
;;       at mid depth.
;;   P3  at initialisation, the largest softmax weight is closer to 1 at mid
;;       depth -- that is what "saturated" means here.
;;   P4  the gradient norm |du| at initialisation is smaller at mid depth.
;;
;; All four must hold for the hypothesis to survive.  If the magnitudes come
;; out equal, or the mid-depth softmax is the flatter one, it is dead and this
;; file says so.
;;
;; Reads the saved states, so no GPU and no donor table.
;; Run:  emacs -Q --batch -L lisp -l tools/attn-scale-probe.el

(defvar nso-as--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-as--here))

(require 'nso-head)

(defvar nso-as--states
  (or (getenv "NSO_P1_STATES")
      (expand-file-name "../build/p1-states.eld" nso-as--here)))

(defun nso-as--rms (v)
  (let ((s 0.0) (n (length v)))
    (dotimes (i n) (setq s (+ s (* (aref v i) (aref v i)))))
    (sqrt (/ s n))))

(defun nso-as--mean (xs)
  (let ((s 0.0)) (dolist (x xs) (setq s (+ s x))) (/ s (length xs))))

(defun nso-as--norm (v)
  (let ((s 0.0)) (dotimes (i (length v)) (setq s (+ s (* (aref v i) (aref v i)))))
       (sqrt s)))

(defun nso-as--depth-stats (rows depth)
  "Magnitudes, score spreads, softmax peaks and |du| at initialisation."
  (let* ((dim (length (car (plist-get (car rows) depth))))
         (model (nso-attn-model-make dim))
         (u (plist-get model :u))
         (rms nil) (spread nil) (peak nil))
    (dolist (r rows)
      (let* ((states (plist-get r depth))
             (m (length states))
             (scores (make-vector m 0.0))
             (i 0) (lo 1.0e30) (hi -1.0e30))
        (dolist (h states)
          (push (nso-as--rms h) rms)
          (let ((s (nso-dot u h)))
            (aset scores i s)
            (setq lo (min lo s) hi (max hi s)))
          (setq i (1+ i)))
        (push (- hi lo) spread)
        (let ((a (nso-softmax-vec scores)) (mx 0.0))
          (dotimes (j m) (setq mx (max mx (aref a j))))
          (push mx peak))))
    ;; One gradient step's worth of signal, at initialisation.
    (let* ((xss (mapcar (lambda (r) (plist-get r depth)) rows))
           (ys (mapcar (lambda (r) (float (plist-get r :label))) rows))
           (g (nso-attn-grad model xss ys 0.0)))
      (list :rms (nso-as--mean rms)
            :spread (nso-as--mean spread)
            :peak (nso-as--mean peak)
            :uniform (/ 1.0 (nso-as--mean (mapcar #'length xss)))
            :du (nso-as--norm (plist-get g :du))
            :dw (nso-as--norm (plist-get g :dw))))))

(if (not (file-readable-p nso-as--states))
    (princ (format "SKIP: no states at %s\n" nso-as--states))
  (let* ((saved (with-temp-buffer
                  (insert-file-contents nso-as--states)
                  (read (buffer-string))))
         (rows (plist-get saved :rows))
         (fin (nso-as--depth-stats rows :final))
         (mid (nso-as--depth-stats rows :mid)))
    (princ (format "%d rows\n\n" (length rows)))
    (princ "                        final        mid      ratio\n")
    (princ "  ------------------------------------------------\n")
    (dolist (k '(:rms :spread :peak :du :dw))
      (let ((f (plist-get fin k)) (m (plist-get mid k)))
        (princ (format "  %-18s %10.4g %10.4g %9.3g\n"
                       (substring (symbol-name k) 1) f m
                       (if (= f 0.0) 0.0 (/ m f))))))
    (princ (format "  %-18s %10.4g %10.4g\n" "uniform weight"
                   (plist-get fin :uniform) (plist-get mid :uniform)))
    (princ "\n  predictions\n")
    (let ((p1 (> (plist-get mid :rms) (plist-get fin :rms)))
          (p2 (> (plist-get mid :spread) (plist-get fin :spread)))
          (p3 (> (plist-get mid :peak) (plist-get fin :peak)))
          (p4 (< (plist-get mid :du) (plist-get fin :du))))
      (princ (format "  P1 mid states are larger            %s\n" (if p1 "HOLDS" "REFUTED")))
      (princ (format "  P2 mid score spread is wider        %s\n" (if p2 "HOLDS" "REFUTED")))
      (princ (format "  P3 mid softmax is nearer one-hot    %s\n" (if p3 "HOLDS" "REFUTED")))
      (princ (format "  P4 mid |du| is smaller              %s\n" (if p4 "HOLDS" "REFUTED")))
      (princ (format "\n  scale hypothesis: %s\n"
                     (if (and p1 p2 p3 p4) "SURVIVES all four"
                       "REFUTED -- at least one prediction failed"))))))

;;; attn-scale-probe.el ends here
