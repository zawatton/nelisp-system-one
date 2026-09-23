;;; p2-fit-probe.el --- did the Choice head fit at all? -*- lexical-binding: t; -*-

;; P2's first run reported 0.250 held-out accuracy on a four-way choice, which
;; is exactly chance, and 0.000 against all twelve options.  Before any of that
;; is read as evidence about section 3.1's architecture, one number has to be
;; explained: TRAINING accuracy was 0.250 on an eight-way task where chance is
;; 0.125.  The head did not fit the data it was fitted on.  A model that did
;; not train says nothing about whether its design can generalise.
;;
;; The suspicion is the failure P1 already diagnosed once.  `nso-choice-train'
;; runs on raw pooled states whose RMS is about 3.7, at a fixed learning rate
;; of 0.5, while `test/choice-test.el' exercised it on unit-norm synthetic
;; vectors whose components are far smaller.  The synthetic test would not see
;; a conditioning problem that only appears at the real scale.
;;
;; So: train accuracy across learning rates, with the states raw and
;; standardised.  If standardising recovers a fit, the architecture was never
;; on trial; if nothing fits at any setting, it was.
;;
;; Run:  emacs -Q --batch -L build/elc -L lisp -l tools/p2-fit-probe.el

(defvar nso-fp--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(add-to-list 'load-path (expand-file-name "../lisp" nso-fp--here))

(load (expand-file-name "../lisp/nso-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nso-choice)
(require 'nso-probe)

(defvar nso-fp--states (expand-file-name "../build/p2-states.eld" nso-fp--here))
(defvar nso-fp--data
  (with-temp-buffer
    (insert-file-contents (expand-file-name "../data/choice-intent.eld" nso-fp--here))
    (read (buffer-string))))

(defun nso-fp--rms (v)
  (let ((s 0.0)) (dotimes (i (length v)) (setq s (+ s (* (aref v i) (aref v i)))))
       (sqrt (/ s (length v)))))

(if (not (file-readable-p nso-fp--states))
    (princ "SKIP: no p2 states\n")
  (let* ((rows (plist-get (with-temp-buffer
                            (insert-file-contents nso-fp--states)
                            (read (buffer-string)))
                          :rows))
         (by-text (let ((h (make-hash-table :test 'equal)))
                    (dolist (r rows) (puthash (plist-get r :text) r h)) h))
         (tmpl (plist-get nso-fp--data :template))
         (topics (append (plist-get nso-fp--data :topics) nil))
         (seen (let (o) (dolist (tp topics)
                          (unless (plist-get tp :held) (push (plist-get tp :name) o)))
                    (nreverse o)))
         (pool (lambda (r) (nso-pool-last (plist-get r :mid))))
         ;; Statistics from the training states only, as everywhere else here.
         (train-states (let (o)
                         (dolist (e (append (plist-get nso-fp--data :examples) nil))
                           (when (member (plist-get e :topic) seen)
                             (push (funcall pool
                                            (gethash (format tmpl (plist-get e :text))
                                                     by-text))
                                   o)))
                         (nreverse o)))
         (std (nso-standardizer train-states))
         (mk (lambda (standardise)
               (let ((vecs (mapcar (lambda (nm)
                                     (let ((v (funcall pool (gethash nm by-text))))
                                       (if standardise (nso-standardize std v) v)))
                                   seen))
                     (out nil))
                 (dolist (e (append (plist-get nso-fp--data :examples) nil))
                   (when (member (plist-get e :topic) seen)
                     (let* ((r (gethash (format tmpl (plist-get e :text)) by-text))
                            (s (funcall pool r))
                            (label nil) (i 0))
                       (dolist (nm seen)
                         (when (equal nm (plist-get e :topic)) (setq label i))
                         (setq i (1+ i)))
                       (push (list :state (if standardise (nso-standardize std s) s)
                                   :options vecs :label label)
                             out))))
                 (nreverse out)))))
    (princ (format "state RMS raw %.3f, standardised %.3f\n"
                   (nso-fp--rms (car train-states))
                   (nso-fp--rms (nso-standardize std (car train-states)))))
    (princ (format "%d training examples, %d options, chance %.3f\n\n"
                   (length (funcall mk nil)) (length seen) (/ 1.0 (length seen))))
    (princ "  states         lr      train acc   loss\n")
    (princ "  ------------------------------------------\n")
    (dolist (mode '(nil t))
      (let ((ex (funcall mk mode)))
        (dolist (lr '(0.5 0.1 0.02 0.005 0.001))
          (let* ((model (nso-choice-train ex 400 lr 0.02))
                 (acc (nso-choice-accuracy model ex))
                 (loss (nso-choice-loss model ex 0.02)))
            (princ (format "  %-14s %-7s %9.3f  %9.4g%s\n"
                           (if mode "standardised" "raw") lr acc loss
                           (if (/= loss loss) "  [NaN]" "")))))))
    (princ "\n  A row that fits where the shipped setting did not means the\n")
    (princ "  architecture was never on trial.\n")))

;;; p2-fit-probe.el ends here
