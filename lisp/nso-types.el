;;; nso-types.el --- typed questions and answers, and the gate that decides -*- lexical-binding: t; -*-

;;; Commentary:

;; A System One answer is a value of a type declared before the question is
;; asked.  That is the whole of the "cannot hallucinate" claim: an ill-typed
;; answer is unrepresentable rather than merely rare.
;;
;; A claim like that is worth exactly as much as the check behind it, so the
;; check lives here and `test/type-gate-test.el' spends most of its length
;; feeding it answers that it must reject.  A gate nobody has watched fail is
;; not evidence of anything.
;;
;; Three question types, one shape: `noul' is `choice' over two options and
;; `score' is `choice' with an ordering imposed on them, so the validation
;; below shares a core and differs only at the edges.

;;; Code:

(defconst nso-max-cardinality 255
  "Largest option set a question may declare.
Adopted from the vendor's published cap.  It is a contract rather than a
kernel limit: it bounds the per-option encoder passes that section 3.1 of
`docs/design/01-system-one.org' trades for a per-query option set.")

(defconst nso-prob-sum-tolerance 1e-9
  "How far a reported distribution may sum from 1.0 and still be one.")

;;; Questions

(defun nso-make-choice (criteria options)
  "A question that picks exactly one of OPTIONS, judged by CRITERIA."
  (list :type 'choice :criteria criteria :options options))

(defun nso-make-score (criteria levels)
  "A question that rates against LEVELS, which are ordered worst-first."
  (list :type 'score :criteria criteria :options levels))

(defun nso-make-noul (question)
  "A yes/no QUESTION."
  (list :type 'noul :criteria question :options '(yes no)))

;;; Predicates

(defun nso-nan-p (x)
  "Non-nil when X is a NaN.
Spelled out rather than borrowed because a NaN that reaches a metric
becomes a quiet wrong number instead of an error."
  (and (floatp x) (/= x x)))

(defun nso-prob-p (x)
  "Non-nil when X is a real number in [0,1]."
  (and (numberp x) (not (nso-nan-p x)) (<= 0 x) (<= x 1)))

;;; Validation

(defun nso-validate-question (q)
  "Return a list of the ways Q fails to be a well-formed question."
  (let ((violations nil)
        (options (plist-get q :options)))
    (cond
     ((not (memq (plist-get q :type) '(choice score noul)))
      (push (format "unknown question type: %S" (plist-get q :type)) violations))
     ((not (listp options))
      (push "options is not a list" violations))
     ((null options)
      (push "option set is empty" violations))
     (t
      (when (> (length options) nso-max-cardinality)
        (push (format "cardinality %d exceeds the cap of %d"
                      (length options) nso-max-cardinality)
              violations))
      (let ((seen nil))
        (dolist (o options)
          (if (member o seen)
              (push (format "duplicate option: %S" o) violations)
            (push o seen))))))
    (nreverse violations)))

(defun nso--answer-key (q)
  "Which key of an answer carries the chosen option for question Q.

Choice and Noul report `:choice'; Score reports `:score', because section 2
gives it that signature.  One validator covers all three, but only if it looks
in the right place: reading `:choice' out of a Score answer finds nil and
rejects it as \"not in the declared option set\", which is a true sentence
about the wrong thing.  This gate did exactly that from P0 until the ordinal
head was built, and its own suite agreed with it -- the rejection tests were
green because every Score answer was rejected, including the valid ones.  A
gate can be blind to an entire primitive while passing tests that only ever
ask it to say no."
  (if (eq (plist-get q :type) 'score) :score :choice))

(defun nso--validate-choice (q a)
  "Return a list of the ways A fails to answer the choice/score question Q."
  (let* ((options (plist-get q :options))
         (choice (plist-get a (nso--answer-key q)))
         (probs (plist-get a :probabilities))
         (conf (plist-get a :confidence))
         (violations nil))
    (unless (member choice options)
      (push (format "answer %S is not in the declared option set" choice)
            violations))
    (if (not (listp probs))
        (push "probabilities is not an alist" violations)
      (dolist (o options)
        (unless (assoc o probs)
          (push (format "no probability reported for option %S" o) violations)))
      (dolist (cell probs)
        (if (not (consp cell))
            (push (format "malformed probability entry: %S" cell) violations)
          (unless (member (car cell) options)
            (push (format "probability reported for undeclared option %S"
                          (car cell))
                  violations))
          (unless (nso-prob-p (cdr cell))
            (push (format "probability for %S is not a probability: %S"
                          (car cell) (cdr cell))
                  violations))))
      ;; Sum and argmax are only meaningful once every entry is a probability
      ;; and the option sets agree; reporting them otherwise would bury the
      ;; real violation under a derived one.
      (when (and (null violations) (= (length probs) (length options)))
        (let ((sum 0.0))
          (dolist (cell probs) (setq sum (+ sum (cdr cell))))
          (unless (< (abs (- sum 1.0)) nso-prob-sum-tolerance)
            (push (format "probabilities sum to %.12f, not 1" sum) violations)))
        (let ((top nil) (best -1.0))
          (dolist (cell probs)
            (when (> (cdr cell) best)
              (setq best (cdr cell) top (car cell))))
          (unless (equal top choice)
            (push (format "answer %S is not the argmax of its own distribution (%S is)"
                          choice top)
                  violations)))))
    (when (and conf (not (nso-prob-p conf)))
      (push (format "confidence is not in [0,1]: %S" conf) violations))
    (nreverse violations)))

(defun nso--validate-noul (_q a)
  "Return a list of the ways A fails to answer a noul question."
  (let ((p (plist-get a :p-yes))
        (conf (plist-get a :confidence))
        (violations nil))
    (unless (nso-prob-p p)
      (push (format "p-yes is not a probability: %S" p) violations))
    (when (and conf (not (nso-prob-p conf)))
      (push (format "confidence is not in [0,1]: %S" conf) violations))
    (nreverse violations)))

(defun nso-validate-answer (q a)
  "Return a list of the ways A fails to be an answer to Q.  Nil means valid."
  (let ((qv (nso-validate-question q)))
    (if qv
        (mapcar (lambda (v) (concat "question: " v)) qv)
      (if (eq (plist-get q :type) 'noul)
          (nso--validate-noul q a)
        (nso--validate-choice q a)))))

(defun nso-type-gate (q a)
  "Return (:pass BOOL :violations LIST) for answer A to question Q."
  (let ((v (nso-validate-answer q a)))
    (list :pass (null v) :violations v)))

(provide 'nso-types)
;;; nso-types.el ends here
