;;; nso-choice.el --- a Choice head that scores options it has never seen -*- lexical-binding: t; -*-

;;; Commentary:

;; Section 3.1 argues that a Choice head must score options against the state
;; rather than classify the state into N fixed slots, because the option set
;; belongs to the question and not to the model.  A slot head would need
;; retraining for every new question and could not read the option's own text.
;; This is that head, and `test/choice-test.el' holds it to the property that
;; motivated it: accuracy on option sets whose members never appeared in
;; training.
;;
;; Parameterisation, kept as small as the Noul head:
;;
;;   logit_i = ((a * s) + c) . o_i / sqrt(dim)
;;
;; where `s' is the pooled state, `o_i' the pooled embedding of option i, `a'
;; an elementwise gate on the state and `c' an option-only direction.  2*dim
;; parameters, against dim for Noul.
;;
;; The two full matrices of section 3.1's sketch, W_s and W_o, would be a
;; million parameters each; on a few hundred examples that is not a model, it
;; is a memory.  The diagonal form keeps the property that matters -- every
;; logit is a function of the option's embedding, so an option never seen in
;; training still gets a score -- and drops the capacity that only invites
;; overfitting at this scale.  Whether the full form is worth its parameters
;; is a question for data this phase does not have.
;;
;; No per-option bias term.  A learned b_i would be indexed by option, which
;; is exactly the slot-classifier assumption this head exists to avoid: it
;; cannot be evaluated on an option that was not in the training set.

;;; Code:

(require 'nso-head)

(defun nso-choice-make (dim)
  "Initial parameters over DIM features.
The gate starts at one and the option direction at zero, so the head begins as
plain similarity between state and option -- a reasonable thing to be before
it has seen anything, and one whose gradient is not structurally zero."
  (list :a (let ((v (make-vector dim 1.0))) v)
        :c (nso-zeros dim)))

(defun nso-choice--query (model s)
  "The vector this MODEL compares against option embeddings for state S."
  (let* ((a (plist-get model :a))
         (c (plist-get model :c))
         (dim (length a))
         (q (make-vector dim 0.0)))
    (dotimes (j dim) (aset q j (+ (* (aref a j) (aref s j)) (aref c j))))
    q))

(defun nso-choice-logits (model s options)
  "Scores of each vector in OPTIONS against state S under MODEL."
  (let* ((q (nso-choice--query model s))
         (scale (/ 1.0 (sqrt (float (length q)))))
         (out (make-vector (length options) 0.0))
         (i 0))
    (dolist (o options)
      (aset out i (* scale (nso-dot q o)))
      (setq i (1+ i)))
    out))

(defun nso-choice-probs (model s options)
  "Distribution over OPTIONS for state S."
  (nso-softmax-vec (nso-choice-logits model s options)))

;;; Loss and gradient
;;
;; An example is (:state VEC :options LIST-OF-VEC :label INDEX).  Cross-entropy
;; over the softmax, which is the proper scoring rule of section 4.2 in its
;; multiclass form; the Noul head's BCE is this with two options.

(defun nso-choice-loss (model examples l2)
  "Mean cross-entropy of MODEL over EXAMPLES plus an L2 term."
  (let ((n (length examples)) (sum 0.0))
    (dolist (e examples)
      (let* ((p (nso-choice-probs model (plist-get e :state) (plist-get e :options)))
             (q (max 1.0e-12 (aref p (plist-get e :label)))))
        (setq sum (+ sum (- (log q))))))
    (+ (/ sum n)
       (let ((a (plist-get model :a)) (c (plist-get model :c)))
         ;; The gate is regularised toward one rather than zero: zero would
         ;; mean "ignore the state", which is not the neutral hypothesis.
         (* 0.5 l2 (+ (let ((s 0.0))
                        (dotimes (j (length a))
                          (let ((d (- (aref a j) 1.0))) (setq s (+ s (* d d)))))
                        s)
                      (nso-dot c c)))))))

(defun nso-choice-grad (model examples l2)
  "Gradient of `nso-choice-loss'.  Returns (:da VEC :dc VEC)."
  (let* ((a (plist-get model :a))
         (c (plist-get model :c))
         (dim (length a))
         (n (length examples))
         (da (nso-zeros dim))
         (dc (nso-zeros dim))
         (scale (/ 1.0 (sqrt (float dim))))
         (dq (make-vector dim 0.0)))
    (dolist (e examples)
      (let* ((s (plist-get e :state))
             (options (plist-get e :options))
             (label (plist-get e :label))
             (p (nso-choice-probs model s options))
             (i 0))
        (dotimes (j dim) (aset dq j 0.0))
        ;; dL/dq = sum_i (p_i - y_i) * scale * o_i
        (dolist (o options)
          (let ((g (* scale (/ (- (aref p i) (if (= i label) 1.0 0.0)) n))))
            (nso-axpy dq g o))
          (setq i (1+ i)))
        (dotimes (j dim)
          (aset da j (+ (aref da j) (* (aref dq j) (aref s j))))
          (aset dc j (+ (aref dc j) (aref dq j))))))
    (dotimes (j dim) (aset da j (+ (aref da j) (* l2 (- (aref a j) 1.0)))))
    (nso-axpy dc l2 c)
    (list :da da :dc dc)))

(defconst nso-choice-gtol 5.0e-3
  "Gradient norm at which Choice fitting is considered converged.")
(defconst nso-choice-armijo 1.0e-4
  "Sufficient-decrease constant for Choice backtracking.")

(defun nso-choice--copy (model)
  (list :a (copy-sequence (plist-get model :a))
        :c (copy-sequence (plist-get model :c))))

(defun nso-choice--gnorm2 (g freeze-c)
  (let ((s 0.0))
    (dolist (part (if freeze-c (list (plist-get g :da))
                    (list (plist-get g :da) (plist-get g :dc))))
      (dotimes (i (length part)) (setq s (+ s (* (aref part i) (aref part i))))))
    s))

(defun nso-choice--descend (model base g step freeze-c)
  ;; Rebuild the trial explicitly so no previous rejected step leaks through.
  (dotimes (i (length (plist-get model :a)))
    (aset (plist-get model :a) i (- (aref (plist-get base :a) i)
                                    (* step (aref (plist-get g :da) i)))))
  (unless freeze-c
    (dotimes (i (length (plist-get model :c)))
      (aset (plist-get model :c) i (- (aref (plist-get base :c) i)
                                      (* step (aref (plist-get g :dc) i))))))
  model)

(defun nso-choice-train (examples &optional steps lr l2 freeze-c)
  "Fit a Choice head to EXAMPLES by gradient descent.

FREEZE-C holds the option-only direction at zero.  That term contributes
`c . o_i' to every logit, which is a score for the option alone with no
reference to the state -- a learned prior over options, expressed through
their embeddings rather than their indices.  On the options it was fitted on
that prior can only help; on options it has never seen it is arbitrary, and
arbitrary in a fixed direction, which is how a head ends up BELOW chance
rather than at it.  Freezing it is the ablation that tells the two apart."
  (let* ((model (nso-choice-make (length (plist-get (car examples) :state))))
         (steps (or steps 400))
         (step (or lr 0.5))
         (l2 (or l2 0.01))
         (i 0) (taken 0) (g nil)
         (tol2 (* nso-choice-gtol nso-choice-gtol)))
    (while (< i steps)
      (setq g (nso-choice-grad model examples l2) taken (1+ taken))
      (if (< (nso-choice--gnorm2 g freeze-c) tol2)
          (setq i steps)
        (let* ((l0 (nso-choice-loss model examples l2))
               (base (nso-choice--copy model))
               (g2 (nso-choice--gnorm2 g freeze-c))
               (tries 0) (ok nil))
          (setq step (* 2.0 step))
          (while (and (not ok) (< tries 60))
            (nso-choice--descend model base g step freeze-c)
            (if (<= (nso-choice-loss model examples l2)
                    (- l0 (* nso-choice-armijo step g2)))
                (setq ok t)
              (setq step (/ step 2.0) tries (1+ tries))))
          (unless ok
            (nso-choice--descend model base g 0.0 freeze-c)
            (setq i steps))))
      (setq i (1+ i)))
    (setq g (nso-choice-grad model examples l2))
    (plist-put (plist-put (plist-put model :final-gnorm
                                      (sqrt (nso-choice--gnorm2 g freeze-c)))
                          :final-loss (nso-choice-loss model examples l2))
               :steps-taken taken)))

;;; Shared low-rank projection head

(defun nso-choice-lowrank-make (dim rank)
  "Make a non-degenerate shared RANK-by-DIM projection.
Rows start as distinct coordinate directions (cycled when RANK exceeds DIM),
which gives nonzero state and option paths at the first gradient evaluation;
zero initial rows would make every P gradient zero."
  (list :rank rank :dim dim
        :p (let ((rows nil) (r 0))
             (while (< r rank)
               (let ((row (nso-zeros dim)))
                 (aset row (mod r dim) 1.0)
                 (push row rows))
               (setq r (1+ r)))
             (nreverse rows))))

(defun nso-choice-lowrank--project (model x)
  (let* ((rows (plist-get model :p)) (out (nso-zeros (length rows))) (i 0))
    (dolist (row rows) (aset out i (nso-dot row x)) (setq i (1+ i)))
    out))

(defun nso-choice-lowrank-logits (model s options)
  "Score OPTIONS after applying shared MODEL projection to state S."
  (let* ((u (nso-choice-lowrank--project model s))
         (scale (/ 1.0 (sqrt (float (plist-get model :rank)))))
         (out (make-vector (length options) 0.0)) (i 0))
    (dolist (o options)
      (aset out i (* scale (nso-dot u (nso-choice-lowrank--project model o))))
      (setq i (1+ i)))
    out))

(defun nso-choice-lowrank-probs (model s options)
  (nso-softmax-vec (nso-choice-lowrank-logits model s options)))

(defun nso-choice-lowrank-loss (model examples l2)
  (let ((sum 0.0) (n (length examples)))
    (dolist (e examples)
      (let* ((p (nso-choice-lowrank-probs model (plist-get e :state)
                                           (plist-get e :options)))
             (q (max 1.0e-12 (aref p (plist-get e :label)))))
        (setq sum (+ sum (- (log q))))))
    (+ (/ sum n)
       (let ((s 0.0))
         (dolist (row (plist-get model :p)) (setq s (+ s (nso-dot row row))))
         (* 0.5 l2 s)))))

(defun nso-choice-lowrank-grad (model examples l2)
  "Hand-derived gradient of the low-rank Choice loss.
For z_i=(Ps).(Po_i)/sqrt(R), d z_i/d P_r is
((P o_i)_r s + (P s)_r o_i)/sqrt(R)."
  (let* ((rows (plist-get model :p)) (rank (length rows))
         (dim (length (car rows))) (gp (mapcar (lambda (_) (nso-zeros dim)) rows))
         (scale (/ 1.0 (sqrt (float rank)))) (n (length examples)))
    (dolist (e examples)
      (let* ((s (plist-get e :state)) (opts (plist-get e :options))
             (label (plist-get e :label))
             (u (nso-choice-lowrank--project model s))
             (p (nso-choice-lowrank-probs model s opts)) (i 0))
        (dolist (o opts)
          (let* ((v (nso-choice-lowrank--project model o))
                 (err (* scale (/ (- (aref p i) (if (= i label) 1.0 0.0)) n))))
            (dotimes (r rank)
              (let ((row (nth r gp)))
                (dotimes (j dim)
                  (aset row j (+ (aref row j)
                                 (* err (+ (* (aref v r) (aref s j))
                                           (* (aref u r) (aref o j))))))))))
          (setq i (1+ i)))))
    (let ((i 0))
      (dolist (row gp)
        (nso-axpy row l2 (nth i rows))
        (setq i (1+ i))))
    (list :dp gp)))

(defun nso-choice-lowrank--gnorm2 (g)
  (let ((s 0.0))
    (dolist (row (plist-get g :dp))
      (dotimes (i (length row)) (setq s (+ s (* (aref row i) (aref row i))))))
    s))

(defun nso-choice-lowrank--copy (model)
  (list :rank (plist-get model :rank) :dim (plist-get model :dim)
        :p (mapcar #'copy-sequence (plist-get model :p))))

(defun nso-choice-lowrank--descend (model base g step)
  (let ((rows (plist-get model :p)) (brows (plist-get base :p))
        (grows (plist-get g :dp)) (r 0))
    (dolist (row rows)
      (let ((b (nth r brows)) (gr (nth r grows)) (i 0))
        (dotimes (_ (length row))
          (aset row i (- (aref b i) (* step (aref gr i))))
          (setq i (1+ i))))
      (setq r (1+ r)))
    model))

(defun nso-choice-lowrank-train (examples rank &optional steps lr l2)
  "Fit a RANK-dimensional shared projection with Armijo backtracking."
  (let* ((model (nso-choice-lowrank-make (length (plist-get (car examples) :state)) rank))
         (steps (or steps 400)) (step (or lr 0.5)) (l2 (or l2 0.01))
         (i 0) (taken 0) (g nil))
    (while (< i steps)
      (setq g (nso-choice-lowrank-grad model examples l2) taken (1+ taken))
      (let ((g2 (nso-choice-lowrank--gnorm2 g)))
        (if (< g2 (* nso-choice-gtol nso-choice-gtol))
            (setq i steps)
          (let ((l0 (nso-choice-lowrank-loss model examples l2))
                (base (nso-choice-lowrank--copy model)) (tries 0) (ok nil))
            (setq step (* 2.0 step))
            (while (and (not ok) (< tries 60))
              (nso-choice-lowrank--descend model base g step)
              (if (<= (nso-choice-lowrank-loss model examples l2)
                      (- l0 (* nso-choice-armijo step g2)))
                  (setq ok t)
                (setq step (/ step 2.0) tries (1+ tries))))
            (unless ok
              (nso-choice-lowrank--descend model base g 0.0)
              (setq i steps)))))
      (setq i (1+ i)))
    (setq g (nso-choice-lowrank-grad model examples l2))
    (plist-put (plist-put (plist-put model :final-gnorm
                                      (sqrt (nso-choice-lowrank--gnorm2 g)))
                          :final-loss (nso-choice-lowrank-loss model examples l2))
               :steps-taken taken)))

(defun nso-choice-lowrank-accuracy (model examples)
  "Top-one accuracy for the low-rank Choice head."
  (let ((ok 0) (n 0))
    (dolist (e examples)
      (let ((p (nso-choice-lowrank-probs model (plist-get e :state)
                                           (plist-get e :options))) (best 0))
        (dotimes (i (length p)) (when (> (aref p i) (aref p best)) (setq best i)))
        (when (= best (plist-get e :label)) (setq ok (1+ ok))))
      (setq n (1+ n)))
    (/ (float ok) n)))

;;; Answers

(defun nso-choice-answer (model s option-names option-vecs)
  "A typed Choice answer, ready for `nso-type-gate'."
  (let* ((p (nso-choice-probs model s option-vecs))
         (best 0)
         (probs nil)
         (i 0))
    (dotimes (j (length p)) (when (> (aref p j) (aref p best)) (setq best j)))
    (dolist (name option-names)
      (push (cons name (aref p i)) probs)
      (setq i (1+ i)))
    (list :choice (nth best option-names)
          :probabilities (nreverse probs)
          ;; Placeholder, as in `nso-noul-answer': section 2 leaves the
          ;; definition of confidence open and P2 does not close it either.
          :confidence (aref p best))))

(defun nso-choice-accuracy (model examples)
  "Top-1 accuracy of MODEL over EXAMPLES."
  (let ((ok 0) (n 0))
    (dolist (e examples)
      (let* ((p (nso-choice-probs model (plist-get e :state) (plist-get e :options)))
             (best 0))
        (dotimes (j (length p)) (when (> (aref p j) (aref p best)) (setq best j)))
        (when (= best (plist-get e :label)) (setq ok (1+ ok))))
      (setq n (1+ n)))
    (/ (float ok) n)))

(provide 'nso-choice)
;;; nso-choice.el ends here
