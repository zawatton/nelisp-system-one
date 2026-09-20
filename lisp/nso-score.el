;;; nso-score.el --- an ordinal head: levels that are ordered, and stay ordered -*- lexical-binding: t; -*-

;;; Commentary:

;; Section 2 makes a claim about Score that is easy to state and easy to let
;; slide: the levels are ORDINAL.  "poor < fair < good" carries information a
;; softmax over three labels discards, and discarding it costs twice -- once in
;; parameters, because K independent weight vectors each rediscover the same
;; direction, and once in the errors that remain, because a nominal head has no
;; reason to prefer a near miss to a far one.
;;
;; The model here is the cumulative-link (proportional odds) form:
;;
;;   f(x) = w . x                    one latent quality, dim parameters
;;   c_k  = sigma(theta_k - f)       = P(level <= k),  k = 0 .. K-2
;;   p_k  = c_k - c_{k-1}            with c_{-1} = 0 and c_{K-1} = 1
;;
;; dim + (K-1) parameters against a nominal head's K*dim.  At dim = 1024 and
;; K = 5 that is 1028 against 5120, fitted on a few hundred examples.
;;
;; ORDERING IS ENFORCED BY CONSTRUCTION, not by a penalty:
;;
;;   theta_0 = t,   theta_k = theta_{k-1} + softplus(d_{k-1})
;;
;; so theta is strictly increasing for every value the optimiser can reach, and
;; every p_k is therefore non-negative for every input.  That is the move
;; `nso-types.el' makes at the boundary, made instead in the parameterisation:
;; a distribution that is not a distribution is unrepresentable rather than
;; merely unlikely.  A penalty would have left the failure possible and quiet,
;; which is the shape of every bug this repository has had to find twice.
;;
;; f carries no bias term.  A constant added to f shifts every cutpoint
;; equally, so (w, b, theta) and (w, 0, theta + b) are the same model; keeping
;; b would hand gradient descent a flat direction to wander along and make two
;; runs incomparable for a reason that has nothing to do with the data.
;;
;; The gradient below is derived by hand, so `test/score-test.el' checks it
;; against central finite differences on every parameter block before it checks
;; anything that the gradient is used to produce.

;;; Code:

(require 'nso-head)
(require 'nso-metrics)
(require 'nso-probe)

(defconst nso-score-softplus-linear 30.0
  "Above this, softplus is its own argument to within a float's resolution.")

(defun nso-score--softplus (x)
  "log(1 + exp(X)), without overflowing at either end."
  (cond ((> x nso-score-softplus-linear) x)
        ((< x (- nso-score-softplus-linear)) (exp x))
        (t (log (+ 1.0 (exp x))))))

;;; Parameters

(defun nso-score-make (dim k)
  "Initial parameters for K ordered levels over DIM features.

The cutpoints start evenly spaced around zero, one unit apart, which is the
spacing a standardised f sees as roughly one standard deviation.  Starting
them all at the same place would make the initial distribution degenerate and
the initial gradient nearly symmetric between adjacent levels, which is a slow
place to begin and an easy one to mistake for a model that cannot learn."
  (unless (>= k 2) (error "nso-score: %d levels is not an ordinal scale" k))
  (list :w (nso-zeros dim)
        :k k
        ;; theta_0, so that the K-1 cutpoints straddle zero.
        :t (- 1.0 (/ k 2.0))
        ;; softplus(d) = 1 gives unit spacing.
        :d (make-vector (max 0 (- k 2)) (log (- (exp 1.0) 1.0)))))

(defun nso-score-cutpoints (model)
  "The K-1 cutpoints of MODEL, strictly increasing by construction."
  (let* ((d (plist-get model :d))
         (out (make-vector (1+ (length d)) 0.0))
         (acc (plist-get model :t)))
    (aset out 0 acc)
    (dotimes (j (length d))
      (setq acc (+ acc (nso-score--softplus (aref d j))))
      (aset out (1+ j) acc))
    out))

;;; Forward

(defun nso-score-margins (model x)
  "The K-1 values theta_k - f(X) under MODEL.

Everything the distribution depends on is here, which is what makes the
ordinal temperature of `nso-score-temperature-fit' a scalar on one vector
rather than a rescaling of K independent logits."
  (let* ((th (nso-score-cutpoints model))
         (f (nso-dot (plist-get model :w) x))
         (z (make-vector (length th) 0.0)))
    (dotimes (k (length th)) (aset z k (- (aref th k) f)))
    z))

(defun nso-score-probs-from-margins (z)
  "Distribution over K = (length Z) + 1 levels from margins Z.

Z is increasing whenever it came from `nso-score-margins', sigma is monotone,
and the differences below are therefore non-negative -- exactly, not
approximately, since the two sigmoids are computed by the same code.  They can
underflow to zero, so callers take a log only through a floor."
  (let* ((m (length z))
         (p (make-vector (1+ m) 0.0))
         (prev 0.0))
    (dotimes (k m)
      (let ((c (nso-sigmoid (aref z k))))
        (aset p k (- c prev))
        (setq prev c)))
    (aset p m (- 1.0 prev))
    p))

(defun nso-score-probs (model x)
  "Distribution over the levels of MODEL for feature vector X."
  (nso-score-probs-from-margins (nso-score-margins model x)))

;;; Readouts
;;
;; Three of them, because an ordinal scale has three defensible point
;; predictions and they do not agree.  The argmax is the mode; the median
;; minimises expected absolute error and the mode does not; the mean is not a
;; level at all but is the right thing to average.  Section 2's contract --
;; shared with Choice, and enforced by `nso-types.el' -- requires the reported
;; answer to be the argmax of the reported distribution, so that is what
;; `nso-score-answer' reports.  The others are measured, and if the median
;; wins on held-out MAE then the shared contract is costing Score something
;; and the place to fix it is section 2, not here.

(defun nso-score-mode (p)
  "Index of the largest entry of distribution P, first on a tie."
  (let ((best 0))
    (dotimes (k (length p)) (when (> (aref p k) (aref p best)) (setq best k)))
    best))

(defun nso-score-median (p)
  "Smallest level of P whose cumulative probability reaches one half."
  (let ((acc 0.0) (out (1- (length p))) (k 0) (done nil))
    (while (and (not done) (< k (length p)))
      (setq acc (+ acc (aref p k)))
      (when (>= acc 0.5) (setq out k done t))
      (setq k (1+ k)))
    out))

(defun nso-score-expected (p)
  "Probability-weighted mean level of P.  Not a level; a number between them."
  (let ((s 0.0))
    (dotimes (k (length p)) (setq s (+ s (* k (aref p k)))))
    s))

(defun nso-score-unimodal-p (p)
  "Non-nil when P rises to its mode and falls after it, with no second peak.

Not a property the parameterisation guarantees, and the commentary would be
wrong to claim it.  With evenly spaced cutpoints the increments of a sigmoid
are bell-shaped in k and the distribution is unimodal -- but the cutpoints are
FITTED, and a wide first and last interval with narrow ones between them puts
mass at both ends and little in the middle.  Whether the fitted model does
that is a measurement, and the Score run reports the fraction rather than
assuming the answer."
  (let ((n (length p)) (m (nso-score-mode p)) (ok t) (k 0))
    (dotimes (j m)
      (when (> (aref p j) (aref p (1+ j))) (setq ok nil)))
    (setq k m)
    (while (< k (1- n))
      (when (< (aref p k) (aref p (1+ k))) (setq ok nil))
      (setq k (1+ k)))
    ok))

;;; Loss and gradient
;;
;; L = -log p_y, the multiclass proper scoring rule of section 4.2, the same
;; one `nso-choice-loss' uses.  Writing s_k = c_k (1 - c_k) = dc_k/dtheta_k:
;;
;;   dL/df        = (s_y - s_{y-1}) / p_y
;;   dL/dtheta_y  = -s_y / p_y                       (absent when y = K-1)
;;   dL/dtheta_{y-1} = +s_{y-1} / p_y                (absent when y = 0)
;;
;; and then through the parameterisation, where d_j feeds every cutpoint from
;; j+1 upwards:
;;
;;   dL/dt   = sum_k dL/dtheta_k
;;   dL/dd_j = sigma(d_j) * sum_{k >= j+1} dL/dtheta_k
;;   dL/dw   = (dL/df) x
;;
;; L2 is applied to w only.  The cutpoints are location parameters of the
;; scale itself: shrinking them toward zero would express a belief that the
;; levels are equally spaced and centred, which is a claim about the data, not
;; a regulariser.

(defconst nso-score-prob-floor 1.0e-12
  "Floor under p_y before taking its log.")

(defun nso-score-loss (model xs ys l2)
  "Mean negative log likelihood of MODEL over XS/YS, plus an L2 term on w."
  (let ((n 0) (sum 0.0) (rest ys))
    (dolist (x xs)
      (let ((p (nso-score-probs model x)))
        (setq sum (+ sum (- (log (max nso-score-prob-floor
                                      (aref p (car rest))))))
              n (1+ n)
              rest (cdr rest))))
    (+ (/ sum n)
       (let ((w (plist-get model :w))) (* 0.5 l2 (nso-dot w w))))))

(defun nso-score-grad (model xs ys l2)
  "Gradient of `nso-score-loss'.  Returns (:dw VEC :dt FLOAT :dd VEC)."
  (let* ((w (plist-get model :w))
         (d (plist-get model :d))
         (dim (length w))
         (m (1+ (length d)))            ; number of cutpoints, K-1
         (n (length xs))
         (dw (nso-zeros dim))
         (dtheta (make-vector m 0.0))
         (rest ys))
    (dolist (x xs)
      (let* ((z (nso-score-margins model x))
             (p (nso-score-probs-from-margins z))
             (y (car rest))
             (py (max nso-score-prob-floor (aref p y)))
             ;; s_k only exists for a cutpoint index in range; the two
             ;; boundary cases (y = 0 and y = K-1) are the missing cutpoints
             ;; below and above the scale, where c is pinned at 0 and 1 and
             ;; the derivative is zero.
             (s-hi (if (< y m) (let ((c (nso-sigmoid (aref z y)))) (* c (- 1.0 c))) 0.0))
             (s-lo (if (> y 0) (let ((c (nso-sigmoid (aref z (1- y))))) (* c (- 1.0 c))) 0.0))
             (df (/ (- s-hi s-lo) (* py n))))
        (nso-axpy dw df x)
        (when (< y m) (aset dtheta y (- (aref dtheta y) (/ s-hi (* py n)))))
        (when (> y 0) (aset dtheta (1- y) (+ (aref dtheta (1- y)) (/ s-lo (* py n)))))
        (setq rest (cdr rest))))
    (let ((dt 0.0) (dd (make-vector (length d) 0.0)))
      (dotimes (k m) (setq dt (+ dt (aref dtheta k))))
      (dotimes (j (length d))
        (let ((tail 0.0) (k (1+ j)))
          (while (< k m) (setq tail (+ tail (aref dtheta k)) k (1+ k)))
          (aset dd j (* (nso-sigmoid (aref d j)) tail))))
      (nso-axpy dw l2 w)
      (list :dw dw :dt dt :dd dd))))

(defconst nso-score-gtol 5.0e-3
  "Gradient norm at which a fit is finished.

An order of magnitude inside the 0.05 that the probe's gate 0 applies, so a
fit that stops here is comfortably inside the gate it will be judged by rather
than resting against it.  Absolute rather than relative because the gate it
mirrors is absolute; if that one ever moves, this moves with it.")

(defconst nso-score-armijo 1.0e-4
  "Sufficient-decrease constant for the backtracking line search.")

(defun nso-score--copy (model)
  (list :w (copy-sequence (plist-get model :w))
        :k (plist-get model :k)
        :t (plist-get model :t)
        :d (copy-sequence (plist-get model :d))))

(defun nso-score--gnorm2 (g)
  (let ((s (* (plist-get g :dt) (plist-get g :dt))))
    (dolist (part (list (plist-get g :dw) (plist-get g :dd)))
      (dotimes (j (length part)) (setq s (+ s (* (aref part j) (aref part j))))))
    s))

(defun nso-score--descend (model base g step)
  "Set MODEL to BASE minus STEP times G."
  (let ((w (plist-get model :w)) (bw (plist-get base :w)) (gw (plist-get g :dw)))
    (dotimes (j (length w)) (aset w j (- (aref bw j) (* step (aref gw j))))))
  (let ((d (plist-get model :d)) (bd (plist-get base :d)) (gd (plist-get g :dd)))
    (dotimes (j (length d)) (aset d j (- (aref bd j) (* step (aref gd j))))))
  (plist-put model :t (- (plist-get base :t) (* step (plist-get g :dt)))))

(defun nso-score-train (xs ys k &optional steps lr l2 tol)
  "Fit an ordinal head with K levels to XS/YS.

LR is the INITIAL trial step, not the step.  Each iteration backtracks until
the Armijo condition holds, so the fit does not depend on LR being right for
the data -- which is the whole reason this is a line search and not the fixed
step it started as.

That first version diverged on the real features and the design document
records why: the suite fitted at dim 16 and the smoke at dim 64, both of which
a step of 0.5 suits, while the encoder produces dim 1024, where a standardised
feature vector has norm about 32 and the same step overshoots on every
iteration.  It is the fourth time in this repository that a step size chosen
against synthetic data met a real scale and lost, and the third time a
synthetic suite was green while the shipped setting failed.  A search removes
the constant rather than replacing it with a better guess.

STEPS is a CAP, not a count: the fit stops as soon as the gradient norm falls
below TOL (default `nso-score-gtol').  A fixed number of iterations is the
same disease as a fixed step -- a constant chosen against one problem and
carried to another -- and it cost this phase a second void run, where 600
steps converged on synthetic ordinal data and left the real features at a
gradient norm of 0.141 against a gate of 0.05.  Stopping on the quantity the
gate measures, computed on the training split, is the criterion that
transfers.

Returns the model with `:final-loss', `:final-gnorm' and `:steps-taken'
attached; a caller that reports an accuracy without reading the gradient norm
is set up to report a measurement of its optimiser."
  (let* ((model (nso-score-make (length (car xs)) k))
         (steps (or steps 600))
         (step (or lr 0.5))
         (l2 (or l2 0.01))
         (tol2 (let ((v (or tol nso-score-gtol))) (* v v)))
         (i 0)
         (taken 0)
         (g nil))
    (while (< i steps)
      (setq g (nso-score-grad model xs ys l2))
      (setq taken (1+ taken))
      (let ((g2 (nso-score--gnorm2 g)))
        (if (< g2 tol2)
            (setq i steps)
          (let ((l0 (nso-score-loss model xs ys l2))
                (base (nso-score--copy model))
                (tries 0)
                (ok nil))
            ;; Grow the trial step between iterations, so a search that had to
            ;; shrink hard once does not stay small for the rest of the fit.
            (setq step (* 2.0 step))
            (while (and (not ok) (< tries 60))
              (nso-score--descend model base g step)
              (if (<= (nso-score-loss model xs ys l2)
                      (- l0 (* nso-score-armijo step g2)))
                  (setq ok t)
                (setq step (/ step 2.0) tries (1+ tries))))
            (unless ok
              ;; No step along the gradient decreases the loss enough: either
              ;; this is a minimum or the objective is flat here.  Restoring is
              ;; the honest move -- the alternative is silently keeping an
              ;; uphill iterate and reporting whatever it scores.
              (nso-score--descend model base g 0.0)
              (setq i steps))))
        (setq i (1+ i))))
    (setq g (nso-score-grad model xs ys l2))
    (setq model (plist-put model :final-gnorm (sqrt (nso-score--gnorm2 g))))
    (setq model (plist-put model :final-loss (nso-score-loss model xs ys l2)))
    (setq model (plist-put model :steps-taken taken))
    model))

;;; The nominal alternative
;;
;; A plain K-way softmax over the same features: K*dim parameters and no
;; knowledge that the levels are ordered.  It exists so the ordinal claim can
;; be tested rather than asserted, and it is trained through the same entry
;; points with the same step count, learning rate and L2, on the same
;; standardised features.  Giving the two heads different optimiser budgets
;; and then comparing them would measure the budget; that mistake has already
;; been made once here, on the attention pooling in P1, and it invalidated the
;; result rather than biasing it.

(defun nso-score-nominal-make (dim k)
  "Initial K-way softmax parameters over DIM features."
  (let ((w (make-vector k nil)))
    (dotimes (i k) (aset w i (nso-zeros dim)))
    (list :w w :k k)))

(defun nso-score-nominal-logits (model x)
  "Per-level scores of MODEL for X."
  (let* ((w (plist-get model :w))
         (k (length w))
         (out (make-vector k 0.0)))
    (dotimes (i k) (aset out i (nso-dot (aref w i) x)))
    out))

(defun nso-score-nominal-probs (model x)
  "Distribution over levels for X under the nominal MODEL."
  (nso-softmax-vec (nso-score-nominal-logits model x)))

(defun nso-score-nominal-loss (model xs ys l2)
  "Mean cross-entropy of the nominal MODEL over XS/YS, plus L2."
  (let ((n 0) (sum 0.0) (rest ys))
    (dolist (x xs)
      (let ((p (nso-score-nominal-probs model x)))
        (setq sum (+ sum (- (log (max nso-score-prob-floor (aref p (car rest))))))
              n (1+ n) rest (cdr rest))))
    (+ (/ sum n)
       (let ((w (plist-get model :w)) (s 0.0))
         (dotimes (i (length w)) (setq s (+ s (nso-dot (aref w i) (aref w i)))))
         (* 0.5 l2 s)))))

(defun nso-score-nominal-grad (model xs ys l2)
  "Gradient of `nso-score-nominal-loss'.  Returns (:dw VECTOR-OF-VEC)."
  (let* ((w (plist-get model :w))
         (k (length w))
         (dim (length (aref w 0)))
         (n (length xs))
         (dw (make-vector k nil))
         (rest ys))
    (dotimes (i k) (aset dw i (nso-zeros dim)))
    (dolist (x xs)
      (let ((p (nso-score-nominal-probs model x))
            (y (car rest)))
        (dotimes (i k)
          (nso-axpy (aref dw i) (/ (- (aref p i) (if (= i y) 1.0 0.0)) n) x))
        (setq rest (cdr rest))))
    (dotimes (i k) (nso-axpy (aref dw i) l2 (aref w i)))
    (list :dw dw)))

(defun nso-score--nominal-gnorm2 (g)
  (let ((s 0.0) (dw (plist-get g :dw)))
    (dotimes (i (length dw))
      (let ((part (aref dw i)))
        (dotimes (c (length part)) (setq s (+ s (* (aref part c) (aref part c)))))))
    s))

(defun nso-score-nominal-train (xs ys k &optional steps lr l2 tol)
  "Fit a K-way softmax to XS/YS.

The SAME backtracking search as `nso-score-train', and that is not tidiness:
the two heads are compared on held-out MAE, so giving one an optimiser the
other does not have turns that comparison into a measurement of the optimiser.
This repository has already published one result that was exactly that.

Same early stop, too.  On the real features this head reaches a gradient norm
of 5e-07 within 600 iterations and its loss does not move in the 4200 after
that, so a larger cap costs it nothing -- which is exactly why raising the cap
cannot be a way of buying a favourable comparison.  Only the head that has not
converged can gain from it."
  (let* ((model (nso-score-nominal-make (length (car xs)) k))
         (steps (or steps 600))
         (step (or lr 0.5))
         (l2 (or l2 0.01))
         (tol2 (let ((v (or tol nso-score-gtol))) (* v v)))
         (i 0)
         (taken 0)
         (g nil))
    (while (< i steps)
      (setq g (nso-score-nominal-grad model xs ys l2))
      (setq taken (1+ taken))
      (let ((g2 (nso-score--nominal-gnorm2 g)))
        (if (< g2 tol2)
            (setq i steps)
          (let* ((l0 (nso-score-nominal-loss model xs ys l2))
                 (base (let ((w (plist-get model :w))
                             (out (make-vector k nil)))
                         (dotimes (j k) (aset out j (copy-sequence (aref w j))))
                         out))
                 (tries 0)
                 (ok nil))
            (setq step (* 2.0 step))
            (while (and (not ok) (< tries 60))
              (let ((w (plist-get model :w)) (dw (plist-get g :dw)))
                (dotimes (j k)
                  (let ((wj (aref w j)) (bj (aref base j)) (gj (aref dw j)))
                    (dotimes (c (length wj))
                      (aset wj c (- (aref bj c) (* step (aref gj c))))))))
              (if (<= (nso-score-nominal-loss model xs ys l2)
                      (- l0 (* nso-score-armijo step g2)))
                  (setq ok t)
                (setq step (/ step 2.0) tries (1+ tries))))
            (unless ok
              (let ((w (plist-get model :w)))
                (dotimes (j k) (aset w j (copy-sequence (aref base j)))))
              (setq i steps))))
        (setq i (1+ i))))
    (setq g (nso-score-nominal-grad model xs ys l2))
    (setq model (plist-put model :final-gnorm (sqrt (nso-score--nominal-gnorm2 g))))
    (setq model (plist-put model :final-loss (nso-score-nominal-loss model xs ys l2)))
    (setq model (plist-put model :steps-taken taken))
    model))

;;; Ordinal metrics

(defun nso-score-mae (preds ys)
  "Mean absolute error in levels between PREDS and YS.

The metric the ordinal claim lives or dies by.  Exact accuracy cannot tell a
head that answers `fair' for `good' from one that answers `poor', and those
are not the same wrong answer."
  (when (null preds) (error "nso-score: MAE over an empty set"))
  (let ((s 0.0) (n 0) (p preds) (y ys))
    (while p
      (setq s (+ s (abs (- (car p) (car y)))) n (1+ n) p (cdr p) y (cdr y)))
    (/ s n)))

(defun nso-score-constant-mae (ys k)
  "Best MAE reachable by answering the same level for everything.

The baseline the head has to beat to have used the state at all.  Reported as
(:level L :mae M) so the report can say WHICH constant it is against: on a set
balanced across K levels that is the middle one, and on an unbalanced set it
is the median, which is a different number and an easy one to quote wrongly."
  (let ((best nil) (best-mae nil))
    (dotimes (c k)
      (let ((m (nso-score-mae (make-list (length ys) c) ys)))
        (when (or (null best-mae) (< m best-mae)) (setq best c best-mae m))))
    (list :level best :mae best-mae)))

(defun nso-score-report (probs ys)
  "Score a list of distributions PROBS against true levels YS.

Returns (:n :accuracy :mae :mae-median :mae-expected :nll :unimodal), or
(:n 0 :empty t).  An empty subset is reported as empty rather than as a set of
zeroes, for the reason `nso-probe-score' gives."
  (if (null probs)
      (list :n 0 :empty t)
    (let* ((modes (mapcar #'nso-score-mode probs))
           (medians (mapcar #'nso-score-median probs))
           (n (length probs))
           (correct 0)
           (uni 0)
           (nll 0.0)
           (rest ys))
      (dolist (p probs)
        (when (= (nso-score-mode p) (car rest)) (setq correct (1+ correct)))
        (when (nso-score-unimodal-p p) (setq uni (1+ uni)))
        (setq nll (+ nll (- (log (max nso-score-prob-floor (aref p (car rest))))))
              rest (cdr rest)))
      (let ((ci (nso-wilson correct n))
            (exp-err (let ((s 0.0) (r ys))
                       (dolist (p probs)
                         (setq s (+ s (abs (- (nso-score-expected p) (car r))))
                               r (cdr r)))
                       (/ s n))))
        (list :n n :correct correct :accuracy (/ (float correct) n)
              :ci-lo (car ci) :ci-hi (cdr ci)
              :mae (nso-score-mae modes ys)
              :mae-median (nso-score-mae medians ys)
              :mae-expected exp-err
              :nll (/ nll n)
              :unimodal (/ (float uni) n))))))

;;; Calibration, on the ordinal scale
;;
;; Temperature scaling for this head divides the MARGINS, not K independent
;; logits: c_k = sigma((theta_k - f) / T).  Dividing by a positive T preserves
;; the order of the margins, so the cumulative probabilities stay monotone and
;; the distribution stays a distribution -- the parameterisation's invariant
;; survives calibration.
;;
;; But the property section 4 leans on for Noul does NOT carry over, and
;; assuming it did was this file's first mistake.  A binary temperature is
;; monotone in the logit and therefore cannot change the answer, which is
;; exactly why it buys calibration for free.  Here each p_k is the INCREMENT
;; of sigma across an interval, and dividing every margin by T rescales all
;; the intervals at once: which interval collects the most mass depends on
;; where the steep part of sigma falls relative to them, and that moves.  The
;; worked case is in `test/score-test.el' -- margins (-10, -3, 0.1, 0.2)
;; report level 2 at T = 1 and level 4 at T = 5.
;;
;; So for Score, calibration can spend accuracy.  A run that applies a
;; temperature has to report accuracy on both sides of it, and one that quotes
;; P1's "temperature is free" here would be quoting a result about a different
;; head.
;;
;; P3 established that ECE is not measurable on a held-out split this size, so
;; the statistic the Score run reports is P3's binning-free one: the drop in
;; NLL a recalibrator can find, against the band a calibrated-by-construction
;; model produces at the same n.

(defun nso-score-temperature-nll (margins ys temp)
  "Mean NLL of MARGINS divided by TEMP against YS."
  (let ((n 0) (sum 0.0) (rest ys))
    (dolist (z margins)
      (let* ((zt (let ((v (make-vector (length z) 0.0)))
                   (dotimes (k (length z)) (aset v k (/ (aref z k) temp)))
                   v))
             (p (nso-score-probs-from-margins zt)))
        (setq sum (+ sum (- (log (max nso-score-prob-floor (aref p (car rest))))))
              n (1+ n) rest (cdr rest))))
    (/ sum n)))

(defun nso-score-temperature-fit (margins ys &optional lo hi iters)
  "Fit an ordinal temperature to MARGINS/YS by golden-section search.

Same contract as `nso-temperature-fit', including `:saturated': a temperature
resting on a bound is a direction and not a value, and printing it as a
measurement is how a report claims to have found something at 20.00.

Unlike the binary temperature, this one CAN change the reported level; see the
commentary above.  Callers report accuracy before and after."
  (let* ((lo (or lo 0.05))
         (hi (or hi 20.0))
         (iters (or iters 80))
         (lo0 lo) (hi0 hi)
         (phi 0.6180339887498949)
         (c (- hi (* phi (- hi lo))))
         (d (+ lo (* phi (- hi lo))))
         (fc (nso-score-temperature-nll margins ys c))
         (fd (nso-score-temperature-nll margins ys d))
         (i 0))
    (while (< i iters)
      (if (< fc fd)
          (progn (setq hi d d c fd fc)
                 (setq c (- hi (* phi (- hi lo))))
                 (setq fc (nso-score-temperature-nll margins ys c)))
        (setq lo c c d fc fd)
        (setq d (+ lo (* phi (- hi lo))))
        (setq fd (nso-score-temperature-nll margins ys d)))
      (setq i (1+ i)))
    (let ((temp (/ (+ lo hi) 2.0)))
      (list :temperature temp
            :saturated (or (< temp (* 1.001 lo0)) (> temp (* 0.999 hi0)))
            :bounds (cons lo0 hi0)
            :nll-before (nso-score-temperature-nll margins ys 1.0)
            :nll-after (nso-score-temperature-nll margins ys temp)))))

(defun nso-score-oof-margins (items ys groups featurizer k
                                    &optional folds steps lr l2)
  "Out-of-fold margins for ITEMS/YS, folded by GROUPS so no group scores itself.

The ordinal counterpart of `nso-probe-oof-logits', and it exists for the same
reason: a temperature fitted on the training margins is fitted on margins the
head has already separated, and it comes back sharpening.  P1 made that
mistake and then made a second one -- refitting the head per fold while
leaving the feature map fitted on all of train -- so FEATURIZER is called per
fold here too, and the standardiser is built inside the fold.

Returns a list of margin vectors aligned with ITEMS."
  (let* ((ngroups (let ((h (make-hash-table :test 'equal)))
                    (dolist (g groups) (puthash g t h))
                    (hash-table-count h)))
         (folds (max 2 (min (or folds 3) ngroups)))
         (map (nso-probe-fold-map groups folds))
         (n (length items))
         (out (make-vector n nil))
         (f 0))
    (while (< f folds)
      (let ((fit nil) (fy nil) (hold nil) (hi nil) (i 0) (rg groups) (ry ys))
        (dolist (x items)
          (if (= f (gethash (car rg) map))
              (progn (push x hold) (push i hi))
            (push x fit) (push (car ry) fy))
          (setq i (1+ i) rg (cdr rg) ry (cdr ry)))
        (setq fit (nreverse fit) fy (nreverse fy)
              hold (nreverse hold) hi (nreverse hi))
        (unless (and fit hold)
          (error "nso-score-oof-margins: fold %d left a side empty" f))
        (let* ((feat (funcall featurizer fit fy))
               (fx (mapcar feat fit))
               (std (nso-standardizer fx))
               (head (nso-score-train
                      (mapcar (lambda (v) (nso-standardize std v)) fx)
                      fy k (or steps 600) (or lr 0.5) (or l2 0.01)))
               (rt hi))
          (dolist (x hold)
            (aset out (car rt)
                  (nso-score-margins head (nso-standardize std (funcall feat x))))
            (setq rt (cdr rt)))))
      (setq f (1+ f)))
    (append out nil)))

;;; Calibration for the nominal head
;;
;; The head Score actually ships, after §3.3's ordinal form lost gate 2.  Its
;; temperature is the ordinary multiclass one -- p = softmax(z / T) -- and it
;; differs from the ordinal temperature above in the property that matters:
;; dividing every logit by a positive T preserves their ORDER, so the argmax
;; is untouched and the correction cannot spend accuracy.  That is the
;; property §4.2 assumes and the ordinal head broke.  It is a claim about this
;; code, so `test/score-test.el' checks it rather than repeating it.

(defun nso-score-nominal-scale (logits temp)
  "LOGITS with every entry divided by TEMP."
  (mapcar (lambda (z)
            (let ((v (make-vector (length z) 0.0)))
              (dotimes (k (length z)) (aset v k (/ (aref z k) temp)))
              v))
          logits))

(defun nso-score-nominal-temperature-nll (logits ys temp)
  "Mean categorical NLL of LOGITS divided by TEMP against YS."
  (let ((n 0) (sum 0.0) (rest ys))
    (dolist (z logits)
      (let ((zt (make-vector (length z) 0.0)))
        (dotimes (k (length z)) (aset zt k (/ (aref z k) temp)))
        (let ((p (nso-softmax-vec zt)))
          (setq sum (+ sum (- (log (max nso-score-prob-floor (aref p (car rest))))))
                n (1+ n) rest (cdr rest)))))
    (/ sum n)))

(defun nso-score-nominal-temperature-fit (logits ys &optional lo hi iters)
  "Fit a multiclass temperature to LOGITS/YS by golden-section search.
Same contract as `nso-temperature-fit', including `:saturated'."
  (let* ((lo (or lo 0.05))
         (hi (or hi 20.0))
         (iters (or iters 80))
         (lo0 lo) (hi0 hi)
         (phi 0.6180339887498949)
         (c (- hi (* phi (- hi lo))))
         (d (+ lo (* phi (- hi lo))))
         (fc (nso-score-nominal-temperature-nll logits ys c))
         (fd (nso-score-nominal-temperature-nll logits ys d))
         (i 0))
    (while (< i iters)
      (if (< fc fd)
          (progn (setq hi d d c fd fc)
                 (setq c (- hi (* phi (- hi lo))))
                 (setq fc (nso-score-nominal-temperature-nll logits ys c)))
        (setq lo c c d fc fd)
        (setq d (+ lo (* phi (- hi lo))))
        (setq fd (nso-score-nominal-temperature-nll logits ys d)))
      (setq i (1+ i)))
    (let ((temp (/ (+ lo hi) 2.0)))
      (list :temperature temp
            :saturated (or (< temp (* 1.001 lo0)) (> temp (* 0.999 hi0)))
            :bounds (cons lo0 hi0)
            :nll-before (nso-score-nominal-temperature-nll logits ys 1.0)
            :nll-after (nso-score-nominal-temperature-nll logits ys temp)))))

(defun nso-score-nominal-oof-logits (items ys groups featurizer k
                                           &optional folds steps lr l2)
  "Out-of-fold logit vectors for ITEMS/YS, folded by GROUPS.

The nominal counterpart of `nso-score-oof-margins', and it exists for the same
reason: a temperature fitted on logits the head has already separated comes
back sharpening a model about to meet data it has not seen.  The feature map
is rebuilt inside each fold, not around it."
  (let* ((ngroups (let ((h (make-hash-table :test 'equal)))
                    (dolist (g groups) (puthash g t h))
                    (hash-table-count h)))
         (folds (max 2 (min (or folds 3) ngroups)))
         (map (nso-probe-fold-map groups folds))
         (n (length items))
         (out (make-vector n nil))
         (f 0))
    (while (< f folds)
      (let ((fit nil) (fy nil) (hold nil) (hi nil) (i 0) (rg groups) (ry ys))
        (dolist (x items)
          (if (= f (gethash (car rg) map))
              (progn (push x hold) (push i hi))
            (push x fit) (push (car ry) fy))
          (setq i (1+ i) rg (cdr rg) ry (cdr ry)))
        (setq fit (nreverse fit) fy (nreverse fy)
              hold (nreverse hold) hi (nreverse hi))
        (unless (and fit hold)
          (error "nso-score-nominal-oof-logits: fold %d left a side empty" f))
        (let* ((feat (funcall featurizer fit fy))
               (fx (mapcar feat fit))
               (std (nso-standardizer fx))
               (head (nso-score-nominal-train
                      (mapcar (lambda (v) (nso-standardize std v)) fx)
                      fy k (or steps 6000) (or lr 0.5) (or l2 0.01)))
               (rt hi))
          (dolist (x hold)
            (aset out (car rt)
                  (nso-score-nominal-logits head (nso-standardize std (funcall feat x))))
            (setq rt (cdr rt)))))
      (setq f (1+ f)))
    (append out nil)))

(defun nso-score-scale-margins (margins temp)
  "MARGINS with every entry divided by TEMP."
  (mapcar (lambda (z)
            (let ((v (make-vector (length z) 0.0)))
              (dotimes (k (length z)) (aset v k (/ (aref z k) temp)))
              v))
          margins))

;;; Answers

(defun nso-score-answer (model x level-names)
  "A typed Score answer for X, ready for `nso-type-gate'.

The reported level is the mode, because section 2's shared contract requires
the answer to be the argmax of its own distribution.  For an ordinal scale
that is a real constraint rather than a formality -- the median is what
minimises absolute error -- so `nso-score-report' measures both and the run
says what the constraint costs."
  (let* ((p (nso-score-probs model x))
         (best (nso-score-mode p))
         (probs nil)
         (i 0))
    (dolist (name level-names)
      (push (cons name (aref p i)) probs)
      (setq i (1+ i)))
    (list :score (nth best level-names)
          :probabilities (nreverse probs)
          ;; Placeholder, as in `nso-choice-answer'.  Section 2 leaves the
          ;; definition of confidence open and this phase does not close it
          ;; either -- but it is worth recording that max p is a worse readout
          ;; here than it is for Choice: a distribution split evenly between
          ;; two ADJACENT levels is nearly certain about the quantity, and one
          ;; split evenly between the top and bottom level knows nothing, and
          ;; max p gives both the same 0.5.  `test/score-test.el' pins that
          ;; case so the placeholder cannot be mistaken for a decision.
          :confidence (aref p best))))

(provide 'nso-score)
;;; nso-score.el ends here
