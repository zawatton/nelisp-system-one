;;; nso-head.el --- poolings, a Noul head, and the temperature that calibrates it -*- lexical-binding: t; -*-

;;; Commentary:

;; Everything P1 does after the frozen encoder has run.  Deliberately separate
;; from the encoder driver, and deliberately free of any dependency on it: the
;; donor table is 570 MiB of gitignored build output, so if the head could only
;; be exercised with the donor present it could not be exercised in `make
;; test' at all.  Here it is checked on synthetic features, gradient by
;; gradient, and the encoder feeds it real ones later.
;;
;; Three poolings, as section 3.2 of the design doc requires, over the
;; per-position hidden states of one example:
;;
;;   last   -- the final position's state.  What a causal decoder is trained to
;;             make informative, since it is the state the next token is read
;;             from.
;;   mean   -- the average over positions.  Ignores the causal asymmetry, which
;;             may be a feature or a defect; that is what P1 measures.
;;   attn   -- a learned pool, scores_i = u . h_i, weights = softmax(scores).
;;             One extra parameter vector, trained jointly with the head.
;;
;; The head is a Noul head: one logit, one sigmoid, binary cross-entropy.  BCE
;; is chosen over anything accuracy-shaped because section 4.2 rests on proper
;; scoring rules, and a head trained on a non-proper objective would make the
;; calibration numbers meaningless before temperature scaling ever ran.

;;; Code:

(require 'nso-types)

(defconst nso-head-exp-clamp 700.0
  "Clamp on exponent arguments.
NeLisp's standalone `exp' returns NaN for large negative arguments and hangs
near -1e6, so every exponent in this file is clamped rather than trusted.")

;;; Small dense linear algebra
;;
;; Plain vectors rather than photon-tensor: a 1024-wide logistic head does not
;; need a tensor library, and not needing one is what keeps `make test'
;; runnable without the sibling repositories.

(defun nso-dot (a b)
  "Inner product of vectors A and B."
  (let ((s 0.0) (n (length a)))
    (dotimes (i n) (setq s (+ s (* (aref a i) (aref b i)))))
    s))

(defun nso-axpy (y alpha x)
  "Y <- Y + ALPHA * X, in place; returns Y."
  (dotimes (i (length y)) (aset y i (+ (aref y i) (* alpha (aref x i)))))
  y)

(defun nso-zeros (n) (make-vector n 0.0))

(defun nso-sigmoid (z)
  "Logistic function, written so neither tail overflows."
  (if (>= z 0.0)
      (/ 1.0 (+ 1.0 (exp (- (min nso-head-exp-clamp z)))))
    (let ((e (exp (max (- nso-head-exp-clamp) z))))
      (/ e (+ 1.0 e)))))

(defun nso-softmax-vec (scores)
  "Softmax of the vector SCORES, shifted by its maximum."
  (let* ((n (length scores))
         (mx -1.0e30)
         (out (nso-zeros n))
         (sum 0.0))
    (dotimes (i n) (setq mx (max mx (aref scores i))))
    (dotimes (i n)
      (let ((e (exp (max (- nso-head-exp-clamp) (- (aref scores i) mx)))))
        (aset out i e)
        (setq sum (+ sum e))))
    (dotimes (i n) (aset out i (/ (aref out i) sum)))
    out))

;;; Poolings
;;
;; STATES is a list of per-position vectors, earliest first.

(defun nso-pool-last (states)
  "The final position's hidden state."
  (copy-sequence (car (last states))))

(defun nso-pool-mean (states)
  "The mean hidden state over positions."
  (let* ((n (length states))
         (out (nso-zeros (length (car states)))))
    (dolist (h states) (nso-axpy out 1.0 h))
    (dotimes (i (length out)) (aset out i (/ (aref out i) n)))
    out))

(defun nso-pool-attn (u states)
  "Pool STATES by softmax(U . h).  Returns (POOLED . WEIGHTS)."
  (let* ((n (length states))
         (scores (nso-zeros n))
         (i 0))
    (dolist (h states)
      (aset scores i (nso-dot u h))
      (setq i (1+ i)))
    (let ((a (nso-softmax-vec scores))
          (out (nso-zeros (length (car states))))
          (j 0))
      (dolist (h states)
        (nso-axpy out (aref a j) h)
        (setq j (1+ j)))
      (cons out a))))

;;; Feature standardisation
;;
;; A 1024-wide probe over a hundred examples is badly conditioned without it,
;; and a badly conditioned probe measures the optimiser rather than the
;; features.  The statistics come from the TRAINING split only; using the
;; held-out split to centre the features is a leak that flatters the result.

(defun nso-standardizer (xs)
  "Per-dimension mean and inverse standard deviation of the vectors XS."
  (let* ((n (length xs))
         (d (length (car xs)))
         (mu (nso-zeros d))
         (iv (nso-zeros d)))
    (dolist (x xs) (nso-axpy mu 1.0 x))
    (dotimes (i d) (aset mu i (/ (aref mu i) n)))
    (dolist (x xs)
      (dotimes (i d)
        (let ((c (- (aref x i) (aref mu i))))
          (aset iv i (+ (aref iv i) (* c c))))))
    (dotimes (i d)
      (let ((sd (sqrt (/ (aref iv i) (max 1 (1- n))))))
        (aset iv i (if (> sd 1.0e-8) (/ 1.0 sd) 0.0))))
    (list :mu mu :inv-sd iv)))

(defun nso-standardize (std x)
  "Apply standardizer STD to vector X, returning a fresh vector."
  (let* ((mu (plist-get std :mu))
         (iv (plist-get std :inv-sd))
         (out (nso-zeros (length x))))
    (dotimes (i (length x))
      (aset out i (* (- (aref x i) (aref mu i)) (aref iv i))))
    out))

;;; Covariance whitening
;;
;; The fitting set is small while the embedding dimension is 1024.  Rather
;; than form or diagonalise a 1024x1024 matrix, this stores the centred fitting
;; vectors and diagonalises their n-by-n Gram matrix.  If q is a Gram
;; eigenvector with eigenvalue lambda, Xq/sqrt((n-1)lambda) is the matching
;; covariance eigenvector.  The remaining dimensions all have the shrunk
;; diagonal eigenvalue, so they need only one scalar multiplier.

(defun nso-whitener (xs &optional shrink)
  "Fit a shrinkage covariance whitener from fitting vectors XS.

SHRINK defaults to 0.1: it leaves most of the measured covariance intact but
keeps the many unobserved directions well-conditioned.  Only XS contributes
statistics; callers must fit it on training vectors and apply it separately.
The result is a plist consumed by `nso-whiten'."
  (let* ((n (length xs))
         (d (length (car xs)))
         (a (or shrink 0.1))
         (beta (- 1.0 a))
         (mu (nso-zeros d))
         (centered nil)
         (denom (float (max 1 (1- n))))
         (tau 0.0))
    (unless (and (>= a 0.0) (<= a 1.0))
      (error "SHRINK must be between zero and one"))
    (dolist (x xs) (nso-axpy mu (/ 1.0 n) x))
    (dolist (x xs)
      (let ((z (nso-zeros d)))
        (nso-axpy z 1.0 x)
        (nso-axpy z -1.0 mu)
        (push z centered)))
    (setq centered (nreverse centered))
    (dolist (z centered)
      (setq tau (+ tau (/ (nso-dot z z) (* denom d)))))
    ;; The sum above already averages over the n-1 covariance denominator;
    ;; dividing by n again would make the shrinkage target n times too small.
    (let* ((gram (make-vector n nil))
           (q (make-vector n nil))
           (vals (nso-zeros n))
           (floor (max 1.0e-12 (* 1.0e-12 (max 1.0 tau))))
           (base (+ (* a tau) floor))
           (basis nil) (factors nil))
      ;; Gram = X'X/(n-1).  Jacobi rotations are adequate for n < 100 and
      ;; avoid allocating a dense matrix in the embedding dimension.
      (dotimes (i n)
        (aset gram i (nso-zeros n))
        (aset q i (nso-zeros n))
        (aset (aref q i) i 1.0)
        (dotimes (j n)
          (aset (aref gram i) j (/ (nso-dot (nth i centered) (nth j centered)) denom))))
      ;; A sweep is not enough for a highly correlated Gram matrix; allow
      ;; several Jacobi sweeps while still keeping the work in the fitting-set
      ;; dimension rather than the 1024-wide embedding dimension.
      (dotimes (iteration (* 100 (max 1 n)))
        (let ((p 0) (r 1) (mx 0.0))
          (ignore iteration)
          (dotimes (i n)
            (dotimes (j i)
              (when (> (abs (aref (aref gram i) j)) mx)
                (setq mx (abs (aref (aref gram i) j)) p j r i))))
          (when (> mx 1.0e-10)
            (let* ((gpp (aref (aref gram p) p))
                   (grr (aref (aref gram r) r))
                   (gpr (aref (aref gram r) p))
                   (theta (* 0.5 (atan (/ (* 2.0 gpr) (- grr gpp)))))
                   (co (cos theta)) (si (sin theta)))
              (dotimes (k n)
                (unless (or (= k p) (= k r))
                  (let ((x (aref (aref gram k) p)) (y (aref (aref gram k) r)))
                    (aset (aref gram k) p (- (* co x) (* si y)))
                    (aset (aref gram p) k (aref (aref gram k) p))
                    (aset (aref gram k) r (+ (* si x) (* co y)))
                    (aset (aref gram r) k (aref (aref gram k) r)))))
              (aset (aref gram p) p (+ (* co co gpp) (* -2.0 co si gpr) (* si si grr)))
              (aset (aref gram r) r (+ (* si si gpp) (* 2.0 co si gpr) (* co co grr)))
              (aset (aref gram p) r 0.0)
              (aset (aref gram r) p 0.0)
              (dotimes (k n)
                (let ((x (aref (aref q k) p)) (y (aref (aref q k) r)))
                  (aset (aref q k) p (- (* co x) (* si y)))
                  (aset (aref q k) r (+ (* si x) (* co y)))))))))
      (dotimes (k n)
        (let ((lambda (max 0.0 (aref (aref gram k) k))))
          (aset vals k lambda)
          ;; Eigenvalues below this relative floor are numerical remnants of
          ;; the singular sample covariance, not observed directions.
          (when (> lambda (* 1.0e-8 (max 1.0 tau)))
            (let ((u (nso-zeros d)))
              (dotimes (i n) (nso-axpy u (aref (aref q i) k) (nth i centered)))
              (let ((scale (/ 1.0 (sqrt (* denom lambda)))))
                (dotimes (i d) (aset u i (* scale (aref u i)))))
              (push u basis)
              (push (cons u (/ 1.0 (sqrt (+ (* beta lambda) (* a tau) floor)))) factors)))))
      (list :mu mu :basis basis :factors factors :outside (/ 1.0 (sqrt base))))))

(defun nso-whiten (w x)
  "Apply whitener W to one vector X, returning a fresh vector."
  (let* ((mu (plist-get w :mu))
         (out (nso-zeros (length x)))
         (z (nso-zeros (length x)))
         (outside (plist-get w :outside)))
    (dotimes (i (length x)) (aset z i (- (aref x i) (aref mu i))))
    (nso-axpy out outside z)
    (dolist (pair (plist-get w :factors))
      (let ((u (car pair)) (delta (- (cdr pair) outside)))
        (nso-axpy out (* delta (nso-dot u z)) u)))
    out))

;;; The Noul head

(defun nso-head-make (dim)
  "A zero-initialised head over DIM features."
  (list :w (nso-zeros dim) :b 0.0))

(defun nso-head-logit (head x)
  "Pre-sigmoid score of HEAD on feature vector X."
  (+ (nso-dot (plist-get head :w) x) (plist-get head :b)))

(defun nso-bce (p y)
  "Binary cross-entropy of probability P against label Y (0.0 or 1.0)."
  (let ((q (min (- 1.0 1.0e-12) (max 1.0e-12 p))))
    (- (+ (* y (log q)) (* (- 1.0 y) (log (- 1.0 q)))))))

(defun nso-head-loss (head xs ys l2)
  "Mean BCE of HEAD over XS/YS plus L2 * |w|^2 / 2."
  (let ((n (length xs)) (sum 0.0) (rest ys))
    (dolist (x xs)
      (setq sum (+ sum (nso-bce (nso-sigmoid (nso-head-logit head x)) (car rest))))
      (setq rest (cdr rest)))
    (+ (/ sum n)
       (let ((w (plist-get head :w)))
         (* 0.5 l2 (nso-dot w w))))))

(defun nso-head-grad (head xs ys l2)
  "Gradient of `nso-head-loss'.  Returns (:dw VEC :db FLOAT)."
  (let* ((n (length xs))
         (w (plist-get head :w))
         (dw (nso-zeros (length w)))
         (db 0.0)
         (rest ys))
    (dolist (x xs)
      (let ((g (- (nso-sigmoid (nso-head-logit head x)) (car rest))))
        (nso-axpy dw (/ g n) x)
        (setq db (+ db (/ g n))))
      (setq rest (cdr rest)))
    (nso-axpy dw l2 w)
    (list :dw dw :db db)))

(defun nso-head-train (xs ys &optional steps lr l2)
  "Fit a head to XS/YS by gradient descent.  Returns the head."
  (let* ((head (nso-head-make (length (car xs))))
         (steps (or steps 400))
         (lr (or lr 0.5))
         (l2 (or l2 0.01))
         (i 0))
    (while (< i steps)
      (let ((g (nso-head-grad head xs ys l2)))
        (nso-axpy (plist-get head :w) (- lr) (plist-get g :dw))
        (setq head (plist-put head :b (- (plist-get head :b)
                                         (* lr (plist-get g :db))))))
      (setq i (1+ i)))
    head))

;;; Joint attention pool + head
;;
;; The pool has parameters, so it cannot be fitted offline the way `last' and
;; `mean' can.  It trains against the same BCE, through the softmax.

(defun nso-attn-model-make (dim)
  "Zero head with a small non-zero pool direction over DIM features.
A zero U makes every softmax weight identical and every gradient through it
zero, so the pool would never move; the asymmetry has to be seeded."
  (list :u (let ((u (nso-zeros dim)) (i 0))
             (while (< i dim)
               (aset u i (* 1.0e-3 (if (= 0 (mod i 2)) 1.0 -1.0)))
               (setq i (1+ i)))
             u)
        :w (nso-zeros dim) :b 0.0))

(defun nso-attn-forward (model states)
  "Run MODEL over one example's STATES.  Returns (:p P :pooled V :a WEIGHTS)."
  (let* ((pa (nso-pool-attn (plist-get model :u) states))
         (pooled (car pa))
         (z (+ (nso-dot (plist-get model :w) pooled) (plist-get model :b))))
    (list :p (nso-sigmoid z) :pooled pooled :a (cdr pa))))

(defun nso-attn-loss (model xss ys l2)
  "Mean BCE of MODEL over the list of state-lists XSS."
  (let ((n (length xss)) (sum 0.0) (rest ys))
    (dolist (states xss)
      (setq sum (+ sum (nso-bce (plist-get (nso-attn-forward model states) :p)
                                (car rest))))
      (setq rest (cdr rest)))
    (+ (/ sum n)
       (let ((w (plist-get model :w)) (u (plist-get model :u)))
         (* 0.5 l2 (+ (nso-dot w w) (nso-dot u u)))))))

(defun nso-attn-grad (model xss ys l2)
  "Gradient of `nso-attn-loss'.  Returns (:du VEC :dw VEC :db FLOAT).

Written against reused scratch rather than by calling `nso-attn-forward' per
example, which is the same arithmetic in the same order and a great deal less
garbage.  The allocating version cost one softmax pair and one DIM-wide pooled
vector per example per step: at 168 examples, 400 steps and dim 1024 that is
67,200 fresh kilobyte vectors per training run, and the training runs four
times per configuration.  Measured on this hardware, that put one attention
configuration at 53 minutes.

The order of every accumulation is unchanged, so the result is bit-identical
to the allocating version and `test/head-test.el' checks exactly that -- an
optimisation of a gradient is worth nothing if it is a different gradient."
  (let* ((dim (length (plist-get model :w)))
         (n (length xss))
         (u (plist-get model :u))
         (w (plist-get model :w))
         (b (plist-get model :b))
         (du (nso-zeros dim))
         (dw (nso-zeros dim))
         (db 0.0)
         (maxseq (let ((m 0)) (dolist (s xss) (setq m (max m (length s)))) m))
         (scores (make-vector (max 1 maxseq) 0.0))
         (aw (make-vector (max 1 maxseq) 0.0))
         (da (make-vector (max 1 maxseq) 0.0))
         (pooled (nso-zeros dim))
         (rest ys))
    (dolist (states xss)
      (let ((m (length states)) (i 0) (mx -1.0e30) (sum 0.0) (dot-sum 0.0)
            z p g)
        (dolist (h states) (aset scores i (nso-dot u h)) (setq i (1+ i)))
        (dotimes (j m) (setq mx (max mx (aref scores j))))
        (dotimes (j m)
          (let ((e (exp (max (- nso-head-exp-clamp) (- (aref scores j) mx)))))
            (aset aw j e)
            (setq sum (+ sum e))))
        (dotimes (j m) (aset aw j (/ (aref aw j) sum)))
        (dotimes (j dim) (aset pooled j 0.0))
        (setq i 0)
        (dolist (h states) (nso-axpy pooled (aref aw i) h) (setq i (1+ i)))
        (setq z (+ (nso-dot w pooled) b))
        (setq p (nso-sigmoid z))
        (setq g (/ (- p (car rest)) n))
        (nso-axpy dw g pooled)
        (setq db (+ db g))
        (setq i 0)
        (dolist (h states) (aset da i (* g (nso-dot w h))) (setq i (1+ i)))
        (dotimes (j m) (setq dot-sum (+ dot-sum (* (aref aw j) (aref da j)))))
        (setq i 0)
        (dolist (h states)
          (nso-axpy du (* (aref aw i) (- (aref da i) dot-sum)) h)
          (setq i (1+ i))))
      (setq rest (cdr rest)))
    (nso-axpy dw l2 w)
    (nso-axpy du l2 u)
    (list :du du :dw dw :db db)))

(defun nso-attn-train (xss ys &optional steps lr l2)
  "Fit an attention pool and head jointly to XSS/YS."
  (let* ((model (nso-attn-model-make (length (car (car xss)))))
         (steps (or steps 400))
         (lr (or lr 0.5))
         (l2 (or l2 0.01))
         (i 0))
    (while (< i steps)
      (let ((g (nso-attn-grad model xss ys l2)))
        (nso-axpy (plist-get model :w) (- lr) (plist-get g :dw))
        (nso-axpy (plist-get model :u) (- lr) (plist-get g :du))
        (setq model (plist-put model :b (- (plist-get model :b)
                                           (* lr (plist-get g :db))))))
      (setq i (1+ i)))
    model))

(defun nso-attn-standardizer (xss)
  "Per-dimension statistics over every position of every example in XSS.

`nso-attn-train' was the one path in this repository fitting a head on raw
features: `last' and `mean' hand their pooled vectors to
`nso-probe-fit-and-score', which standardises before fitting, while the joint
model trained straight on hidden states whose RMS is about 3.  With one
learning rate shared between the paths, that difference decided whether the
joint model converged, and it decided it differently at the two depths --
which is what produced P1's unexplained 0.500.  Measured: at 200 steps the
joint model reaches 0.976 training accuracy at mid depth with lr 0.1 and
0.500 at 0.5, 0.02 and 0.005, with losses up to 19 where it diverges."
  (nso-standardizer (apply #'append xss)))

(defun nso-attn-standardize (std xss)
  "Apply STD to every position of every example in XSS."
  (mapcar (lambda (states)
            (mapcar (lambda (h) (nso-standardize std h)) states))
          xss))

;;; Temperature scaling
;;
;; One scalar, fitted on a held-out split by minimising NLL.  It is monotone,
;; so it cannot change a single answer -- only the confidence attached to it.
;; That is exactly why it is the first thing to reach for: it buys calibration
;; without spending accuracy.

(defun nso-temperature-nll (logits ys temp)
  "Mean BCE of LOGITS divided by TEMP against YS."
  (let ((n (length logits)) (sum 0.0) (rest ys))
    (dolist (z logits)
      (setq sum (+ sum (nso-bce (nso-sigmoid (/ z temp)) (car rest))))
      (setq rest (cdr rest)))
    (/ sum n)))

(defun nso-temperature-fit (logits ys &optional lo hi iters)
  "Fit a temperature to LOGITS/YS by golden-section search on [LO,HI].
Returns (:temperature T :nll-before N0 :nll-after N1)."
  (let* ((lo (or lo 0.05))
         (hi (or hi 20.0))
         (iters (or iters 80))
         (lo0 lo) (hi0 hi)
         (phi 0.6180339887498949)
         (c (- hi (* phi (- hi lo))))
         (d (+ lo (* phi (- hi lo))))
         (fc (nso-temperature-nll logits ys c))
         (fd (nso-temperature-nll logits ys d))
         (i 0))
    (while (< i iters)
      (if (< fc fd)
          (progn (setq hi d d c fd fc)
                 (setq c (- hi (* phi (- hi lo))))
                 (setq fc (nso-temperature-nll logits ys c)))
        (setq lo c c d fc fd)
        (setq d (+ lo (* phi (- hi lo))))
        (setq fd (nso-temperature-nll logits ys d)))
      (setq i (1+ i)))
    (let ((temp (/ (+ lo hi) 2.0)))
      (list :temperature temp
            ;; A temperature resting on the edge of the search range is a
            ;; direction, not a value: the optimum is somewhere past the
            ;; bound and the number reported is the bound.  Callers that
            ;; print it should say so rather than let 20.00 read as a
            ;; measurement.
            :saturated (or (< temp (* 1.001 lo0)) (> temp (* 0.999 hi0)))
            :bounds (cons lo0 hi0)
            :nll-before (nso-temperature-nll logits ys 1.0)
            :nll-after (nso-temperature-nll logits ys temp)))))

(defun nso-platt-nll (logits ys a b)
  "Mean BCE of sigmoid(A*z + B) over LOGITS against YS."
  (let ((n (length logits)) (sum 0.0) (rest ys))
    (dolist (z logits)
      (setq sum (+ sum (nso-bce (nso-sigmoid (+ (* a z) b)) (car rest))))
      (setq rest (cdr rest)))
    (/ sum n)))

(defun nso-platt-fit (logits ys &optional iters _lr)
  "Fit a two-parameter calibration sigmoid(A*z + B) to LOGITS/YS.

Vector scaling, in the binary case: a slope and an intercept where temperature
scaling has only a slope.  The extra parameter is not free.  Temperature is
monotone THROUGH THE ORIGIN, so it cannot move the 0.5 boundary and cannot
change a single answer; an intercept moves the boundary and can.  So this is
judged on accuracy as well as on calibration, and the result reports the flips
to make that possible.

Fitted by Newton rather than gradient descent, which removes the step size
instead of tuning it.  Two earlier attempts here failed on exactly that: from
the identity, descent at lr 0.5 finished at NLL 0.811 where temperature
reached 0.679; started at temperature's own solution it reached 0.915, worse
still, because the logits ran to 8 and a fixed step of that size oscillates.
The problem is two-dimensional and convex, so the Hessian is a 2x2 and there
is no reason to guess a step at all.  A fitted Platt can then never be worse
than temperature, which it contains, and the suite asserts that in the
direction that can fail.

Returns (:a A :b B :nll-before N0 :nll-after N1 :flips F)."
  (let* ((iters (or iters 50))
         (n (length logits))
         (a 1.0) (b 0.0)
         (i 0))
    (ignore _lr)
    (while (< i iters)
      (let ((ga 0.0) (gb 0.0) (haa 0.0) (hab 0.0) (hbb 0.0) (rest ys))
        (dolist (z logits)
          (let* ((p (nso-sigmoid (+ (* a z) b)))
                 (g (- p (car rest)))
                 (w (* p (- 1.0 p))))
            (setq ga (+ ga (* g z)) gb (+ gb g)
                  haa (+ haa (* w z z)) hab (+ hab (* w z)) hbb (+ hbb w)))
          (setq rest (cdr rest)))
        (setq ga (/ ga n) gb (/ gb n)
              haa (/ haa n) hab (/ hab n) hbb (/ hbb n))
        ;; A ridge keeps the solve well posed when the weights collapse, which
        ;; they do once the fit is confident everywhere.
        (setq haa (+ haa 1.0e-8) hbb (+ hbb 1.0e-8))
        (let ((det (- (* haa hbb) (* hab hab))))
          (when (< (abs det) 1.0e-14) (setq det 1.0e-14))
          (let* ((da (/ (- (* hbb ga) (* hab gb)) det))
                 (db (/ (- (* haa gb) (* hab ga)) det))
                 ;; Backtracking, because an undamped Newton step is not safe
                 ;; here.  Once the fit saturates, p(1-p) collapses, the
                 ;; Hessian is nearly singular and the solve returns something
                 ;; enormous: the undamped version reached b = 4.2e5 and an
                 ;; NLL of 15.9 on a set where temperature managed 0.679.
                 ;; Halving until the objective actually falls makes each step
                 ;; a decrease by construction.
                 (base (nso-platt-nll logits ys a b))
                 (t* 1.0)
                 (tries 0)
                 (ok nil))
            (while (and (not ok) (< tries 40))
              (let ((na (- a (* t* da))) (nb (- b (* t* db))))
                (if (<= (nso-platt-nll logits ys na nb) base)
                    (setq a na b nb ok t)
                  (setq t* (* 0.5 t*) tries (1+ tries)))))
            (when (or (not ok)
                      (< (* t* (+ (abs da) (abs db))) 1.0e-12))
              (setq i iters)))))
      (setq i (1+ i)))
    (let ((flips 0))
      (dolist (z logits)
        (unless (eq (>= z 0.0) (>= (+ (* a z) b) 0.0))
          (setq flips (1+ flips))))
      (list :a a :b b :flips flips
            :nll-before (nso-platt-nll logits ys 1.0 0.0)
            :nll-after (nso-platt-nll logits ys a b)))))

;;; Answers

(defun nso-noul-answer (p)
  "Build a Noul answer plist from P(yes).
Confidence is left as the distance from an even split, scaled to [0,1].  This
is a placeholder with a named shape, not a calibrated quantity -- section 2 of
the design doc leaves the definition open and P1 does not close it."
  (list :p-yes p :confidence (abs (- (* 2.0 p) 1.0))))

(provide 'nso-head)
;;; nso-head.el ends here
