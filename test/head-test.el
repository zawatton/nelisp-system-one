;;; head-test.el --- poolings, the Noul head, and the leak control -*- lexical-binding: t; -*-

;;; Commentary:

;; The head is checked here on synthetic features, before the encoder that
;; will feed it real ones exists.  That ordering is not incidental: one
;; encoder pass costs 19 seconds on this hardware, so a bug found after the
;; dataset has been encoded costs hours, and a bug never found at all turns
;; into a P1 conclusion about the donor that is really a conclusion about an
;; arithmetic slip.
;;
;; Every gradient is checked against finite differences.  A transposed index
;; or a dropped softmax term still descends -- more slowly, to a worse place --
;; so a training curve that goes down is not evidence that a gradient is right.
;;
;; The last check is the one that matters most for P1's honesty: the same
;; pipeline run on pure noise must land at chance on held-out data.  A probe
;; that reports 0.9 on random features has a leak, and every later number
;; would inherit it.

;;; Code:

(require 'nso-head)
(require 'nso-metrics)
(require 'nso-stub)
(load (expand-file-name "nso-test-helper.el"
                        (file-name-directory (or load-file-name buffer-file-name))))

(message "== head ==")

;;; --- numerics ------------------------------------------------------------

(nso-t-num "sigmoid(0) is a half" (nso-sigmoid 0.0) 0.5 1e-12)
(nso-t-num "sigmoid saturates high without overflowing"
           (nso-sigmoid 1000.0) 1.0 1e-12)
(nso-t-num "sigmoid saturates low without underflowing to NaN"
           (nso-sigmoid -1000.0) 0.0 1e-12)

(let ((s (nso-softmax-vec (vector 1.0 2.0 3.0)))
      (shifted (nso-softmax-vec (vector 101.0 102.0 103.0))))
  (nso-t-num "softmax sums to one"
             (+ (aref s 0) (aref s 1) (aref s 2)) 1.0 1e-12)
  (nso-t-num "softmax is shift invariant"
             (aref shifted 2) (aref s 2) 1e-12)
  (nso-t-num "softmax survives a large shift without NaN"
             (aref (nso-softmax-vec (vector 900.0 900.0)) 0) 0.5 1e-12))

;;; --- poolings ------------------------------------------------------------

(let ((states (list (vector 1.0 2.0) (vector 3.0 4.0) (vector 5.0 6.0))))
  (nso-t-num "last pooling takes the final position"
             (aref (nso-pool-last states) 0) 5.0 1e-12)
  (nso-t-num "mean pooling averages the positions"
             (aref (nso-pool-mean states) 1) 4.0 1e-12)
  (let ((pa (nso-pool-attn (vector 0.0 0.0) states)))
    (nso-t-num "a zero pool direction makes attention the mean"
               (aref (car pa) 0) 3.0 1e-12)
    (nso-t-num "attention weights sum to one"
               (+ (aref (cdr pa) 0) (aref (cdr pa) 1) (aref (cdr pa) 2))
               1.0 1e-12))
  (let ((pa (nso-pool-attn (vector 10.0 0.0) states)))
    (nso-t "a large pool direction concentrates on the matching position"
           (> (aref (cdr pa) 2) 0.99))))

;;; --- standardiser --------------------------------------------------------

(let* ((rng (nso-rng 3))
       (xs (let (out (i 0))
             (while (< i 50)
               (let ((v (make-vector 4 0.0)))
                 (dotimes (j 4)
                   (aset v j (+ (* 3.0 j) (* 7.0 (nso-rng-float rng)))))
                 (push v out))
               (setq i (1+ i)))
             out))
       (std (nso-standardizer xs))
       (zs (mapcar (lambda (x) (nso-standardize std x)) xs))
       (mu 0.0) (var 0.0))
  (dolist (z zs) (setq mu (+ mu (aref z 2))))
  (setq mu (/ mu (length zs)))
  (dolist (z zs) (setq var (+ var (* (- (aref z 2) mu) (- (aref z 2) mu)))))
  (setq var (/ var (1- (length zs))))
  (nso-t-num "standardised features are centred" mu 0.0 1e-9)
  (nso-t-num "standardised features have unit variance" var 1.0 1e-9))

;;; --- gradients against finite differences --------------------------------
;;
;; The perturbation is 1e-5 and the tolerance a relative 1e-5: tight enough
;; that a dropped term shows, loose enough that the central difference's own
;; truncation error does not.

(defun ht--relerr (got want)
  (/ (abs (- got want)) (max 1.0e-8 (abs want))))

(let* ((rng (nso-rng 11))
       (dim 6) (n 7)
       (xs (let (out (i 0))
             (while (< i n)
               (let ((v (make-vector dim 0.0)))
                 (dotimes (j dim) (aset v j (- (nso-rng-float rng) 0.5)))
                 (push v out))
               (setq i (1+ i)))
             out))
       (ys (let (out (i 0))
             (while (< i n)
               (push (if (< (nso-rng-float rng) 0.5) 0.0 1.0) out)
               (setq i (1+ i)))
             out))
       (head (list :w (let ((w (make-vector dim 0.0)))
                        (dotimes (j dim) (aset w j (- (nso-rng-float rng) 0.5)))
                        w)
                   :b 0.3))
       (l2 0.05)
       (g (nso-head-grad head xs ys l2))
       (eps 1.0e-5)
       (worst 0.0))
  (dotimes (j dim)
    (let* ((w (plist-get head :w))
           (orig (aref w j)))
      (aset w j (+ orig eps))
      (let ((lp (nso-head-loss head xs ys l2)))
        (aset w j (- orig eps))
        (let ((lm (nso-head-loss head xs ys l2)))
          (aset w j orig)
          (setq worst (max worst (ht--relerr (aref (plist-get g :dw) j)
                                             (/ (- lp lm) (* 2 eps)))))))))
  (nso-t-lt "head dL/dw matches finite differences" worst 1.0e-5)
  (let* ((b (plist-get head :b))
         (lp (nso-head-loss (plist-put (copy-sequence head) :b (+ b eps)) xs ys l2))
         (lm (nso-head-loss (plist-put (copy-sequence head) :b (- b eps)) xs ys l2)))
    (nso-t-lt "head dL/db matches finite differences"
              (ht--relerr (plist-get g :db) (/ (- lp lm) (* 2 eps))) 1.0e-5)))

(let* ((rng (nso-rng 23))
       (dim 5) (n 6) (seq 4)
       (xss (let (out (i 0))
              (while (< i n)
                (let (states (p 0))
                  (while (< p seq)
                    (let ((v (make-vector dim 0.0)))
                      (dotimes (j dim) (aset v j (- (nso-rng-float rng) 0.5)))
                      (push v states))
                    (setq p (1+ p)))
                  (push (nreverse states) out))
                (setq i (1+ i)))
              out))
       (ys (let (out (i 0))
             (while (< i n)
               (push (if (< (nso-rng-float rng) 0.5) 0.0 1.0) out)
               (setq i (1+ i)))
             out))
       (model (list :u (let ((u (make-vector dim 0.0)))
                         (dotimes (j dim) (aset u j (- (nso-rng-float rng) 0.5)))
                         u)
                    :w (let ((w (make-vector dim 0.0)))
                         (dotimes (j dim) (aset w j (- (nso-rng-float rng) 0.5)))
                         w)
                    :b -0.2))
       (l2 0.03)
       (g (nso-attn-grad model xss ys l2))
       (eps 1.0e-5))
  (dolist (probe (list (cons :w :dw) (cons :u :du)))
    (let ((worst 0.0)
          (vec (plist-get model (car probe)))
          (want (plist-get g (cdr probe))))
      (dotimes (j dim)
        (let ((orig (aref vec j)))
          (aset vec j (+ orig eps))
          (let ((lp (nso-attn-loss model xss ys l2)))
            (aset vec j (- orig eps))
            (let ((lm (nso-attn-loss model xss ys l2)))
              (aset vec j orig)
              (setq worst (max worst (ht--relerr (aref want j)
                                                 (/ (- lp lm) (* 2 eps)))))))))
      (nso-t-lt (format "attention model dL/d%s matches finite differences"
                        (substring (symbol-name (car probe)) 1))
                worst 1.0e-5)))
  (let* ((b (plist-get model :b))
         (lp (nso-attn-loss (plist-put (copy-sequence model) :b (+ b eps)) xss ys l2))
         (lm (nso-attn-loss (plist-put (copy-sequence model) :b (- b eps)) xss ys l2)))
    (nso-t-lt "attention model dL/db matches finite differences"
              (ht--relerr (plist-get g :db) (/ (- lp lm) (* 2 eps))) 1.0e-5)))

;;; --- temperature scaling -------------------------------------------------

(let* ((rng (nso-rng 31))
       (n 2000)
       (logits nil) (ys nil))
  ;; Labels drawn from sigmoid(z), so z is the calibrated logit by
  ;; construction; then present 4z, which is overconfident by a factor of four
  ;; and should be corrected by a temperature near 4.
  (dotimes (_ n)
    (let* ((z (* 4.0 (- (nso-rng-float rng) 0.5)))
           (y (if (< (nso-rng-float rng) (nso-sigmoid z)) 1.0 0.0)))
      (push (* 4.0 z) logits)
      (push y ys)))
  (let* ((fit (nso-temperature-fit logits ys))
         (temp (plist-get fit :temperature)))
    (message "  fitted temperature %.3f, NLL %.4f -> %.4f"
             temp (plist-get fit :nll-before) (plist-get fit :nll-after))
    (nso-t-num "a 4x-overconfident set is corrected by a temperature near 4"
               temp 4.0 0.6)
    (nso-t-lt "temperature scaling lowers NLL"
              (plist-get fit :nll-after) (plist-get fit :nll-before))
    ;; Monotone, so no answer can move -- this is why it is reached for first.
    (let ((flips 0) (rest ys))
      (dolist (z logits)
        (let ((a (>= (nso-sigmoid z) 0.5))
              (b (>= (nso-sigmoid (/ z temp)) 0.5)))
          (unless (eq a b) (setq flips (1+ flips))))
        (setq rest (cdr rest)))
      (nso-t "temperature scaling changes no answer" (= 0 flips)))
    ;; And the calibration gate must see the improvement.
    (let* ((mk (lambda (tt)
                 (let ((out nil) (rest ys))
                   (dolist (z logits)
                     (let ((p (nso-sigmoid (/ z tt))))
                       (push (nso-sample (list (- 1.0 p) p)
                                         (if (= 1.0 (car rest)) 1 0))
                             out))
                     (setq rest (cdr rest)))
                   (nreverse out))))
           (before (nso-ece (funcall mk 1.0)))
           (after (nso-ece (funcall mk temp))))
      (message "  ECE %.4f -> %.4f (bins %d, n %d)"
               (plist-get before :ece) (plist-get after :ece)
               (plist-get after :bins) (plist-get after :n))
      (nso-t-lt "and the calibration gate sees ECE fall"
                (plist-get after :ece) (plist-get before :ece))
      (nso-t-red "the uncorrected set fails the gate"
                 (nso-calibration-gate (funcall mk 1.0)))
      (nso-t-green "the corrected set passes it"
                   (nso-calibration-gate (funcall mk temp))))))


;;; --- vector scaling, and what it costs -----------------------------------
;;
;; Platt scaling has a slope and an intercept where temperature has only a
;; slope.  The point of these checks is the difference that makes: temperature
;; is monotone through the origin and provably cannot change an answer, while
;; an intercept moves the 0.5 boundary and can.  A calibration step that
;; quietly reclassifies examples is a different kind of object from one that
;; only adjusts confidence, and the suite should say which one it is holding.

(let* ((rng (nso-rng 53))
       (n 2000)
       (logits nil) (ys nil))
  ;; Labels drawn from sigmoid(z - 1.2): the calibrated answer needs BOTH a
  ;; slope and a shift, so temperature alone cannot reach it.
  (dotimes (_ n)
    (let* ((z (* 4.0 (- (nso-rng-float rng) 0.5)))
           (y (if (< (nso-rng-float rng) (nso-sigmoid (- z 1.2))) 1.0 0.0)))
      (push z logits) (push y ys)))
  (let* ((tfit (nso-temperature-fit logits ys))
         (pfit (nso-platt-fit logits ys 3000 0.5))
         (temp (plist-get tfit :temperature)))
    (message "  temperature %.3f -> NLL %.4f;  Platt a=%.3f b=%.3f -> NLL %.4f (%d flips)"
             temp (plist-get tfit :nll-after)
             (plist-get pfit :a) (plist-get pfit :b)
             (plist-get pfit :nll-after) (plist-get pfit :flips))
    (nso-t-num "Platt recovers the shift it was given"
               (plist-get pfit :b) -1.2 0.35)
    (nso-t-lt "and beats temperature where a shift is needed"
              (plist-get pfit :nll-after) (plist-get tfit :nll-after))
    (nso-t-lt "never worse than temperature, which contains it"
              (plist-get pfit :nll-after) (+ 1.0e-6 (plist-get tfit :nll-after)))
    (nso-t-gt "which it pays for by moving answers across the boundary"
              (float (plist-get pfit :flips)) 0.5)
    ;; The contrast that makes the previous line meaningful.
    (let ((flips 0))
      (dolist (z logits)
        (unless (eq (>= (nso-sigmoid z) 0.5)
                    (>= (nso-sigmoid (/ z temp)) 0.5))
          (setq flips (1+ flips))))
      (nso-t "while temperature moves none of them" (= 0 flips)))))

;; Where no shift is needed, the intercept should stay near zero and the two
;; methods should agree -- otherwise the extra parameter is just noise.
(let* ((rng (nso-rng 59))
       (logits nil) (ys nil))
  (dotimes (_ 2000)
    (let* ((z (* 4.0 (- (nso-rng-float rng) 0.5)))
           (y (if (< (nso-rng-float rng) (nso-sigmoid (* 0.25 z))) 1.0 0.0)))
      (push (* 4.0 z) logits) (push y ys)))
  (let ((pfit (nso-platt-fit logits ys 3000 0.5))
        (tfit (nso-temperature-fit logits ys)))
    (message "  no shift needed: Platt b=%.3f, NLL %.4f vs temperature %.4f"
             (plist-get pfit :b) (plist-get pfit :nll-after)
             (plist-get tfit :nll-after))
    (nso-t-lt "with no shift to find, Platt leaves the intercept near zero"
              (abs (plist-get pfit :b)) 0.2)
    ;; The containment, asserted in the direction that can fail.  Temperature
    ;; is Platt with the intercept pinned at zero, so a fitted Platt can never
    ;; be worse; if it is, the optimiser did not converge.  The earlier form
    ;; of this check compared the gap to a tolerance and passed while Platt
    ;; was losing by 0.13.
    (nso-t-lt "Platt is never worse than temperature, here too"
              (plist-get pfit :nll-after) (+ 1.0e-6 (plist-get tfit :nll-after)))))

;;; --- the probe learns, and the leak control ------------------------------

(defun ht--split (xs ys k)
  "Split XS/YS after K items.  Returns (TRX TRY TEX TEY)."
  (let ((trx nil) (try nil) (tex nil) (tey nil) (i 0) (ry ys))
    (dolist (x xs)
      (if (< i k)
          (progn (push x trx) (push (car ry) try))
        (push x tex) (push (car ry) tey))
      (setq ry (cdr ry) i (1+ i)))
    (list (nreverse trx) (nreverse try) (nreverse tex) (nreverse tey))))

(defun ht--accuracy (head xs ys)
  (let ((ok 0) (n 0) (ry ys))
    (dolist (x xs)
      (when (eq (>= (nso-sigmoid (nso-head-logit head x)) 0.5) (= 1.0 (car ry)))
        (setq ok (1+ ok)))
      (setq n (1+ n) ry (cdr ry)))
    (/ (float ok) n)))

;; Separable features: one informative dimension buried in 31 noisy ones.
(let* ((rng (nso-rng 41))
       (dim 32) (n 120)
       (xs nil) (ys nil))
  (dotimes (_ n)
    (let ((y (if (< (nso-rng-float rng) 0.5) 0.0 1.0))
          (v (make-vector dim 0.0)))
      (dotimes (j dim) (aset v j (- (nso-rng-float rng) 0.5)))
      (aset v 7 (+ (aref v 7) (if (= y 1.0) 1.2 -1.2)))
      (push v xs) (push y ys)))
  (let* ((sp (ht--split (nreverse xs) (nreverse ys) 80))
         (std (nso-standardizer (nth 0 sp)))
         (trx (mapcar (lambda (x) (nso-standardize std x)) (nth 0 sp)))
         (tex (mapcar (lambda (x) (nso-standardize std x)) (nth 2 sp)))
         (head (nso-head-train trx (nth 1 sp) 600 0.5 0.02))
         (acc (ht--accuracy head tex (nth 3 sp))))
    (message "  separable features: held-out accuracy %.3f" acc)
    (nso-t-gt "the probe learns a genuinely separable feature" acc 0.75)))

;; Pure noise: the same pipeline, no signal.  Held-out must sit at chance.
;; A leak -- standardising on the full set, or evaluating on training data --
;; shows up here and nowhere else.
(let* ((rng (nso-rng 43))
       (dim 32) (n 120)
       (xs nil) (ys nil))
  (dotimes (_ n)
    (let ((v (make-vector dim 0.0)))
      (dotimes (j dim) (aset v j (- (nso-rng-float rng) 0.5)))
      (push v xs)
      (push (if (< (nso-rng-float rng) 0.5) 0.0 1.0) ys)))
  (let* ((sp (ht--split (nreverse xs) (nreverse ys) 80))
         (std (nso-standardizer (nth 0 sp)))
         (trx (mapcar (lambda (x) (nso-standardize std x)) (nth 0 sp)))
         (tex (mapcar (lambda (x) (nso-standardize std x)) (nth 2 sp)))
         (head (nso-head-train trx (nth 1 sp) 600 0.5 0.02))
         (tr-acc (ht--accuracy head trx (nth 1 sp)))
         (te-acc (ht--accuracy head tex (nth 3 sp))))
    (message "  pure noise: train accuracy %.3f, held-out %.3f" tr-acc te-acc)
    (nso-t-gt "the probe can memorise noise on the training split" tr-acc 0.8)
    (nso-t-lt "but lands at chance on held-out noise -- no leak" te-acc 0.68)))

;;; --- covariance whitening ------------------------------------------------

(defun ht--covariance (zs)
  "Sample covariance matrix of four-dimensional vectors ZS."
  (let ((means (make-vector 4 0.0))
        (cov (make-vector 4 nil))
        (denom (float (max 1 (1- (length zs))))))
    (dotimes (i 4) (aset cov i (make-vector 4 0.0)))
    (dolist (z zs) (dotimes (i 4) (aset means i (+ (aref means i) (aref z i)))))
    (dotimes (i 4) (aset means i (/ (aref means i) (length zs))))
    (dolist (z zs)
      (dotimes (i 4)
        (dotimes (j 4)
          (aset (aref cov i) j
                (+ (aref (aref cov i) j)
                   (* (- (aref z i) (aref means i))
                      (- (aref z j) (aref means j))))))))
    (dotimes (i 4)
      (dotimes (j 4)
        (aset (aref cov i) j (/ (aref (aref cov i) j) denom))))
    cov))

(defun ht--max-diag-error (cov)
  "Largest absolute deviation of COV's diagonal from one."
  (let ((out 0.0))
    (dotimes (i 4) (setq out (max out (abs (- (aref (aref cov i) i) 1.0)))))
    out))

(defun ht--max-offdiag (cov)
  "Largest absolute off-diagonal entry of COV."
  (let ((out 0.0))
    (dotimes (i 4)
      (dotimes (j i) (setq out (max out (abs (aref (aref cov i) j))))))
    out))

(defun ht--basis-orthonormal-p (basis)
  "Whether BASIS has unit norms and mutually orthogonal vectors."
  (let ((ok t))
    (dolist (u basis)
      (unless (< (abs (- (nso-dot u u) 1.0)) 1.0e-7) (setq ok nil)))
    (dolist (u basis)
      (dolist (v basis)
        (unless (or (eq u v) (< (abs (nso-dot u v)) 1.0e-7))
          (setq ok nil))))
    ok))

(let* ((rng (nso-rng 47))
       (xs (let (out)
             (dotimes (_ 44 (nreverse out))
               (let* ((v (make-vector 4 0.0))
                      (a (- (nso-rng-float rng) 0.5))
                      (b (- (nso-rng-float rng) 0.5))
                      (c (- (nso-rng-float rng) 0.5))
                      (d (- (nso-rng-float rng) 0.5)))
                 (aset v 0 (+ a (* 0.3 c)))
                 (aset v 1 (+ (* 0.9 a) (* 0.1 b) (* 0.3 d)))
                 (aset v 2 (+ (* 0.7 a) (* 0.3 b) (* 0.3 c) (* 0.3 d)))
                 (aset v 3 (+ b (* 0.3 c) (* 0.3 d)))
                 (push v out)))))
       (w (nso-whitener xs 0.0001))
       (wz (mapcar (lambda (x) (nso-whiten w x)) xs))
       (cov (ht--covariance wz)))
  (message "whitened covariance: %S" cov)
  (nso-t "whitener basis vectors are orthonormal"
         (ht--basis-orthonormal-p (plist-get w :basis)))
  ;; With shrink near zero, the fitting covariance should be identity.  A
  ;; larger shrink is intentionally partial: it pulls measured eigenvalues
  ;; toward tau, so exact identity is no longer the expected result.
  (nso-t-num "whitened covariance diagonals are one"
              (ht--max-diag-error cov) 0.0 0.05)
  (nso-t-num "whitened covariance off-diagonals are zero"
              (ht--max-offdiag cov) 0.0 0.05)
  (let* ((mean (make-vector 4 0.0))
         (tau 0.0)
         (z (make-vector 4 0.0))
         (w-full (nso-whitener xs 1.0)))
    (dolist (x xs) (dotimes (i 4) (aset mean i (+ (aref mean i) (aref x i)))))
    (dotimes (i 4) (aset mean i (/ (aref mean i) (length xs))))
    (dolist (x xs)
      (dotimes (i 4)
        (aset z i (- (aref x i) (aref mean i))))
      (setq tau (+ tau (/ (nso-dot z z) (* (1- (length xs)) 4.0)))))
    (let ((x (car xs)) (got (nso-whiten w-full (car xs))) (err 0.0))
      (dotimes (i 4)
        (setq err (max err
                       (abs (- (aref got i)
                               (/ (- (aref x i) (aref mean i)) (sqrt tau)))))))
      (message "shrink-one scalar error: %.12g (tau %.12g)" err tau)
      (nso-t-num "shrink one uses the sample mean variance tau"
                  err 0.0 1.0e-9)))
  (let* ((middle (nso-whitener xs 0.1))
         (middle-cov
          (ht--covariance (mapcar (lambda (x) (nso-whiten middle x)) xs)))
         (diag-error (ht--max-diag-error middle-cov))
         (offdiag-error (ht--max-offdiag middle-cov)))
    (message "shrink 0.1 covariance errors: diagonal %.12g, off-diagonal %.12g"
             diag-error offdiag-error)
    ;; These two are REGRESSION PINS, not property assertions, and the names say
  ;; so because the first draft's did not.  At shrink 0.1 on rank-deficient
  ;; data the correct whitener does NOT reach the identity -- the small
  ;; eigenvalues are floored at a*tau, which is what shrinkage is for -- so
  ;; the figures below are what correct code measures, held to 1e-9 so that a
  ;; change in tau moves them.  Mutating tau makes them read 0.0897 and 0.0754
  ;; instead: closer to the identity, because too small a tau shrinks less.
  ;; A test called "diagonals are one" that asserts they equal 0.63 would
  ;; mislead the next reader into thinking this data whitens fully.
  (nso-t-num "shrink 0.1 diagonal error is pinned, not one"
                diag-error 0.6305974002199162 1.0e-9)
    (nso-t-num "shrink 0.1 off-diagonal is pinned, not zero"
                offdiag-error 0.282456995620153 1.0e-9))
  (let* ((partial (nso-whitener xs 0.5))
         (partial-cov
          (ht--covariance (mapcar (lambda (x) (nso-whiten partial x)) xs))))
    (nso-t-gt "large shrinkage deliberately remains partial"
               (ht--max-diag-error partial-cov) 0.1))
  ;; Applying a fit to a distinct set is both the intended leak boundary and
  ;; a check that the stored low-dimensional basis is self-contained.
  (nso-t "a whitener fitted on one set applies to another"
         (vectorp (nso-whiten w (vector 0.2 -0.1 0.4 0.7))))
  (let* ((same (list (vector 1.0 2.0 3.0 4.0) (vector 1.0 2.0 3.0 4.0)))
         (z (nso-whiten (nso-whitener same) (car same))))
    (nso-t "a degenerate whitener produces finite values"
           (let ((ok t)) (dotimes (i 4) (unless (= (aref z i) (aref z i)) (setq ok nil))) ok))))

(nso-t-done "head")

;;; head-test.el ends here
