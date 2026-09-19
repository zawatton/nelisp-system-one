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

(nso-t-done "head")

;;; head-test.el ends here
